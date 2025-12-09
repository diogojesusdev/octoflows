import asyncio
import base64
from dataclasses import dataclass, field
from itertools import groupby
import json
import random
import time
import aiohttp
import cloudpickle
import os
from typing import Any

from src.dag import dag
from src.task_worker_resource_configuration import TaskWorkerResourceConfiguration
from src.utils.logger import create_logger
from src.workers.worker import Worker
from src.utils.utils import calculate_data_structure_size_bytes

logger = create_logger(__name__)

class DockerWorker(Worker):
    @dataclass
    class Config(Worker.Config):
        external_docker_gateway_addresses: list[tuple[str, int]] = field(default_factory=list)
        internal_docker_gateway_address = ("host.docker.internal", 5000)
        container_monitoring_addresses: list[tuple[str, int]] = field(default_factory=list)
      
        def create_instance(self) -> "DockerWorker": 
            super().create_instance()
            return DockerWorker(self)

    docker_config: Config

    """
    Invokes workers by calling a Flask web server with the serialized subsubdag
    Waits for the completion of all workers
    """
    def __init__(self, config: Config):
        super().__init__(config)
        self.docker_config = config
        # self.ARTIFICIAL_NETWORK_LATENCY_S = 0.030 # ~15 ms each way (request, response)
        self.ARTIFICIAL_NETWORK_LATENCY_S = 0 # ~15 ms each way (request, response)
        self.MAX_DAG_SIZE_BYTES = 300 * 1024 # 300KB
        self.MAX_DAG_CACHED_RESULTS_BYTES = 250 * 1024 # 250KB
        # On linux, docker containers don't have access to host.docker.internal. They can just call localhost, on Windows they have to use host.docker.internal
        self.is_docker_host_linux = os.getenv("HOST_OS") == "linux"

    async def _simulate_network_latency(self) -> None:
        await asyncio.sleep(self.ARTIFICIAL_NETWORK_LATENCY_S)

    async def delegate(self, subdags: list[dag.SubDAG], fulldag: dag.FullDAG, called_by_worker: bool = True):
        '''
        Each invocation is done inside a new Coroutine without blocking the owner Thread
        All HTTP requests are executed in parallel, and the function only returns once all requests are completed
        '''
        from src.storage.metadata.metrics_types import WorkerStartupMetrics
        if len(subdags) == 0: 
            raise Exception("DockerWorker.delegate() received an empty list of subdags to delegate!")
        
        subdags.sort(key=lambda sd: sd.root_node.worker_config.worker_id or "", reverse=True)
        
        relevant_cached_results: dict[str, Any] = {}
        aggregated_results_size_bytes = 0
        for subdag in subdags:
            rn = subdag.root_node
            # Go through all tasks and add as much results as possible witohut exceeding {MAX_DAG_CACHED_RESULTS_BYTES}
            for utask in rn.upstream_nodes:
                if utask.cached_result is None: continue
                serialized_result = cloudpickle.dumps(utask.cached_result.result)
                utask_result_size = calculate_data_structure_size_bytes(serialized_result)
                if aggregated_results_size_bytes + utask_result_size < self.MAX_DAG_CACHED_RESULTS_BYTES:
                    aggregated_results_size_bytes += utask_result_size
                    relevant_cached_results[utask.id.get_full_id()] = serialized_result
                
        # Separate tasks with None worker_id from those with specific worker_ids
        tasks_with_worker_id = []
        tasks_without_worker_id = []
        
        for subdag in subdags:
            worker_id = subdag.root_node.worker_config.worker_id
            if worker_id is None:
                tasks_without_worker_id.append(subdag)
            else:
                tasks_with_worker_id.append(subdag)
        
        # Group tasks with specific worker_ids
        tasks_grouped_by_id = {
            worker_id: list(tasks)
            for worker_id, tasks in groupby(tasks_with_worker_id, key=lambda sd: sd.root_node.worker_config.worker_id)
        }
        
        http_tasks = []
        async def make_worker_request(worker_id, worker_subdags):
            _worker_subdags: list[dag.SubDAG] = worker_subdags
            targetWorkerResourcesConfig = _worker_subdags[0].root_node.worker_config
            gateway_address = self.docker_config.internal_docker_gateway_address if called_by_worker and not self.is_docker_host_linux else random.choice(self.docker_config.external_docker_gateway_addresses)

            logger.info(f"Invoking docker gateway ({gateway_address[0]}:{gateway_address[1]}) | CPUs: {targetWorkerResourcesConfig.cpus} | Memory: {targetWorkerResourcesConfig.memory_mb} | Worker ID: {worker_id} | Root Tasks: {[subdag.root_node.id.get_full_id() for subdag in _worker_subdags]}")
            await self.metadata_storage.store_invoker_worker_startup_metrics(
                WorkerStartupMetrics(
                    master_dag_id=_worker_subdags[0].master_dag_id,
                    start_time_ms=time.time() * 1000,
                    resource_configuration=targetWorkerResourcesConfig,
                    state=None,
                    end_time_ms=None,
                    initial_task_ids=[subdag.root_node.id.get_full_id() for subdag in _worker_subdags]
                ),
                task_ids=[subdag.root_node.id.get_full_id() for subdag in _worker_subdags]
            )

            fulldag_size_below_threshold = False
            if self.docker_config.optimized_dag:
                fulldag_size = calculate_data_structure_size_bytes(self.docker_config.optimized_dag)
                fulldag_size_below_threshold = fulldag_size < self.MAX_DAG_SIZE_BYTES
            
            async with aiohttp.ClientSession() as session:
                async with await session.post(
                    f"http://{gateway_address[0]}:{gateway_address[1]}" + "/job",
                    data=json.dumps({
                        "resource_configuration": base64.b64encode(cloudpickle.dumps(targetWorkerResourcesConfig)).decode('utf-8'),
                        "dag_id": _worker_subdags[0].master_dag_id,
                        # if dag size is below 200KB, send the dag in the invocation, else, send the ID and the worker has to fetch it from storage
                        "fulldag": self.docker_config.optimized_dag if self.docker_config.optimized_dag and fulldag_size_below_threshold else None,
                        # "fulldag": None,
                        "task_ids": base64.b64encode(cloudpickle.dumps([subdag.root_node.id for subdag in _worker_subdags])).decode('utf-8'),
                        "relevant_cached_results": base64.b64encode(cloudpickle.dumps(relevant_cached_results)).decode('utf-8'),
                        "config": base64.b64encode(cloudpickle.dumps(self.docker_config)).decode('utf-8'),
                    }),
                    headers={'Content-Type': 'application/json'}
                ) as response:
                    if response.status != 202:
                        text = await response.text()
                        raise Exception(f"Failed to invoke worker: {text}")
                    return response.status
        
        await self._simulate_network_latency()

        # Create a task for each worker_id (grouped requests)
        for worker_id, worker_subdags in tasks_grouped_by_id.items():
            http_tasks.append(make_worker_request(worker_id, worker_subdags))
        
        # Create individual tasks for each subdag with worker_id = None
        for subdag in tasks_without_worker_id:
            http_tasks.append(make_worker_request(None, [subdag]))
        
        # Wait for all HTTP requests to complete
        await asyncio.gather(*http_tasks)

    async def warmup(self, dag_id: str, resource_configurations: list[TaskWorkerResourceConfiguration]):
        """
        Sends a warmup request to the worker with the given resource configuration.
        """
        await self._simulate_network_latency()

        gateway_address = self.docker_config.internal_docker_gateway_address if not self.is_docker_host_linux else random.choice(self.docker_config.external_docker_gateway_addresses)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    # because only docker workers will make warmup requests to each other (and never the client requesting a warmup)
                    f"http://{gateway_address}:{gateway_address[1]}" + "/warmup",
                    data=json.dumps({
                        "dag_id": dag_id,
                        "resource_configurations": base64.b64encode(cloudpickle.dumps(resource_configurations)).decode('utf-8')
                    }),
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status != 202:
                        logger.error(f"Warmup request to {gateway_address}/warmup failed with status {response.status}")
                        response_text = await response.text()
                        logger.error(f"Response: {response_text}")
                    else:
                        logger.info(f"Warmup request to {gateway_address}/warmup completed successfully")
        except Exception as e:
            logger.error(f"Error during warmup request: {str(e)}")
            raise