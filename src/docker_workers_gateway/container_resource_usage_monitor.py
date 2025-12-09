import asyncio
from collections import defaultdict
from typing import Any, Optional, Tuple
import time

import aiohttp
from src.storage.metadata.metrics_types import DAGResourceUsageMetrics

SAMPLE_INTERVAL_S = 0.5

class DockerContainerUsageMonitor:
    """
    Monitors allocated CPU and memory of Docker containers whose names contain a given DAG ID.

    CPU seconds are based on *allocated CPUs* (not actual usage) to mimic reserved resources.
    """

    _dag_data: dict[str, dict[str, Any]] = {}
    _tasks: dict[str, asyncio.Task] = {}
    _container_limits_cache: dict[str, Tuple[int, float]] = {}  # {container_id: (mem_limit_bytes, num_cpus)}
    _container_monitoring_addresses: list[tuple[str, int]] = []

    @staticmethod
    async def _get_containers(session: aiohttp.ClientSession) -> list[dict]:
        async def _fetch_from_host(host: str) -> list[dict]:
            try:
                async with session.get(f"{host}/containers/json") as resp:
                    resp.raise_for_status()
                    containers = await resp.json()
                    for c in containers:
                        c["DockerHost"] = host
                    return containers
            except Exception as e:
                print(f"[WARN] Failed to list containers from {host}: {e}")
                return []

        tasks = [
            _fetch_from_host(f"http://{host}:{port}")
            for host, port in DockerContainerUsageMonitor._container_monitoring_addresses
        ]
        list_of_containers = await asyncio.gather(*tasks)

        return [c for containers in list_of_containers for c in containers]

    @staticmethod
    async def _get_container_limits(
        session: aiohttp.ClientSession, container_id: str, docker_host: str
    ) -> Optional[Tuple[int, float]]:
        """
        Get (memory_limit_bytes, allocated_cpus) for a container.
        Uses a cache if container was removed before inspection.
        """
        try:
            async with session.get(f"{docker_host}/containers/{container_id}/json") as resp:
                if resp.status == 404:
                    return DockerContainerUsageMonitor._container_limits_cache.get(container_id)
                resp.raise_for_status()
                data = await resp.json()
                mem_limit = data["HostConfig"]["Memory"]
                cpu_quota = data["HostConfig"]["CpuQuota"]
                cpu_period = data["HostConfig"]["CpuPeriod"]
                num_cpus = cpu_quota / cpu_period if cpu_quota > 0 else 1
                # Cache for later
                DockerContainerUsageMonitor._container_limits_cache[container_id] = (mem_limit, num_cpus)
                return mem_limit, num_cpus
        except Exception as e:
            print(f"[WARN] Failed to inspect container {container_id} on {docker_host}: {e}")
            return DockerContainerUsageMonitor._container_limits_cache.get(container_id)

    @staticmethod
    async def _monitor_dag(dag_id: str):
        """
        Periodically samples container resource limits for containers belonging to a DAG
        and accumulates allocated memory*time.
        """
        data = DockerContainerUsageMonitor._dag_data[dag_id]
        data["start_time"] = time.perf_counter()
        data["memory_seconds"] = defaultdict(float)
        data["container_ids"] = set()

        async with aiohttp.ClientSession() as session:
            while not data["stop"]:
                containers = await DockerContainerUsageMonitor._get_containers(session)
                for c in containers:
                    name = c["Names"][0]
                    if dag_id in name:
                        cid = c["Id"]
                        docker_host = c["DockerHost"]
                        data["container_ids"].add(cid)
                        limits = await DockerContainerUsageMonitor._get_container_limits(
                            session, cid, docker_host
                        )
                        if limits is not None:
                            mem_limit, _ = limits
                            # Accumulate allocated memory * sample interval
                            data["memory_seconds"][cid] += mem_limit * SAMPLE_INTERVAL_S
                await asyncio.sleep(SAMPLE_INTERVAL_S)

        data["end_time"] = time.perf_counter()

    @staticmethod
    def start_monitoring(dag_id: str, worker_config):
        from src.workers.docker_worker import DockerWorker
        assert isinstance(worker_config, DockerWorker.Config)
        if len(worker_config.external_docker_gateway_addresses) == 0: raise Exception(f"No Gateway IPs provided!")
        DockerContainerUsageMonitor._container_monitoring_addresses = worker_config.container_monitoring_addresses
        if dag_id in DockerContainerUsageMonitor._tasks:
            raise Exception(f"Monitoring already started for DAG {dag_id}")
        DockerContainerUsageMonitor._dag_data[dag_id] = {"stop": False}
        DockerContainerUsageMonitor._tasks[dag_id] = asyncio.create_task(
            DockerContainerUsageMonitor._monitor_dag(dag_id)
        )

    @staticmethod
    async def stop_monitoring(dag_id: str) -> DAGResourceUsageMetrics:
        if dag_id not in DockerContainerUsageMonitor._tasks:
            raise Exception(f"No monitoring task for DAG {dag_id}")

        # Signal the coroutine to stop and wait for it
        DockerContainerUsageMonitor._dag_data[dag_id]["stop"] = True
        await DockerContainerUsageMonitor._tasks[dag_id]

        # Retrieve and clean up
        data = DockerContainerUsageMonitor._dag_data.pop(dag_id)
        DockerContainerUsageMonitor._tasks.pop(dag_id)

        runtime = data["end_time"] - data["start_time"]
        total_memory_bytes_seconds = sum(data["memory_seconds"].values())
        total_cpu_seconds = 0

        # Calculate total allocated CPU seconds and clean cache
        total_cpus = 0
        for cid in data.get("container_ids", []):
            limits = DockerContainerUsageMonitor._container_limits_cache.pop(cid, None)
            if limits:
                _, num_cpus = limits
                total_cpus += num_cpus
                total_cpu_seconds += num_cpus * runtime

        return DAGResourceUsageMetrics(
            master_dag_id=dag_id,
            run_time_seconds=runtime,
            cpu_seconds=total_cpu_seconds,
            # AWS-like calculation: GB-s, but scaled down for easier comparison
            gb_seconds=total_memory_bytes_seconds / (1024**3),
        )
