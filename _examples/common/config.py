from src.planning.optimizations.preload import PreLoadOptimization
from src.planning.optimizations.prewarm import PreWarmOptimization
from src.planning.optimizations.wukong_optimizations import WukongOptimizations
from src.planning.sla import Percentile, SLA
from src.storage.redis_storage import RedisStorage
from src.workers.docker_worker import DockerWorker
from src.storage.metadata.metadata_storage import MetadataStorage
from src.planning.uniform_planner import UniformPlanner
from src.planning.non_uniform_planner import NonUniformPlanner
from src.task_worker_resource_configuration import TaskWorkerResourceConfiguration
from src.planning.wukong_planner import WUKONGPlanner
import sys
import os

def get_planner_from_sys_argv():
    supported_planners = ["wukong", "wukong-opt", "uniform", "uniform-opt", "non-uniform", "non-uniform-opt"]
    
    if len(sys.argv) < 2:
        print(f"Usage: python <script.py> <planner_type: {supported_planners}> <sla: average or 0-100>")
        sys.exit(-1)
    
    script_name = os.path.basename(sys.argv[0])

    planner_type = sys.argv[1]
    if planner_type not in supported_planners:
        print(f"Unknown planner type: {planner_type}")
        sys.exit(-1)

    is_montage_workflow = script_name == "montage.py"

    montage_min_worker_resource_config = TaskWorkerResourceConfiguration(8192)

    base_resources = (
        montage_min_worker_resource_config
        if is_montage_workflow
        else TaskWorkerResourceConfiguration(512)
    )

    # mid_resources = TaskWorkerResourceConfiguration(base_resources.memory_mb * 4) # 2GB
    # checking if resource contention is the issue
    mid_resources = TaskWorkerResourceConfiguration(base_resources.memory_mb * 16) # 4GB

    non_uniform_resources = (
        [
            base_resources,
            TaskWorkerResourceConfiguration(base_resources.memory_mb * 2),
        ]
        if is_montage_workflow
        else [
            mid_resources, # worst resource config possible is the same that uniform planners and WUKONG use
            TaskWorkerResourceConfiguration(base_resources.memory_mb * 8), # 4GB
            TaskWorkerResourceConfiguration(base_resources.memory_mb * 16), # 8GB
        ]
    )

    sla: SLA
    sla_str: str = sys.argv[2]
    if sla_str != "average":
        if int(sla_str) not in range(1, 101):
            print(f"Invalid SLA: {sla_str}. Accepted: 'average' or 0-100 (for percentile)")
            sys.exit(-1)
        sla = Percentile(int(sla_str))
    else:
        sla = "average"

    if planner_type == "wukong":
        return WUKONGPlanner.Config(
            sla=sla, # won't be used
            worker_resource_configurations=[mid_resources],
            optimizations=[],
        )
    elif planner_type == "wukong-opt":
        return WUKONGPlanner.Config(
            sla=sla,
            worker_resource_configurations=[mid_resources],
            optimizations=[
                WukongOptimizations.configured(
                    task_clustering_fan_outs=True, 
                    task_clustering_fan_ins=True, 
                    delayed_io=True, 
                    large_output_b=5 * 1024 * 1024 # 5MB
                ) 
            ],
        )
    elif planner_type == "uniform":
        return UniformPlanner.Config(
            sla=sla,
            worker_resource_configurations=[mid_resources],
            optimizations=[],
        )
    elif planner_type == "uniform-opt":
        return UniformPlanner.Config(
            sla=sla,
            worker_resource_configurations=[mid_resources],
            optimizations=[PreLoadOptimization, PreWarmOptimization],
        )
    elif planner_type == "non-uniform":
        return NonUniformPlanner.Config(
            sla=sla,
            worker_resource_configurations=non_uniform_resources,
            optimizations=[],
        )
    elif planner_type == "non-uniform-opt":
        return NonUniformPlanner.Config(
            sla=sla,
            worker_resource_configurations=non_uniform_resources,
            optimizations=[PreLoadOptimization, PreWarmOptimization]
        )
    else:
        raise ValueError(f"Unhandled planner type: {planner_type}")

# STORAGE CONFIGS
_REDIS_INTERMEDIATE_STORAGE_CONFIG = RedisStorage.Config(
    host="localhost",
    port=6379,
    password="redisdevpwd123"
)

_REDIS_METADATA_STORAGE_CONFIG = RedisStorage.Config(
    host="localhost",
    port=6380,
    password="redisdevpwd123"
)

# WORKER CONFIGS
WORKER_CONFIG = DockerWorker.Config(
    external_docker_gateway_address="http://localhost:5000",
    intermediate_storage_config=_REDIS_INTERMEDIATE_STORAGE_CONFIG,
    metadata_storage_config=MetadataStorage.Config(storage_config=_REDIS_METADATA_STORAGE_CONFIG),
    planner_config=get_planner_from_sys_argv()
)