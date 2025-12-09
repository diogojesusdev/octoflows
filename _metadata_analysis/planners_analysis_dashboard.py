import seaborn as sns
import matplotlib.pyplot as plt
import plotly.subplots as sp
import plotly.graph_objects as go
import streamlit as st
import cloudpickle
from typing import Dict, List
import pandas as pd
import numpy as np
import plotly.express as px
import hashlib
import colorsys
from dataclasses import dataclass
import sys
import os
import asyncio
import redis.asyncio as aioredis
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.storage.metadata.metadata_storage import MetadataStorage
from src.storage.metadata.metrics_types import TaskMetrics, UserDAGSubmissionMetrics
from src.planning.abstract_dag_planner import AbstractDAGPlanner
from src.storage.prefixes import DAG_PREFIX
from src.dag.dag import FullDAG
from src.storage.metadata.metrics_types import FullDAGPrepareTime, WorkerStartupMetrics, DAGResourceUsageMetrics
from src.utils.timer import Timer
from src.planning.optimizations.preload import PreLoadOptimization
from src.planning.optimizations.prewarm import PreWarmOptimization

def get_redis_connection(host: str, port: int):
    return aioredis.Redis(
        host=host,
        port=port,
        password="redisdevpwd123",
        decode_responses=False,
    )


@dataclass
class WorkflowInstanceTaskInfo:
    global_task_id: str
    internal_task_id: str
    metrics: TaskMetrics
    input_size_downloaded_bytes: int
    output_size_uploaded_bytes: int

    optimization_preloads_done: int

    optimization_prewarms_done: int
    optimization_prewarms_successful: int


@dataclass
class WorkflowInstanceInfo:
    master_dag_id: str
    plan: AbstractDAGPlanner.PlanOutput | None
    dag: FullDAG
    dag_download_stats: List[FullDAGPrepareTime]
    start_time_ms: float
    total_worker_startup_time_ms: float
    total_workers: int
    tasks: List[WorkflowInstanceTaskInfo]
    resource_usage: DAGResourceUsageMetrics
    total_transferred_data_bytes: int
    total_inputs_downloaded_bytes: float
    total_outputs_uploaded_bytes: float
    warm_starts_count: int
    cold_starts_count: int


@dataclass
class WorkflowInfo:
    type: str
    representative_dag: FullDAG
    instances: List[WorkflowInstanceInfo]


async def async_scan_keys(conn: aioredis.Redis, pattern: str) -> List[str]:
    """Efficient async scan for Redis keys."""
    cursor = b"0"
    keys = []
    while cursor:
        cursor, batch = await conn.scan(cursor=cursor, match=pattern, count=1000)
        keys.extend(batch)
    return keys


async def get_workflows_information(
    metadata_storage_conn: aioredis.Redis,
) -> tuple[List[WorkerStartupMetrics], Dict[str, WorkflowInfo]]:
    workflow_types: Dict[str, WorkflowInfo] = {
        "All": WorkflowInfo("All", None, [])  # type: ignore
    }
    worker_startup_metrics: List[WorkerStartupMetrics] = []

    try:
        all_dag_keys = await async_scan_keys(metadata_storage_conn, f"{DAG_PREFIX}*")
        print(f"Found {len(all_dag_keys)} workflow instances")

        async def process_dag_key(dag_key: str):
            try:
                dag_data = await metadata_storage_conn.get(dag_key)
                if not dag_data:
                    return None
                dag: FullDAG = cloudpickle.loads(dag_data)

                plan_key = f"{MetadataStorage.PLAN_KEY_PREFIX}{dag.master_dag_id}"
                plan_data = await metadata_storage_conn.get(plan_key)
                plan_output: AbstractDAGPlanner.PlanOutput | None = (
                    cloudpickle.loads(plan_data) if plan_data else None
                )
                if plan_output is None:
                    return None

                predicted_makespan_s = (
                    plan_output.nodes_info[
                        dag.sink_node.id.get_full_id()
                    ].task_completion_time_ms
                    / 1000
                ) if len(plan_output.nodes_info.keys()) > 0 else 0 # wukong has empty nodes info
                # Accept bad predictions for wukong but not us (because its normal on wukong, since we don't use the predictions)
                if predicted_makespan_s > 200 and "wukong" not in plan_output.planner_name:
                    print(
                        f"{dag.dag_name} | planner: {plan_output.planner_name} | predicted makespan of {predicted_makespan_s} | Discard plan, assume it didn't have it"
                    )
                    plan_output = None

                # DAG download stats
                download_keys = await async_scan_keys(
                    metadata_storage_conn,
                    f"{MetadataStorage.DAG_MD_KEY_PREFIX}{dag.master_dag_id}*",
                )
                if download_keys:
                    download_data = await metadata_storage_conn.mget(download_keys)
                    dag_download_stats = [
                        cloudpickle.loads(d) for d in download_data if d
                    ]
                else:
                    dag_download_stats = []

                # Tasks data (pipeline to minimize round-trips)
                task_keys = [
                    f"{MetadataStorage.TASK_MD_KEY_PREFIX}{t.id.get_full_id_in_dag(dag)}"
                    for t in dag._all_nodes.values()
                ]
                if task_keys:
                    task_data = await metadata_storage_conn.mget(task_keys)
                else:
                    task_data = []

                tasks = [
                    WorkflowInstanceTaskInfo(
                        t.id.get_full_id_in_dag(dag),
                        t.id.get_full_id(),
                        cloudpickle.loads(td),
                        -1,
                        -1,
                        0,
                        0,
                        0,
                    )
                    for t, td in zip(dag._all_nodes.values(), task_data)
                    if td
                ]
                if len(tasks) != len(dag._all_nodes):
                    print(f"[WARNING] Skipping incomplete metrics for {dag.dag_name}")
                    return None

                # Worker startup metrics
                worker_keys = await async_scan_keys(
                    metadata_storage_conn,
                    f"{MetadataStorage.WORKER_STARTUP_PREFIX}{dag.master_dag_id}*",
                )
                worker_data = await metadata_storage_conn.mget(worker_keys)
                this_workflow_wsm = [
                    cloudpickle.loads(d) for d in worker_data if d
                ]
                worker_startup_metrics.extend(this_workflow_wsm)

                total_inputs_downloaded = 0
                total_outputs_uploaded = 0

                for task in tasks:
                    tm = task.metrics
                    task.input_size_downloaded_bytes = sum(
                        [
                            m.serialized_size_bytes
                            for m in tm.input_metrics.input_download_metrics.values()
                            if m.time_ms is not None
                        ]
                    )
                    task.output_size_uploaded_bytes = (
                        tm.output_metrics.serialized_size_bytes
                        if tm.output_metrics.tp_time_ms is not None
                        else 0
                    )
                    total_inputs_downloaded += task.input_size_downloaded_bytes
                    total_outputs_uploaded += task.output_size_uploaded_bytes

                    if tm.optimization_metrics:
                        task.optimization_preloads_done = len(
                            [
                                om
                                for om in tm.optimization_metrics
                                if isinstance(om, PreLoadOptimization.OptimizationMetrics)
                            ]
                        )
                        task.optimization_prewarms_done = len(
                            [
                                om
                                for om in tm.optimization_metrics
                                if isinstance(om, PreWarmOptimization.OptimizationMetrics)
                            ]
                        )
                        task.optimization_prewarms_successful = len(
                            [
                                om
                                for om in tm.optimization_metrics
                                if isinstance(
                                    om, PreWarmOptimization.OptimizationMetrics
                                )
                                and [
                                    wsm
                                    for wsm in this_workflow_wsm
                                    if wsm.resource_configuration.worker_id
                                    == om.resource_config.worker_id
                                ][0].state
                                == "warm"
                            ]
                        )

                # if plan_output:
                #     print(f"{plan_output.planner_name} | Total preloads assigned: {sum([len([o for o in n.optimizations if isinstance(o, PreLoadOptimization)]) for n in dag._all_nodes.values()])} | Preloads Done: {sum([t.optimization_preloads_done for t in tasks])}")

                submission_key = (
                    f"{MetadataStorage.USER_DAG_SUBMISSION_PREFIX}{dag.master_dag_id}"
                )
                submission_data = await metadata_storage_conn.get(submission_key)
                if not submission_data:
                    return None
                dag_submission_metrics: UserDAGSubmissionMetrics = cloudpickle.loads(
                    submission_data
                )

                #* DEBUG
                # for task in tasks:
                #     for om in task.metrics.optimization_metrics:
                #         if not isinstance(om, PreWarmOptimization.OptimizationMetrics): continue
                #         target_worker_id = om.resource_config.worker_id
                #         wsm = [m for m in this_workflow_wsm if m.resource_configuration.worker_id == target_worker_id]
                #         if not len(wsm): continue
                #         this_worker_startup_metrics = wsm[0]
                #         if plan_output:
                #             print(f"{plan_output.planner_name} | time to worker start prewarm at: {(this_worker_startup_metrics.start_time_ms / 1000) - om.absolute_trigger_timestamp_s} | Worker state: {this_worker_startup_metrics.state}")

                warm_starts_count = len(
                    [m for m in this_workflow_wsm if m.state == "warm"]
                )
                cold_starts_count = len(
                    [m for m in this_workflow_wsm if m.state == "cold"]
                )

                total_worker_startup_time_ms = sum(
                    [
                        (m.end_time_ms - m.start_time_ms)
                        for m in this_workflow_wsm
                        if m.end_time_ms
                    ]
                )
                total_workers = len(this_workflow_wsm)

                resource_usage_key = (
                    f"{MetadataStorage.DAG_RESOURCE_USAGE_PREFIX}{dag.master_dag_id}"
                )
                resource_usage_data = await metadata_storage_conn.get(resource_usage_key)
                resource_usage: DAGResourceUsageMetrics = cloudpickle.loads(
                    resource_usage_data
                )

                total_transferred_data_bytes = (
                    total_inputs_downloaded + total_outputs_uploaded
                )

                if dag.dag_name not in workflow_types:
                    workflow_types[dag.dag_name] = WorkflowInfo(dag.dag_name, dag, [])

                instance_info = WorkflowInstanceInfo(
                    dag.master_dag_id,
                    plan_output,
                    dag,
                    dag_download_stats,
                    dag_submission_metrics.dag_submission_time_ms,
                    total_worker_startup_time_ms,
                    total_workers,
                    tasks,
                    resource_usage,
                    total_transferred_data_bytes,
                    total_inputs_downloaded,
                    total_outputs_uploaded,
                    warm_starts_count,
                    cold_starts_count,
                )

                workflow_types[dag.dag_name].instances.append(instance_info)
                workflow_types["All"].instances.append(instance_info)

            except Exception as e:
                print(f"Error processing DAG {dag_key}: {e}")
                return None

        # Process all DAGs concurrently (limit concurrency to avoid overloading Redis)
        semaphore = asyncio.Semaphore(50)

        async def limited_process(dag_key):
            async with semaphore:
                return await process_dag_key(dag_key)

        await asyncio.gather(*(limited_process(k) for k in all_dag_keys))

    except Exception as e:
        print(f"Error accessing Redis: {e}")
        raise e

    return worker_startup_metrics, workflow_types

def format_bytes(size: float) -> tuple[float, str, str]:
    """Convert bytes to human-readable format"""
    # returns [value, unit, formatted_string]
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return size, unit, f"{size:.2f} {unit}"
        size /= 1024.0
    return size, 'TB', f"{size:.2f} TB"

def get_color_for_workflow(workflow_name: str) -> str:
    """Generate a consistent color for each workflow type"""
    # Create a hash of the workflow name
    hash_obj = hashlib.md5(workflow_name.encode())
    hash_int = int(hash_obj.hexdigest(), 16)
    
    # Generate a hue value that is more spaced out
    hue = (hash_int % 360)  # Full hue spectrum (0-359 degrees)
    saturation = 0.7  # Keep colors vibrant
    lightness = 0.6   # Slightly lighter for better visibility
    
    # Convert HSL to RGB (values between 0-1)
    r, g, b = colorsys.hls_to_rgb(hue / 360, lightness, saturation)
    
    # Scale to 0-255 and format as RGB
    return f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})"

def calculate_prediction_error(actual, predicted):
    """Calculate relative prediction error percentage"""
    if actual == 0 and predicted == 0:
        return 0
    if actual == 0:
        return float('inf')
    return abs(actual - predicted) / actual * 100

async def main():
    # Configure page layout for better visualization
    st.set_page_config(layout="wide")
    st.title("Planning Analysis Dashboard")
    
    # Connect to both Redis instances
    # metadata_storage_conn = get_redis_connection("localhost", 6380)
    metadata_storage_conn = get_redis_connection("146.193.41.126", 6380)
    
    # Initialize workflow types in session state if not already loaded
    if 'workflow_types' not in st.session_state:
        timer = Timer()
        st.session_state.worker_startup_metrics, st.session_state.workflow_types = await get_workflows_information(metadata_storage_conn)
        print(f"Time to load workflow information: {(timer.stop() / 1_000):.2f} s")
    
    workflow_types = st.session_state.workflow_types

    all_resource_usages = []

    if not workflow_types:
        st.warning("No DAGs found in Redis")
        st.stop()
    
    # Sidebar for workflow type selection
    st.sidebar.title("Workflow Filter")
    
    # Create a dropdown to select workflow type
    selected_workflow = st.sidebar.selectbox(
        "Select Workflow Type",
        options=sorted(list(workflow_types.keys())),
        index=0
    )
    
    st.sidebar.subheader("Workflow Statistics")
    workflow_stats = []
    for workflow, keys in workflow_types.items():
        workflow_stats.append({
            "Workflow Type": workflow,
            "Count": len(keys.instances),
            "Color": get_color_for_workflow(workflow)
        })
    
    # Bar Chart of workflow counts
    if workflow_stats:
        df_workflows = pd.DataFrame(workflow_stats)
        fig = px.bar(
            df_workflows, 
            x="Workflow Type", 
            y="Count",
            color="Workflow Type",
            color_discrete_map={row["Workflow Type"]: row["Color"] for _, row in df_workflows.iterrows()},
            title="Workflow Type Distribution"
        )
        fig.update_layout(showlegend=False)
        st.sidebar.plotly_chart(fig, use_container_width=True)
    
    st.header(selected_workflow)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Workflow Instances", len(workflow_types[selected_workflow].instances))
    if selected_workflow != 'All':
        with col2:
            st.metric("Workflow Tasks", len(workflow_types[selected_workflow].representative_dag._all_nodes))
    
    instance_data = []
    for idx, instance in enumerate(workflow_types[selected_workflow].instances):
        if not instance.plan or not instance.tasks:
            continue
            
        # Calculate actual metrics
        actual_total_download = sum([sum([input_metric.time_ms / 1000 for input_metric in task.metrics.input_metrics.input_download_metrics.values() if input_metric.time_ms is not None]) for task in instance.tasks])
        actual_execution = sum(task.metrics.tp_execution_time_ms / 1000 for task in instance.tasks)  # in seconds
        actual_total_upload = sum(task.metrics.output_metrics.tp_time_ms / 1000 for task in instance.tasks if task.metrics.output_metrics.tp_time_ms is not None)  # in seconds
        actual_invocation = sum(task.metrics.total_invocation_time_ms / 1000 for task in instance.tasks if task.metrics.total_invocation_time_ms is not None)  # in seconds
        actual_dependency_update = sum(task.metrics.update_dependency_counters_time_ms / 1000 for task in instance.tasks if task.metrics.update_dependency_counters_time_ms is not None)  # in seconds
        actual_input_size = sum([sum([input_metric.serialized_size_bytes for input_metric in task.metrics.input_metrics.input_download_metrics.values()]) + task.metrics.input_metrics.hardcoded_input_size_bytes for task in instance.tasks])  # in bytes
        actual_output_size = sum([task.metrics.output_metrics.serialized_size_bytes for task in instance.tasks])  # in bytes
        actual_total_worker_startup_time_s = instance.total_worker_startup_time_ms / 1000  # in seconds
        
        # Calculate actual makespan
        from src.task_worker_resource_configuration import TaskWorkerResourceConfiguration
        common_resources: TaskWorkerResourceConfiguration | None = None
        for task in instance.tasks:
            if common_resources is None: common_resources = task.metrics.worker_resource_configuration
            elif common_resources.cpus != task.metrics.worker_resource_configuration.cpus or common_resources.memory_mb != task.metrics.worker_resource_configuration.memory_mb: 
                # print(f"Found a diff resource config. Prev: {common_resources} New: {task.metrics.worker_resource_configuration}")
                common_resources = None
        
        sink_task_metrics = [t for t in instance.tasks if t.internal_task_id == instance.dag.sink_node.id.get_full_id()][0].metrics
        sink_task_ended_timestamp_ms = (sink_task_metrics.started_at_timestamp_s * 1000) + (sink_task_metrics.input_metrics.tp_total_time_waiting_for_inputs_ms or 0) + (sink_task_metrics.tp_execution_time_ms or 0) + (sink_task_metrics.output_metrics.tp_time_ms or 0) + (sink_task_metrics.total_invocation_time_ms or 0)
        actual_makespan_s = (sink_task_ended_timestamp_ms - instance.start_time_ms) / 1000
        actual_data_transferred = instance.total_transferred_data_bytes
        actual_total_downloadable_input_size_bytes = instance.total_inputs_downloaded_bytes
        actual_total_uploadable_output_size_bytes = instance.total_outputs_uploaded_bytes

        # Get predicted metrics if available
        predicted_total_downloadable_input_size_bytes = predicted_total_download = predicted_execution = predicted_total_upload = predicted_makespan_s = 0
        predicted_total_uploadable_output_size_bytes = predicted_input_size_bytes = predicted_output_size = predicted_total_worker_startup_time_s = 0
        if instance.plan and instance.plan.nodes_info:
            predicted_total_download = sum(info.total_download_time_ms / 1000 for info in instance.plan.nodes_info.values())  # in seconds
            predicted_execution = sum(info.tp_exec_time_ms / 1000 for info in instance.plan.nodes_info.values())  # in seconds
            predicted_total_upload = sum(info.tp_upload_time_ms / 1000 for info in instance.plan.nodes_info.values())  # in seconds
            predicted_input_size_bytes = sum(info.serialized_input_size for info in instance.plan.nodes_info.values())  # in bytes
            predicted_output_size = sum(info.serialized_output_size for info in instance.plan.nodes_info.values())  # in bytes
            predicted_makespan_s = instance.plan.nodes_info[instance.dag.sink_node.id.get_full_id()].task_completion_time_ms / 1000
            workers_accounted_for = set()
            predicted_total_worker_startup_time_s = 0
            for info in instance.plan.nodes_info.values():
                if info.node_ref.worker_config.worker_id is None or info.node_ref.worker_config.worker_id not in workers_accounted_for:
                    predicted_total_worker_startup_time_s += info.tp_worker_startup_time_ms / 1000
                    workers_accounted_for.add(info.node_ref.worker_config.worker_id)
            predicted_total_downloadable_input_size_bytes = instance.plan.total_downloadable_input_size_bytes
            predicted_total_uploadable_output_size_bytes = instance.plan.total_uploadable_output_size_bytes
        
        # Store actual values for SLA comparison
        instance_metrics = {
            'makespan': actual_makespan_s,
            'execution': actual_execution,
            'download': actual_total_download,
            'upload': actual_total_upload,
            'input_size': actual_input_size,
            'output_size': actual_output_size,
            'worker_startup': actual_total_worker_startup_time_s
        }
        
        def convert_bytes_to_GB(bytes):
            return bytes / (1024 * 1024 * 1024)

        # Calculate differences and percentages with sample counts
        def format_metric(actual, predicted, metric_name=None, samples=None):
            # Store the actual value for SLA comparison
            if metric_name is not None:
                instance_metrics[metric_name] = actual
            
            # Format the predicted vs actual comparison
            if predicted == 0 and actual == 0:
                comparison = "0.000s (0.000%)"
            else:
                diff = actual - predicted
                pct_diff = (diff / predicted * 100) if predicted != 0 else float('inf')
                sign = "+" if diff >= 0 else ""
                comparison = f"{predicted:.3f}s → {actual:.3f}s ({sign}{diff:.3f}s, {sign}{pct_diff:.1f}%)"
            
            # Add sample count if provided
            if samples is not None:
                comparison = f"{comparison}\n(samples: {samples})"
            
            return comparison
        
        # Get sample counts if available
        sample_counts = instance.plan.prediction_sample_counts if instance.plan and hasattr(instance.plan, 'prediction_sample_counts') else None
        
        def format_size_metric(actual, predicted, metric_name=None, samples=None):
            # Store the actual value for SLA comparison
            if metric_name is not None:
                instance_metrics[metric_name] = actual
                
            formatted_actual = format_bytes(actual)[2]
            formatted_predicted = format_bytes(predicted)[2]
            
            # Format the predicted vs actual comparison
            if predicted == 0 and actual == 0:
                comparison = f"{formatted_predicted} → {formatted_actual} (0.0%)"
            else:
                diff = actual - predicted
                pct_diff = (diff / predicted * 100) if predicted != 0 else float('inf')
                sign = "+" if diff >= 0 else ""
                comparison = f"{formatted_predicted} → {formatted_actual} ({sign}{pct_diff:.1f}%)"
            
            # Add sample count if provided
            if samples is not None:
                comparison = f"{comparison}\n(samples: {samples})"
                
            return comparison
            
        # Format SLA for display and store for later use
        sla_percentile = None
        sla_value = 'N/A'
        if instance.plan:
            if instance.plan.sla == 'average':
                sla_value = 'average'
                sla_percentile = None
            else:
                sla_value = f'p{instance.plan.sla.value}'
                sla_percentile = instance.plan.sla.value
        
        # Calculate total DAG download time across all downloads
        # total_download_time = sum(stat.download_time_ms for stat in instance.dag_download_stats)
        # dag_download_time = f"{total_download_time / 1000:.3f}s"

        unique_worker_ids = set()
        for task_metrics in instance.tasks:
            unique_worker_ids.add(task_metrics.metrics.worker_resource_configuration.worker_id)
        

        is_wukong_instance = 'wukong' in instance.plan.planner_name.lower()  if instance.plan else None
        # Store the instance data with metrics for SLA comparison
        instance_data.append({
            'Workflow Type': selected_workflow,
            'Planner': instance.plan.planner_name if instance.plan else 'N/A',
            'Resources': f"{common_resources.cpus} CPUs {common_resources.memory_mb} MB" if common_resources else 'Non-Uniform',
            'SLA': sla_value,
            '_sla_percentile': sla_percentile,
            'Master DAG ID': instance.master_dag_id,
            'Makespan': format_metric(actual_makespan_s, 0 if is_wukong_instance else predicted_makespan_s, 'makespan',
                                sample_counts.for_execution_time if sample_counts else None),
            'Total Execution Time': format_metric(actual_execution, 0 if is_wukong_instance else predicted_execution, 'execution',
                                        sample_counts.for_execution_time if sample_counts else None),
            'Total Downloaded Data': f"{0 if is_wukong_instance else format_bytes(predicted_total_downloadable_input_size_bytes)[2]} -> {format_bytes(actual_total_downloadable_input_size_bytes)[2]}",
            'Total Download Time': format_metric(actual_total_download, 0 if is_wukong_instance else predicted_total_download, 'download',
                                    sample_counts.for_download_speed if sample_counts else None),
            'Total Uploaded Data': f"{0 if is_wukong_instance else format_bytes(predicted_total_uploadable_output_size_bytes)[2]} -> {format_bytes(actual_total_uploadable_output_size_bytes)[2]}",
            'Total Upload Time': format_metric(actual_total_upload, 0 if is_wukong_instance else predicted_total_upload, 'upload',
                                    sample_counts.for_upload_speed if sample_counts else None),
            'Total Input Size': format_size_metric(actual_input_size, 0 if is_wukong_instance else predicted_input_size_bytes, 'input_size',
                                    sample_counts.for_output_size if sample_counts else None),
            'Total Output Size': format_size_metric(actual_output_size, 0 if is_wukong_instance else predicted_output_size, 'output_size',
                                    sample_counts.for_output_size if sample_counts else None),
            'Total Data Transferred': format_bytes(actual_data_transferred)[2],
            'Total Task Invocation Time': f"{actual_invocation:.2f}s",
            'Total Dependency Counter Update Time': f"{actual_dependency_update:.2f}s",
            # 'Total DAG Download Time': dag_download_time,
            'Total Worker Startup Time': format_metric(actual_total_worker_startup_time_s, 0 if is_wukong_instance else predicted_total_worker_startup_time_s, 'worker_startup') + f" (Workers: {len(unique_worker_ids)})",
            'CPU Time': f"{instance.resource_usage.cpu_seconds:.2f}",
            'Resource Usage': f"{instance.resource_usage.gb_seconds:.2f}",
            'Warm/Cold Starts': f"{instance.warm_starts_count}/{instance.cold_starts_count}",
            '_actual_worker_startup': actual_total_worker_startup_time_s,
            '_actual_invocation': actual_invocation,
            '_actual_dependency_update': actual_dependency_update,
            '_sample_count': sample_counts.for_execution_time if sample_counts else 0,
            '_metrics': instance_metrics
        })

    if instance_data:
        # Create a DataFrame for the table
        df_instances = pd.DataFrame(instance_data)
        
        # Ensure we're working with a pandas DataFrame
        if not isinstance(df_instances, pd.DataFrame):
            df_instances = pd.DataFrame(df_instances)
        
        # Calculate SLA metrics across all instances for each metric type
        if len(df_instances) > 0 and '_metrics' in df_instances.columns:
            # Get all metrics from all instances
            all_metrics = {
                'makespan': [],
                'execution': [],
                'download': [],
                'upload': [],
                'input_size': [],
                'output_size': [],
                'worker_startup': []
            }
            
            # Collect all metrics
            for _, row in df_instances.iterrows():
                for metric, value in row['_metrics'].items():
                    all_metrics[metric].append(value)
            
            # Sort instances by start time to ensure we're looking at previous instances correctly
            if not df_instances.empty and 'Master DAG ID' in df_instances.columns:
                # Create a list to store all previous metrics for each metric type
                all_previous_metrics = {
                    'makespan': [],
                    'execution': [],
                    'download': [],
                    'upload': [],
                    'input_size': [],
                    'output_size': [],
                    'worker_startup': []
                }
                
                # Create a new column to store the formatted values
                formatted_values = {col: [None] * len(df_instances) for col in [
                    'Makespan', 'Total Execution Time', 'Total Download Time',
                    'Total Upload Time', 'Total Input Size', 'Total Output Size',
                    'Total Worker Startup Time'
                ]}
                
                # Process each instance in order
                for idx, row in df_instances.iterrows():
                    metrics = row['_metrics']
                    sla_pct = row['_sla_percentile']
                    sla_value = row['SLA']
                    
                    # Helper function to format SLA comparison for this instance
                    def format_with_sla(metric_name: str, value: float, unit: str = 's', current_metrics: Dict[str, float] = None) -> str:
                        try:
                            if not current_metrics or metric_name not in current_metrics or pd.isna(current_metrics[metric_name]):
                                return f"{value:.3f}{unit}"
                            
                            sla = float(current_metrics[metric_name])
                            offset = value - sla
                            sign = "+" if offset >= 0 else ""
                            meets_sla = value <= sla
                            emoji = "✅" if meets_sla else "❌"
                            
                            # Get the current cell value (contains the predicted vs actual comparison)
                            current_value = str(df_instances.at[idx, col_name])
                            
                            # Format the SLA information
                            if unit == 's':
                                sla_info = f"{emoji} SLA: {sla:.3f}s (offset: {sign}{abs(offset):.3f}s)"
                            else:  # for sizes
                                sla_info = f"{emoji} SLA: {format_bytes(sla)[2]} (offset: {sign}{format_bytes(abs(offset))[2]})"
                            
                            return f"{current_value}\n{sla_info}"
                        except Exception as e:
                            print(f"Error formatting SLA for {metric_name}: {e}")
                            return str(value)
                    
                    # Calculate SLA based on previous instances
                    current_sla_metrics = {}
                    if sla_pct is not None and all_previous_metrics['makespan']:  # Only calculate if we have previous data
                        for metric in all_previous_metrics.keys():
                            if not all_previous_metrics[metric]:
                                continue
                                
                            try:
                                if sla_value == "average":
                                    sla_value = float(np.average(all_previous_metrics[metric]))
                                else:  # specific percentile
                                    sla_value = float(np.percentile(all_previous_metrics[metric], sla_pct))
                                current_sla_metrics[metric] = sla_value
                            except (TypeError, ValueError) as e:
                                print(f"Error calculating SLA for {metric}: {e}")
                    
                    # Update each metric with SLA comparison if we have SLA data
                    metric_columns = {
                        'Makespan': ('makespan', 's'),
                        'Total Execution Time': ('execution', 's'),
                        'Total Download Time': ('download', 's'),
                        'Total Upload Time': ('upload', 's'),
                        'Total Input Size': ('input_size', 'b'),
                        'Total Output Size': ('output_size', 'b'),
                        'Total Worker Startup Time': ('worker_startup', 's')
                    }
                    
                    for col_name, (metric_name, unit) in metric_columns.items():
                        if col_name in df_instances.columns and metric_name in metrics:
                            try:
                                metric_value = float(metrics[metric_name])
                                
                                if current_sla_metrics:  # Only add SLA if we have previous data to compare with
                                    formatted = format_with_sla(metric_name, metric_value, unit, current_sla_metrics)
                                else:
                                    formatted = str(df_instances.at[idx, col_name])
                                    
                                if formatted is not None:
                                    formatted_values[col_name][idx] = formatted
                            except (ValueError, TypeError) as e:
                                print(f"Error processing {metric_name}: {e}")
                                formatted_values[col_name][idx] = str(metrics.get(metric_name, 'N/A'))
                    
                    # Add current instance's metrics to the history for the next iteration
                    for metric_name in all_previous_metrics.keys():
                        if metric_name in metrics:
                            all_previous_metrics[metric_name].append(float(metrics[metric_name]))
                
                # Update the dataframe with the formatted values
                for col_name, values in formatted_values.items():
                    if col_name in df_instances.columns:
                        df_instances[col_name] = values
        
        # Sort by sample count in descending order
        if '_sample_count' in df_instances.columns:
            df_instances = df_instances.sort_values('_sample_count', ascending=False)
        
        # Remove the temporary columns before display
        cols_to_drop = [col for col in [
            '_sample_count', 
            '_actual_invocation', 
            '_actual_dependency_update',
            '_actual_worker_startup',
            '_metrics',
            '_sla_percentile'
        ] if col in df_instances.columns]
        
        if cols_to_drop:
            df_instances = df_instances.drop(columns=cols_to_drop)
        
        # Reorder columns to put Master DAG ID first
        if 'Master DAG ID' in df_instances.columns:
            columns = ['Master DAG ID'] + [col for col in df_instances.columns if col != 'Master DAG ID']
            df_instances = df_instances[columns]

        TAB_RAW, TAB_PREDICTIONS, TAB_ACTUAL_VALUES = st.tabs(["Raw", "Predictions", "Actual Values"])

        with TAB_RAW:
            # Add planner filter dropdown
            if isinstance(df_instances, pd.DataFrame) and 'Planner' in df_instances.columns:
                # Get unique planners using a method that works with both pandas and numpy arrays
                planner_values = df_instances['Planner'].values if hasattr(df_instances['Planner'], 'values') else df_instances['Planner']
                if hasattr(planner_values, 'tolist'):
                    planner_values = planner_values.tolist()
                
                # Convert to a set to get unique values, then back to a list
                unique_planners = list(set(str(p) for p in planner_values if pd.notna(p) and p is not None))
                all_planners = ['All'] + sorted(unique_planners)
                
                selected_planner = st.selectbox(
                    'Filter by Planner:',
                    all_planners,
                    index=0
                )
                
                # Filter by selected planner if not 'All'
                if selected_planner != 'All':
                    df_instances = df_instances[df_instances['Planner'].astype(str) == selected_planner]
            
            # Display the instance comparison table with row count
            st.markdown(f"### Instance Comparison ({len(df_instances)} instances)")
            
            # Display the table
            st.dataframe(
                df_instances,
                column_config={
                    'Workflow Type': "Workflow Type",
                    'Resources': "Resources",
                    'SLA': "SLA",
                    'Makespan': "Makespan (Predicted → Actual)",
                    'Total Execution Time': "Total Execution Time (Predicted → Actual)",
                    'Total Downloaded Data': "Total Downloaded Data (Predicted → Actual)",
                    'Total Download Time': "Total Download Time (Predicted → Actual)",
                    'Total Uploaded Data': "Total Uploaded Data (Predicted → Actual)",
                    'Total Upload Time': "Total Upload Time (Predicted → Actual)",
                    'Total Input Size': "Total Input Size (Predicted → Actual)",
                    'Total Output Size': "Total Output Size (Predicted → Actual)",
                    'Total Data Transferred': "Total Data Transferred",
                    'Total Task Invocation Time': "Total Task Invocation Time",
                    'Total Dependency Counter Update Time': "Total Dependency Counter Update Time",
                    # 'Total DAG Download Time': "Total DAG Download Time",
                    'Total Worker Startup Time': "Total Worker Startup Time (Predicted → Actual)",
                    'Run Time': "Run Time",
                    'CPU Time': "CPU Time",
                    'Resource Usage': "Resource Usage",
                    'Warm/Cold Starts': "Warm/Cold Starts",
                },
                use_container_width=True,
                height=min(400, 35 * (len(df_instances) + 1)),
                hide_index=True,
                column_order=[
                    'Workflow Type', 
                    'Planner',
                    'Resources',
                    'SLA',
                    'Master DAG ID',
                    'Makespan', 
                    'Total Execution Time', 
                    'Total Downloaded Data',
                    'Total Download Time',
                    'Total Uploaded Data',
                    'Total Upload Time',
                    'Total Input Size',
                    'Total Output Size',
                    'Total Data Transferred',
                    'Total Task Invocation Time',
                    'Total Dependency Counter Update Time',
                    'Total Worker Startup Time',
                    'Run Time',
                    'CPU Time',
                    'Resource Usage',
                    'Warm/Cold Starts',
                ]
            )

        # st.markdown("---")
        with TAB_PREDICTIONS:
            # Add comparison bar chart for predicted vs actual metrics
            st.markdown("### Reality vs Predictions")
            
            # Calculate averages for the comparison
            metrics_data = []
            for instance in workflow_types[selected_workflow].instances:
                if not instance.plan or not instance.tasks:
                    continue
                if 'wukong' in instance.plan.planner_name.lower():
                    continue
                    
                # Calculate actual metrics
                sink_task_metrics = [t for t in instance.tasks if t.internal_task_id == instance.dag.sink_node.id.get_full_id()][0].metrics
                sink_task_ended_timestamp_ms = (sink_task_metrics.started_at_timestamp_s * 1000) + (sink_task_metrics.input_metrics.tp_total_time_waiting_for_inputs_ms or 0) + (sink_task_metrics.tp_execution_time_ms or 0) + (sink_task_metrics.output_metrics.tp_time_ms or 0) + (sink_task_metrics.total_invocation_time_ms or 0)
                actual_makespan_s = (sink_task_ended_timestamp_ms - instance.start_time_ms) / 1000
                actual_execution = sum(task.metrics.tp_execution_time_ms / 1000 for task in instance.tasks)
                actual_total_download = sum([sum([input_metric.time_ms / 1000 for input_metric in task.metrics.input_metrics.input_download_metrics.values() if input_metric.time_ms is not None]) for task in instance.tasks])
                actual_total_upload = sum(task.metrics.output_metrics.tp_time_ms / 1000 for task in instance.tasks if task.metrics.output_metrics.tp_time_ms is not None)
                actual_input_size = sum([sum([input_metric.serialized_size_bytes for input_metric in task.metrics.input_metrics.input_download_metrics.values()]) + task.metrics.input_metrics.hardcoded_input_size_bytes for task in instance.tasks])
                actual_output_size = sum([task.metrics.output_metrics.serialized_size_bytes for task in instance.tasks])
                actual_worker_startup_time_s = sum([metric.end_time_ms - metric.start_time_ms for metric in st.session_state.worker_startup_metrics if metric.master_dag_id == instance.master_dag_id and metric.end_time_ms is not None])

                # Get predicted metrics if available
                predicted_makespan_s = predicted_execution = predicted_total_download = predicted_total_upload = predicted_input_size_bytes = predicted_output_size = predicted_worker_startup_time_s = 0 # initialize them outside
                if instance.plan and instance.plan.nodes_info:
                    predicted_makespan_s = instance.plan.nodes_info[instance.dag.sink_node.id.get_full_id()].task_completion_time_ms / 1000
                    predicted_total_download = sum(info.total_download_time_ms / 1000 for info in instance.plan.nodes_info.values())
                    predicted_execution = sum(info.tp_exec_time_ms / 1000 for info in instance.plan.nodes_info.values())
                    predicted_total_upload = sum(info.tp_upload_time_ms / 1000 for info in instance.plan.nodes_info.values())
                    predicted_input_size_bytes = sum(info.serialized_input_size for info in instance.plan.nodes_info.values())
                    predicted_output_size = sum(info.serialized_output_size for info in instance.plan.nodes_info.values())
                    workers_accounted_for = set()
                    predicted_worker_startup_time_s = 0
                    for info in instance.plan.nodes_info.values():
                        if info.node_ref.worker_config.worker_id is None or info.node_ref.worker_config.worker_id not in workers_accounted_for:
                            predicted_worker_startup_time_s += info.tp_worker_startup_time_ms / 1000
                            workers_accounted_for.add(info.node_ref.worker_config.worker_id)
                
                metrics_data.append({
                    'makespan_actual': actual_makespan_s,
                    'makespan_predicted': predicted_makespan_s,
                    'execution_actual': actual_execution,
                    'execution_predicted': predicted_execution,
                    'download_actual': actual_total_download,
                    'download_predicted': predicted_total_download,
                    'upload_actual': actual_total_upload,
                    'upload_predicted': predicted_total_upload,
                    'input_size_actual': actual_input_size,
                    'input_size_predicted': predicted_input_size_bytes,
                    'output_size_actual': actual_output_size,
                    'output_size_predicted': predicted_output_size,
                    'worker_startup_time_actual': actual_worker_startup_time_s,
                    'worker_startup_time_predicted': predicted_worker_startup_time_s,
                })
            
            if metrics_data:
                # --- Group metrics by planner and SLA ---
                planner_metrics = {}
                instances_clone = workflow_types[selected_workflow].instances.copy()
                for instance in instances_clone:
                    if not instance.plan or not instance.tasks:
                        continue
                    if 'wukong' in instance.plan.planner_name.lower():
                        continue

                    # Ensure SLA exists
                    if not instance.plan.sla or instance.plan.sla.value is None:
                        continue
                    sla_value = instance.plan.sla.value

                    # Compute actual metrics
                    sink_task_metrics = [t for t in instance.tasks if t.internal_task_id == instance.dag.sink_node.id.get_full_id()][0].metrics
                    sink_task_ended_timestamp_ms = (sink_task_metrics.started_at_timestamp_s * 1000) + (sink_task_metrics.input_metrics.tp_total_time_waiting_for_inputs_ms or 0) + (sink_task_metrics.tp_execution_time_ms or 0) + (sink_task_metrics.output_metrics.tp_time_ms or 0) + (sink_task_metrics.total_invocation_time_ms or 0)
                    actual_makespan_s = (sink_task_ended_timestamp_ms - instance.start_time_ms) / 1000

                    actual_execution = sum(task.metrics.tp_execution_time_ms / 1000 for task in instance.tasks)
                    actual_total_download = sum([
                        sum([
                            input_metric.time_ms / 1000
                            for input_metric in task.metrics.input_metrics.input_download_metrics.values()
                            if input_metric.time_ms is not None
                        ]) for task in instance.tasks
                    ])
                    actual_total_upload = sum(
                        task.metrics.output_metrics.tp_time_ms / 1000
                        for task in instance.tasks
                        if task.metrics.output_metrics.tp_time_ms is not None
                    )
                    actual_input_size = sum([
                        sum([
                            input_metric.serialized_size_bytes
                            for input_metric in task.metrics.input_metrics.input_download_metrics.values()
                        ]) + task.metrics.input_metrics.hardcoded_input_size_bytes
                        for task in instance.tasks
                    ])
                    actual_output_size = sum([
                        task.metrics.output_metrics.serialized_size_bytes for task in instance.tasks
                    ])
                    actual_worker_startup_time_s = sum([
                        (metric.end_time_ms - metric.start_time_ms) / 1000
                        for metric in st.session_state.worker_startup_metrics
                        if metric.master_dag_id == instance.master_dag_id and metric.end_time_ms is not None
                    ])

                    # Compute predicted metrics
                    predicted_makespan_s = predicted_execution = predicted_total_download = predicted_total_upload = predicted_input_size_bytes = predicted_output_size = predicted_worker_startup_time_s = 0
                    if instance.plan and instance.plan.nodes_info:
                        predicted_makespan_s = instance.plan.nodes_info[instance.dag.sink_node.id.get_full_id()].task_completion_time_ms / 1000
                        predicted_total_download = sum(info.total_download_time_ms / 1000 for info in instance.plan.nodes_info.values())
                        predicted_execution = sum(info.tp_exec_time_ms / 1000 for info in instance.plan.nodes_info.values())
                        predicted_total_upload = sum(info.tp_upload_time_ms / 1000 for info in instance.plan.nodes_info.values())
                        predicted_input_size_bytes = sum(info.serialized_input_size for info in instance.plan.nodes_info.values())
                        predicted_output_size = sum(info.serialized_output_size for info in instance.plan.nodes_info.values())
                        workers_accounted_for = set()
                        predicted_worker_startup_time_s = 0
                        for info in instance.plan.nodes_info.values():
                            if info.node_ref.worker_config.worker_id is None or info.node_ref.worker_config.worker_id not in workers_accounted_for:
                                predicted_worker_startup_time_s += info.tp_worker_startup_time_ms / 1000
                                workers_accounted_for.add(info.node_ref.worker_config.worker_id)

                        planner_name = instance.plan.planner_name
                        key = (planner_name, sla_value)
                        if key not in planner_metrics:
                            planner_metrics[key] = []
                        planner_metrics[key].append({
                            'makespan_actual': actual_makespan_s,
                            'makespan_predicted': predicted_makespan_s,
                            'execution_actual': actual_execution,
                            'execution_predicted': predicted_execution,
                            'download_actual': actual_total_download,
                            'download_predicted': predicted_total_download,
                            'upload_actual': actual_total_upload,
                            'upload_predicted': predicted_total_upload,
                            'input_size_actual': actual_input_size,
                            'input_size_predicted': predicted_input_size_bytes,
                            'output_size_actual': actual_output_size,
                            'output_size_predicted': predicted_output_size,
                            'worker_startup_time_actual': actual_worker_startup_time_s,
                            'worker_startup_time_predicted': predicted_worker_startup_time_s,
                        })

                # --- Calculate medians per planner per SLA ---
                planner_median_metrics = {}
                for (planner_name, sla_value), data_points in planner_metrics.items():
                    if not data_points:
                        continue
                    if 'wukong' in planner_name.lower():
                        continue

                    planner_median_metrics[(planner_name, sla_value)] = {
                        'Makespan (s)': {
                            'actual': np.median([m['makespan_actual'] for m in data_points]),
                            'predicted': np.median([m['makespan_predicted'] for m in data_points])
                        },
                        'Execution Time (s)': {
                            'actual': np.median([m['execution_actual'] for m in data_points]),
                            'predicted': np.median([m['execution_predicted'] for m in data_points])
                        },
                        'Download Time (s)': {
                            'actual': np.median([m['download_actual'] for m in data_points]),
                            'predicted': np.median([m['download_predicted'] for m in data_points])
                        },
                        'Upload Time (s)': {
                            'actual': np.median([m['upload_actual'] for m in data_points]),
                            'predicted': np.median([m['upload_predicted'] for m in data_points])
                        },
                        'Input Size (bytes)': {
                            'actual': np.median([m['input_size_actual'] for m in data_points]),
                            'predicted': np.median([m['input_size_predicted'] for m in data_points])
                        },
                        'Output Size (bytes)': {
                            'actual': np.median([m['output_size_actual'] for m in data_points]),
                            'predicted': np.median([m['output_size_predicted'] for m in data_points])
                        },
                        'Worker Startup Time (s)': {
                            'actual': np.median([m['worker_startup_time_actual'] for m in data_points]),
                            'predicted': np.median([m['worker_startup_time_predicted'] for m in data_points])
                        }
                    }

                # --- Prepare DataFrame for Plotly ---
                plot_data = []
                for (planner_name, sla_value), metrics in planner_median_metrics.items():
                    for metric_name, values in metrics.items():
                        plot_data.append({
                            'Planner': planner_name,
                            'Metric': metric_name,
                            'Type': 'Actual',
                            'Value': values['actual'],
                            'SLA': sla_value
                        })
                        plot_data.append({
                            'Planner': planner_name,
                            'Metric': metric_name,
                            'Type': 'Predicted',
                            'Value': values['predicted'],
                            'SLA': sla_value
                        })

                if plot_data:
                    df_plot = pd.DataFrame(plot_data)
                    df_plot['SLA'] = pd.to_numeric(df_plot['SLA'], errors='coerce')

                    # --- Sort SLA (ascending: lower at top) ---
                    sla_order = sorted(df_plot['SLA'].unique())

                    # --- Sort planners globally by median makespan ---
                    planner_order = sorted(df_plot['Planner'].unique())

                    # --- Plot ---
                    fig = px.bar(
                        df_plot,
                        x='Planner',
                        y='Value',
                        color='Type',
                        facet_row='SLA',      # one SLA per row
                        facet_col='Metric',   # one metric per column
                        barmode='overlay',
                        opacity=0.6,
                        title='Actual vs Predicted Metrics (per SLA and Planner)',
                        category_orders={
                            'SLA': sla_order,
                            'Planner': planner_order
                        },
                        color_discrete_map={'Actual': '#1f77b4', 'Predicted': '#ff7f0e'}
                    )

                    # Layout improvements
                    fig.update_layout(
                        height=1400,
                        legend_title='',
                        plot_bgcolor='rgba(0,0,0,0)',
                        yaxis_type='log'
                    )

                    fig.update_xaxes(tickangle=-45)
                    fig.update_traces(
                        texttemplate='%{y:.2f}',
                        textposition='outside',
                        textfont_size=9
                    )

                    st.plotly_chart(fig, use_container_width=True)

            
            # Add prediction accuracy evolution chart
            # st.markdown("### Prediction Accuracy Evolution")
            
            # Collect all data
            data = []
            sla_results = []

            for workflow_name, workflow in workflow_types.items():
                # Sort instances by start time for SLA calculation
                sorted_instances = sorted(workflow.instances, key=lambda inst: inst.start_time_ms)
                
                # Keep track of history for each metric
                history = {
                    'makespan': [], 
                    'execution': [], 
                    'download': [], 
                    'upload': [], 
                    'input_size': [], 
                    'output_size': [], 
                    'worker_startup_time': []
                }
                
                for instance in sorted_instances:
                    if not instance.plan or not instance.tasks:
                        continue
                    planner_name = instance.plan.planner_name.lower()
                    if 'wukong' in planner_name:
                        continue
                    if not instance.plan.sla or instance.plan.sla.value is None:
                        continue

                    sla_value = instance.plan.sla.value

                    # Actual metrics
                    actual_metrics = {
                        'makespan': (
                            max([
                                (task.metrics.started_at_timestamp_s * 1000) +
                                (task.metrics.input_metrics.tp_total_time_waiting_for_inputs_ms or 0) +
                                (task.metrics.tp_execution_time_ms or 0) +
                                (task.metrics.output_metrics.tp_time_ms or 0) +
                                (task.metrics.total_invocation_time_ms or 0)
                                for task in instance.tasks
                            ]) - instance.start_time_ms
                        ) / 1000,
                        'execution': sum(task.metrics.tp_execution_time_ms or 0 for task in instance.tasks) / 1000,
                        'download': sum(
                            sum(input_metric.time_ms for input_metric in task.metrics.input_metrics.input_download_metrics.values() if input_metric.time_ms is not None)
                            for task in instance.tasks
                        ) / 1000,
                        'upload': sum(task.metrics.output_metrics.tp_time_ms or 0 for task in instance.tasks) / 1000,
                        'input_size': sum(
                            sum(input_metric.serialized_size_bytes for input_metric in task.metrics.input_metrics.input_download_metrics.values()) +
                            (task.metrics.input_metrics.hardcoded_input_size_bytes or 0)
                            for task in instance.tasks
                        ),
                        'output_size': sum(
                            task.metrics.output_metrics.serialized_size_bytes
                            for task in instance.tasks if hasattr(task.metrics, 'output_metrics')
                        ),
                        'worker_startup_time': sum(
                            (metric.end_time_ms - metric.start_time_ms) / 1000
                            for metric in st.session_state.worker_startup_metrics
                            if metric.master_dag_id == instance.master_dag_id and metric.end_time_ms is not None
                        )
                    }

                    # Predicted metrics
                    predicted_metrics = {}
                    if instance.plan and instance.plan.nodes_info:
                        sink_node = instance.dag.sink_node.id.get_full_id()
                        workers_accounted_for = set()
                        predicted_worker_startup_time_s = 0
                        for info in instance.plan.nodes_info.values():
                            worker_id = info.node_ref.worker_config.worker_id
                            if worker_id is not None and worker_id not in workers_accounted_for:
                                predicted_worker_startup_time_s += info.tp_worker_startup_time_ms / 1000
                                workers_accounted_for.add(worker_id)

                        predicted_metrics = {
                            'makespan': instance.plan.nodes_info[sink_node].task_completion_time_ms / 1000,
                            'execution': sum(info.tp_exec_time_ms / 1000 for info in instance.plan.nodes_info.values()),
                            'download': sum(info.total_download_time_ms / 1000 for info in instance.plan.nodes_info.values()),
                            'upload': sum(info.tp_upload_time_ms / 1000 for info in instance.plan.nodes_info.values()),
                            'input_size': sum(info.serialized_input_size for info in instance.plan.nodes_info.values()),
                            'output_size': sum(info.serialized_output_size for info in instance.plan.nodes_info.values()),
                            'worker_startup_time': predicted_worker_startup_time_s
                        }

                    # Append to data with relative error calculated
                    for metric in actual_metrics:
                        actual = actual_metrics[metric]
                        predicted = predicted_metrics.get(metric, 0)
                        relative_error = (abs(predicted - actual) / actual * 100) if actual != 0 else 0

                        data.append({
                            'planner': planner_name,
                            'metric': metric,
                            'sla': sla_value,
                            'actual': actual,
                            'predicted': predicted,
                            'relative_error': relative_error
                        })
                        
                        # Calculate SLA fulfillment
                        prev_values = history[metric]
                        if prev_values:
                            threshold = np.percentile(prev_values, sla_value)
                            success = actual <= threshold
                            sla_results.append({
                                'sla': sla_value,
                                'metric': metric,
                                'success': success
                            })
                        
                        # Update history
                        history[metric].append(actual)

            # Create DataFrames
            df = pd.DataFrame(data)
            df_sla = pd.DataFrame(sla_results)

            # Compute median relative error grouped by metric and SLA
            error_summary = df.groupby(['metric', 'sla'])['relative_error'].median().reset_index()

            # Compute SLA fulfillment rate
            if not df_sla.empty:
                sla_summary = df_sla.groupby(['sla', 'metric'])['success'].mean().reset_index()
                sla_summary['fulfillment_rate'] = sla_summary['success'] * 100
                sla_summary = sla_summary[['metric', 'sla', 'fulfillment_rate']]
                
                # Merge with error summary
                error_summary = error_summary.merge(sla_summary, on=['metric', 'sla'], how='left')
            else:
                error_summary['fulfillment_rate'] = 0

            # Define metric order
            metric_order = ['makespan', 'execution', 'download', 'upload', 'input_size', 'output_size', 'worker_startup_time']
            error_summary['metric'] = pd.Categorical(error_summary['metric'], categories=metric_order, ordered=True)
            error_summary = error_summary.sort_values('metric')

            # Sort SLA numerically
            sla_order = sorted(error_summary['sla'].unique())
            error_summary['sla'] = error_summary['sla'].astype(str)

            # Create figure with secondary y-axis
            from plotly.subplots import make_subplots
            import plotly.graph_objects as go

            fig = make_subplots(specs=[[{"secondary_y": True}]])

            # Add bars for relative error
            for sla_val in [str(s) for s in sla_order]:
                sla_data = error_summary[error_summary['sla'] == sla_val]
                fig.add_trace(
                    go.Bar(
                        name=f'P{sla_val} Error',
                        x=sla_data['metric'],
                        y=sla_data['relative_error'],
                        text=sla_data['relative_error'].apply(lambda v: f'{v:.1f}%'),
                        textposition='outside',
                        textfont=dict(size=14),  # 🔹 Make bar text bigger
                        legendgroup=sla_val,
                        showlegend=True
                    ),
                    secondary_y=False
                )

            # Add line for SLA fulfillment rate
            for sla_val in [str(s) for s in sla_order]:
                sla_data = error_summary[error_summary['sla'] == sla_val]
                fig.add_trace(
                    go.Scatter(
                        name=f'P{sla_val} Fulfillment',
                        x=sla_data['metric'],
                        y=sla_data['fulfillment_rate'],
                        mode='lines+markers+text',
                        marker=dict(size=10),
                        line=dict(width=3, dash='dash'),
                        text=sla_data['fulfillment_rate'].apply(lambda v: f'{v:.1f}%'),
                        textposition='top center',
                        textfont=dict(size=13),  # 🔹 Make line text bigger too
                        legendgroup=sla_val,
                        showlegend=True
                    ),
                    secondary_y=True
                )

            # Update layout
            fig.update_layout(
                title='Predictions Relative Error and SLA Fulfillment (per Metric and SLA)',
                xaxis_title='Metric',
                height=550,
                width=900,
                barmode='group',
                legend=dict(
                    orientation='v',
                    yanchor='top',
                    y=1,
                    xanchor='left',
                    x=1.05
                ),
            )

            # 🔹 Improve X-axis readability and label size
            fig.update_xaxes(
                tickangle=-35,  # Diagonal labels
                tickfont=dict(size=14)  # Bigger text for metrics
            )

            # 🔹 Improve Y-axes text sizes
            fig.update_yaxes(title_text='Median Relative Error (%)', secondary_y=False, tickfont=dict(size=13))
            fig.update_yaxes(title_text='SLA Fulfillment Rate (%)', range=[0, 110], secondary_y=True, tickfont=dict(size=13))

            # Display chart
            st.plotly_chart(fig, use_container_width=False)
            
        with TAB_ACTUAL_VALUES:
            # Prepare data for all metrics comparison
            metrics_data = []

            for instance in workflow_types[selected_workflow].instances:
                if not instance.plan or not instance.tasks:
                    continue

                # Calculate all metrics for this instance
                sink_task_metrics = [t for t in instance.tasks if t.internal_task_id == instance.dag.sink_node.id.get_full_id()][0].metrics

                # Calculate makespan
                sink_task_ended_timestamp_ms = (
                    sink_task_metrics.started_at_timestamp_s * 1000 +
                    (sink_task_metrics.input_metrics.tp_total_time_waiting_for_inputs_ms or 0) +
                    (sink_task_metrics.tp_execution_time_ms or 0) +
                    (sink_task_metrics.output_metrics.tp_time_ms or 0) +
                    (sink_task_metrics.total_invocation_time_ms or 0)
                )

                # Calculate total prewarms for this instance
                total_prewarms = sum(task.optimization_prewarms_done for task in instance.tasks)
                total_preloads = sum(task.optimization_preloads_done for task in instance.tasks)
                
                # **Calculate unique workers for this instance**
                unique_workers = len(set(
                    task.metrics.worker_resource_configuration.worker_id
                    for task in instance.tasks
                    if hasattr(task.metrics, 'worker_resource_configuration')
                ))

                # Create instance metrics dictionary
                instance_metrics = {
                    'Makespan [s]': (sink_task_ended_timestamp_ms - instance.start_time_ms) / 1000,
                    'Execution Time [s]': sum(task.metrics.tp_execution_time_ms / 1000 for task in instance.tasks),
                    'Total Time Waiting for Inputs [s]': sum(
                        (task.metrics.input_metrics.tp_total_time_waiting_for_inputs_ms or 0) / 1000
                        for task in instance.tasks
                    ),
                    'Download Time [s]': sum(
                        sum(input_metric.time_ms / 1000
                            for input_metric in task.metrics.input_metrics.input_download_metrics.values()
                            if input_metric.time_ms is not None)
                        for task in instance.tasks
                    ),
                    'Upload Time [s]': sum(
                        task.metrics.output_metrics.tp_time_ms / 1000
                        for task in instance.tasks
                        if task.metrics.output_metrics.tp_time_ms is not None
                    ),
                    'Total Data Transferred': instance.total_transferred_data_bytes,
                    'Worker Startup Time [s]': instance.total_worker_startup_time_ms / 1000,
                    'Resource Usage': instance.resource_usage.gb_seconds,
                    'Total Prewarms': total_prewarms,
                    'Total Preloads': total_preloads,
                }

                # Add all metrics to the data list
                for metric_name, value in instance_metrics.items():
                    metrics_data.append({
                        'Metric': metric_name,
                        'Value': value,
                        'Planner': instance.plan.planner_name if instance.plan else 'No Planner',
                        'SLA': instance.plan.sla.value if instance.plan and hasattr(instance.plan, 'sla') and hasattr(instance.plan.sla, 'value') else 'No SLA',
                        'Instance ID': instance.master_dag_id.split('-')[0],
                        'Total Prewarms': total_prewarms,
                        'Total Preloads': total_preloads,
                        'Unique Workers': unique_workers
                    })

            st.markdown("### Metrics Comparison")
            if metrics_data:
                import math
                # Convert metrics_data to DataFrame
                df_metrics = pd.DataFrame(metrics_data)

                # Sort DataFrame alphabetically by Metric
                df_metrics = df_metrics.sort_values(by='Metric', key=lambda col: col.str.lower())

                # Get sorted list of metric names
                sorted_metrics = df_metrics['Metric'].unique()

                # Create a list of (df_for_metric, metric_name) for each metric
                all_metrics_to_plot = []
                for metric in sorted_metrics:
                    df_metric = df_metrics[df_metrics['Metric'] == metric]
                    all_metrics_to_plot.append((df_metric, metric))

                # Plot in 3 columns per row
                num_cols = 2
                num_rows = math.ceil(len(all_metrics_to_plot) / num_cols)

                for i in range(num_rows):
                    cols = st.columns(num_cols)
                    for j in range(num_cols):
                        idx = i * num_cols + j
                        if idx >= len(all_metrics_to_plot):
                            break
                        df_plot, metric_name = all_metrics_to_plot[idx]
                        if df_plot.empty:
                            cols[j].write(f"No data for {metric_name}")
                            continue

                        fig = px.box(
                            df_plot,
                            x='Planner',
                            y='Value',
                            color='Planner',
                            points="all",
                            hover_data=['Instance ID'],
                            title=f"{metric_name} Distribution (per Planner)",
                            category_orders={"Planner": sorted(df_plot['Planner'].unique())}
                        )

                        # Calculate medians per Planner
                        medians = df_plot.groupby('Planner')['Value'].median()

                        # Add scatter for median to make it visually prominent
                        for planner, median_value in medians.items():
                            fig.add_trace(
                                go.Scatter(
                                    x=[planner],
                                    y=[median_value],
                                    mode='markers+text',
                                    marker=dict(color='black', symbol='diamond', size=12),
                                    text=[f"{median_value:.2f}"],
                                    textposition='top center',
                                    showlegend=False
                                )
                            )

                        fig.update_layout(
                            plot_bgcolor='rgba(0,0,0,0)',
                            boxmode='group',
                            height=500,
                            legend_title="Planner",
                            xaxis_title="Planner",
                            yaxis_title="Value",
                            boxgap=0.001,
                            boxgroupgap=0.001
                        )

                        cols[j].plotly_chart(fig, use_container_width=True)


                # ---------------------                        
                # Add pie charts for time distribution by activity for each planner
                st.markdown("### Time Breakdown Analysis")

                time_metrics = ['Worker Startup Time [s]', 'Total Time Waiting for Inputs [s]', 'Execution Time [s]', 'Upload Time [s]']
                time_metrics_df = df_metrics[df_metrics['Metric'].isin(time_metrics)]

                if not time_metrics_df.empty:
                    # Calculate median unique workers for each Planner
                    median_workers = df_metrics.groupby(['Planner'])['Unique Workers'].median().reset_index()
                    median_workers = median_workers.rename(columns={'Unique Workers': 'Median Unique Workers'})

                    # Group by Planner and Metric, calculate median
                    bar_data = time_metrics_df.groupby(['Planner', 'Metric'])['Value'].median().reset_index()
                    
                    # Pivot table for stacked bar chart
                    df_bar = bar_data.pivot_table(index='Planner', columns='Metric', values='Value').reset_index()
                    
                    # Merge median workers
                    df_bar = pd.merge(df_bar, median_workers, on='Planner', how='left')
                    df_bar['Median Unique Workers'] = df_bar['Median Unique Workers'].fillna(0).astype(int)
                    df_bar = df_bar.fillna(0)

                    metric_cols_ordered = [m for m in time_metrics if m in df_bar.columns]

                    # Create stacked bar chart
                    fig_bar = px.bar(
                        df_bar,
                        x='Planner',
                        y=metric_cols_ordered,
                        title="Time Breakdown Analysis (per Planner)",
                        labels={'value': 'Time (s)', 'Planner': 'Planner'},
                        category_orders={"Planner": sorted(df_bar['Planner'].unique())},
                        custom_data=['Median Unique Workers']
                    )

                    fig_bar.update_layout(
                        barmode='stack',
                        plot_bgcolor='rgba(0,0,0,0)',
                        height=500,
                        width=600,
                        legend_title='Activity',
                        yaxis_title='Time (s)'
                    )

                    # Custom hover templates
                    for trace in fig_bar.data:
                        template = '<b>Planner:</b> %{x}<br><b>Time (s):</b> %{y:.2f}s<extra></extra>'
                        if trace.name == 'Worker Startup Time [s]':
                            template = '<b>Planner:</b> %{x}<br><b>Time (s):</b> %{y:.2f}s<br><b>Median Unique Workers:</b> %{customdata[0]}<extra></extra>'
                        trace.hovertemplate = template

                    st.plotly_chart(fig_bar, use_container_width=False)
                else:
                    st.write("No time metrics data available to display.")
            
            # Calculate metrics by planner type
            # Collect metrics per planner
            planner_metrics = {}

            for instance in workflow_types[selected_workflow].instances:
                if not instance.plan or not instance.tasks: continue

                planner = instance.plan.planner_name if instance.plan else 'Unknown'
                if planner not in planner_metrics:
                    # store lists to compute median later
                    planner_metrics[planner] = {
                        'makespan': [],
                        'execution': [],
                        'download': [],
                        'upload': [],
                        'input_size': [],
                        'output_size': [],
                        'data_transferred': [],
                        'data_size_uploaded': [],
                        'data_size_downloaded': [],
                        'invocation': [],
                        'dependency_update': [],
                        'dag_download': [],
                        'worker_startup': [],
                        'resource_usage': [],
                        'warm_starts': [],
                        'cold_starts': [],
                        'total_time_waiting_for_inputs': [],
                        'unique_workers': []
                    }
                
                metrics = planner_metrics[planner]
                
                sink_task_metrics = [t for t in instance.tasks if t.internal_task_id == instance.dag.sink_node.id.get_full_id()][0].metrics
                sink_task_ended_timestamp_ms = (
                    sink_task_metrics.started_at_timestamp_s * 1000
                    + (sink_task_metrics.input_metrics.tp_total_time_waiting_for_inputs_ms or 0)
                    + (sink_task_metrics.tp_execution_time_ms or 0)
                    + (sink_task_metrics.output_metrics.tp_time_ms or 0)
                    + (sink_task_metrics.total_invocation_time_ms or 0)
                )
                actual_makespan_s = (sink_task_ended_timestamp_ms - instance.start_time_ms) / 1000
                total_time_waiting_for_inputs_s = sink_task_metrics.input_metrics.tp_total_time_waiting_for_inputs_ms / 1000 if sink_task_metrics.input_metrics.tp_total_time_waiting_for_inputs_ms else 0
                actual_unique_workers_count = len(set([
                        task.metrics.worker_resource_configuration.worker_id
                        for task in instance.tasks
                        if task.metrics.worker_resource_configuration.worker_id is not None
                    ]))

                metrics['makespan'].append(actual_makespan_s)
                metrics['execution'].append(sum(task.metrics.tp_execution_time_ms / 1000 for task in instance.tasks))
                metrics['download'].append(sum(
                    sum(input_metric.time_ms / 1000 for input_metric in task.metrics.input_metrics.input_download_metrics.values() if input_metric.time_ms is not None)
                    for task in instance.tasks
                ))
                metrics['upload'].append(sum(
                    task.metrics.output_metrics.tp_time_ms / 1000
                    for task in instance.tasks
                    if task.metrics.output_metrics.tp_time_ms is not None
                ))
                metrics['input_size'].append(sum(
                    sum(input_metric.serialized_size_bytes for input_metric in task.metrics.input_metrics.input_download_metrics.values())
                    for task in instance.tasks
                ))
                metrics['output_size'].append(sum(
                    task.metrics.output_metrics.serialized_size_bytes for task in instance.tasks
                ))
                metrics['data_transferred'].append(instance.total_transferred_data_bytes)
                metrics['invocation'].append(sum(
                    task.metrics.total_invocation_time_ms / 1000
                    for task in instance.tasks
                    if task.metrics.total_invocation_time_ms is not None
                ))
                metrics['dependency_update'].append(sum(
                    task.metrics.update_dependency_counters_time_ms / 1000
                    for task in instance.tasks
                    if hasattr(task.metrics, 'update_dependency_counters_time_ms') and task.metrics.update_dependency_counters_time_ms is not None
                ))
                metrics['total_time_waiting_for_inputs'].append(total_time_waiting_for_inputs_s)
                metrics['worker_startup'].append(instance.total_worker_startup_time_ms / 1000)
                metrics['resource_usage'].append(instance.resource_usage.gb_seconds)
                metrics['data_size_uploaded'].append(sum(task.metrics.output_metrics.serialized_size_bytes for task in instance.tasks))
                metrics['data_size_downloaded'].append(sum(
                    sum(input_metric.serialized_size_bytes for input_metric in task.metrics.input_metrics.input_download_metrics.values() if input_metric.time_ms is not None)
                    for task in instance.tasks
                ))
                metrics['warm_starts'].append(instance.warm_starts_count)
                metrics['cold_starts'].append(instance.cold_starts_count)
                metrics['unique_workers'].append(actual_unique_workers_count)

            # Compute median per planner

            # Prepare plot data with median and std
            plot_data = []
            # The user asked to keep network_data intact, so we'll leave its logic as is,
            # even though it's not used in the new plotting scheme.
            network_data = [] 

            for planner_name, metrics in planner_metrics.items():
                metric_names = [
                    ('Makespan (s)', 'makespan'),
                    ('Execution Time (s)', 'execution'),
                    ('Download Time (s)', 'download'),
                    ('Upload Time (s)', 'upload'),
                    ('Data Transferred (GB)', 'data_transferred'),
                    ('Total Time Waiting for Inputs [s]', 'total_time_waiting_for_inputs'),
                    ('Task Invocation Time (s)', 'invocation'),
                    ('Dependency Counter Update Time (s)', 'dependency_update'),
                    ('Worker Startup Time (s)', 'worker_startup'),
                    ('Resource Usage (GB-seconds)', 'resource_usage'),
                    ('Data Size Uploaded (MB)', 'data_size_uploaded'),
                    ('Data Size Downloaded (MB)', 'data_size_downloaded'),
                    ('Warm Starts', 'warm_starts'),
                    ('Cold Starts', 'cold_starts'),
                    ('Unique Workers', 'unique_workers'),
                ]
                
                for display_name, key in metric_names:
                    # Ensure the key exists and the list is not empty before calculating median
                    if key in metrics and metrics[key]:
                        median_val = np.median(metrics[key])
                        
                        # Convert units where needed
                        if display_name == 'Data Transferred (GB)':
                            median_val /= 1024**3
                        if display_name in ['Data Size Uploaded (MB)', 'Data Size Downloaded (MB)']:
                            median_val /= 1024**2
                        
                        plot_data.append({
                            'Planner': planner_name,
                            'Metric': display_name,
                            'Value': median_val,
                        })

                        # This section remains unchanged as requested
                        if display_name == 'Data Size Uploaded (MB)':
                            network_data.append({
                                'Planner': planner_name, 'Type': 'Upload (MB)', 'Value': median_val
                            })
                        if display_name == 'Data Size Downloaded (MB)':
                            network_data.append({
                                'Planner': planner_name, 'Type': 'Download (MB)', 'Value': median_val
                            })

            if plot_data:
                df_plot = pd.DataFrame(plot_data)
                
                # **1. Create a sorted list of planner names**
                sorted_planners = sorted(df_plot['Planner'].unique())
                
                # Create a dropdown menu to select the metric
                st.markdown("### Metric Comparison (Median)")
                selected_metric = st.selectbox(
                    'Select a metric to display',
                    options=sorted(df_plot['Metric'].unique())
                )

                # Filter the DataFrame based on the selection
                df_filtered = df_plot[df_plot['Metric'] == selected_metric]
                
                # Create a bar chart for the selected metric
                if not df_filtered.empty:
                    fig = px.bar(
                        df_filtered,
                        x='Planner',
                        y='Value',
                        color='Planner',
                        title=f'{selected_metric} (per Planner)',
                        labels={'Value': 'Median Value', 'Planner': 'Planner'},
                        # **2. Use the sorted list to enforce the order**
                        category_orders={'Planner': sorted_planners}
                    )

                    # Add median value text above bars
                    fig.update_traces(
                        texttemplate='%{y:.3f}',
                        textposition='outside',
                    )

                    fig.update_layout(
                        xaxis_title='Planner',
                        yaxis_title='Median Value',
                        legend_title='Planner',
                        plot_bgcolor='rgba(0,0,0,0)',
                        height=600,
                        # **3. The line for sorting by value has been removed**
                    )

                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No data available to plot.")

            st.markdown("## Optimizations")

            # Collect prewarm data
            prewarm_data = []

            for instance in workflow_types[selected_workflow].instances:
                if not instance.plan:
                    continue
                
                planner_name = instance.plan.planner_name.lower()
                
                # Count prewarms for this instance
                total_prewarms = sum(task.optimization_prewarms_done for task in instance.tasks)
                successful_prewarms = sum(task.optimization_prewarms_successful for task in instance.tasks)
                
                if total_prewarms > 0:
                    prewarm_data.append({
                        'workflow': selected_workflow,
                        'planner': planner_name,
                        'total_prewarms': total_prewarms,
                        'successful_prewarms': successful_prewarms,
                        'failed_prewarms': total_prewarms - successful_prewarms,
                        'success_rate': (successful_prewarms / total_prewarms * 100)
                    })

            if prewarm_data:
                df = pd.DataFrame(prewarm_data)
                
                # Sort planners alphabetically
                sorted_planners = sorted(df['planner'].unique())
                
                # Calculate summary statistics per planner
                summary = df.groupby('planner').agg({
                    'total_prewarms': 'sum',
                    'successful_prewarms': 'sum',
                    'failed_prewarms': 'sum'
                }).reset_index()
                
                summary['success_rate'] = (summary['successful_prewarms'] / summary['total_prewarms'] * 100)
                summary['planner'] = pd.Categorical(summary['planner'], categories=sorted_planners, ordered=True)
                summary = summary.sort_values('planner')
                
                # Melt for stacked bar chart
                summary_melted = summary.melt(
                    id_vars=['planner', 'success_rate'],
                    value_vars=['successful_prewarms', 'failed_prewarms'],
                    var_name='status',
                    value_name='count'
                )
                
                # Create stacked bar chart
                fig = px.bar(
                    summary_melted,
                    x='planner',
                    y='count',
                    color='status',
                    title='Prewarm Success and Failure Counts by Planner',
                    labels={'count': 'Number of Prewarms', 'planner': 'Planner', 'status': 'Status'},
                    color_discrete_map={
                        'successful_prewarms': '#2ecc71',
                        'failed_prewarms': '#e74c3c'
                    },
                    text='count'
                )
                
                # Add success rate as annotations
                for i, row in summary.iterrows():
                    fig.add_annotation(
                        x=row['planner'],
                        y=row['total_prewarms'],
                        text=f"{row['success_rate']:.1f}%",
                        showarrow=False,
                        yshift=10,
                        font=dict(size=12, color='black', family='Arial Black')
                    )
                
                fig.update_layout(
                    xaxis_title='Planner',
                    yaxis_title='Number of Prewarms',
                    height=500,
                    showlegend=True,
                    legend_title='Status',
                    barmode='stack'
                )
                
                fig.update_traces(textposition='inside', textfont_size=10)
                fig.update_xaxes(tickangle=-45)
                
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("## Resource Usage")
            ######### Resource Usage plot
            # Collect resource usage metrics per planner
            resource_data = []

            for instance in workflow_types[selected_workflow].instances:
                if not instance.plan or not instance.tasks:
                    continue
                
                planner = instance.plan.planner_name if instance.plan else 'Unknown'
                usage = instance.resource_usage

                resource_data.append({
                    'Planner': planner,
                    'Metric': 'CPU Time (s)',
                    'Value': usage.cpu_seconds
                })
                resource_data.append({
                    'Planner': planner,
                    'Metric': 'GB-seconds',
                    'Value': usage.gb_seconds
                })

            ######## Network I/O
            df_network = pd.DataFrame(network_data)

            # Sort planners alphabetically (or use custom order)
            sorted_planners = sorted(df_network['Planner'].unique())

            # Create side-by-side bar chart
            fig = px.bar(
                df_network,
                x='Planner',
                y='Value',
                color='Type',
                barmode='group',  # side-by-side bars
                title='Network I/O (Median)',
                labels={'Value': 'MB', 'Planner': 'Planner', 'Type': 'I/O Type'},
                category_orders={'Planner': sorted_planners}  # consistent order
            )

            # Layout improvements
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                yaxis_title='Data (MB)',
                height=500
            )

            # Show values on top of bars
            fig.update_traces(
                texttemplate='%{y:.2f}',
                textposition='outside',
                textfont_size=9
            )

            st.plotly_chart(fig, use_container_width=True)

            df_resource = pd.DataFrame(resource_data)

            # Plot each metric separately
            metrics = ['CPU Time (s)', 'GB-seconds']

            cols = st.columns(2)

            for i, metric in enumerate(metrics):
                df_metric = df_resource[df_resource['Metric'] == metric]
                
                fig = px.box(
                    df_metric,
                    x='Planner',
                    y='Value',
                    color='Planner',
                    points='all',
                    title=f'{metric} Distribution',
                    labels={'Value': metric, 'Planner': 'Planner'},
                    category_orders={'Planner': sorted_planners}
                )
                
                # Calculate median per Planner and add as scatter points
                medians = df_metric.groupby('Planner')['Value'].median()
                for planner, median_value in medians.items():
                    fig.add_trace(
                        go.Scatter(
                            x=[planner],
                            y=[median_value],
                            mode='markers+text',
                            marker=dict(color='black', symbol='diamond', size=12),
                            text=[f"{median_value:.2f}"],
                            textposition='top center',
                            showlegend=False
                        )
                    )
                
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=400,
                    showlegend=False
                )
                
                # Place the plot in the correct column
                cols[i % 2].plotly_chart(fig, use_container_width=True)
                
                # Move to a new row after 2 plots
                if i % 2 == 1:
                    cols = st.columns(2)


            ##########
            
            # manual_order = ["NonUniformPlanner", "UniformPlanner", "WUKONGPlanner"]
            # manual_order = ["NonUniformPlanner-opt", "UniformPlanner-opt", "WUKONGPlanner-opt"]
            manual_order = [
                "NonUniformPlanner", "UniformPlanner", "WUKONGPlanner",
                "NonUniformPlanner-opt", "UniformPlanner-opt", "WUKONGPlanner-opt"
            ]

            data = []
            for instance in workflow_types[selected_workflow].instances:
                if not instance.plan:
                    continue
                planner_name = instance.plan.planner_name
                if planner_name not in manual_order:
                    continue

                sink_task_metrics = [
                    t for t in instance.tasks
                    if t.internal_task_id == instance.dag.sink_node.id.get_full_id()
                ][0].metrics

                sink_task_ended_timestamp_ms = (
                    sink_task_metrics.started_at_timestamp_s * 1000 +
                    (sink_task_metrics.input_metrics.tp_total_time_waiting_for_inputs_ms or 0) +
                    (sink_task_metrics.tp_execution_time_ms or 0) +
                    (sink_task_metrics.output_metrics.tp_time_ms or 0) +
                    (sink_task_metrics.total_invocation_time_ms or 0)
                )

                actual_makespan_s = (sink_task_ended_timestamp_ms - instance.start_time_ms) / 1000
                ru = instance.resource_usage

                data.append({
                    "workflow": selected_workflow,
                    "makespan": actual_makespan_s,
                    "planner": planner_name,
                    "GB-seconds": ru.gb_seconds
                })

            df = pd.DataFrame(data)

            # Sort planners by custom manual order
            sorted_planners = sorted(
                df["planner"].unique(),
                key=lambda x: manual_order.index(x) if x in manual_order else len(manual_order)
            )

            # Calculate median per planner
            df_summary = df.groupby("planner").agg({
                "makespan": "median",
                "GB-seconds": "median",
            }).reset_index()

            # === Chart 1: Makespan ===
            fig_makespan = px.bar(
                df_summary,
                x="planner",
                y="makespan",
                text=df_summary["makespan"].apply(lambda v: f"{v:.2f}"),
                color="planner",
                category_orders={"planner": sorted_planners},
                labels={"planner": "Planner", "makespan": "Median Makespan (s)"},
                title="Makespan (per Planner)"
            )
            fig_makespan.update_traces(
                textposition="outside",
                textfont=dict(size=14)  # 🔹 Increase text on top of bars
            )
            fig_makespan.update_layout(
                xaxis_title="Planner",
                yaxis_title="Median Makespan (s)",
                height=650,
                width=650,
                legend_title="Planner",
                xaxis=dict(tickfont=dict(size=14)),  # 🔹 Increase x-axis labels
                yaxis=dict(tickfont=dict(size=12)),
                title_font=dict(size=20),
                bargap=0.05,          # 🔹 Decrease gap between bars
                bargroupgap=0.02,      # 🔹 Decrease gap between groups of bars
                showlegend=False
            )

            # === Chart 2: Resource Usage (GB-seconds) ===
            fig_resource = px.bar(
                df_summary,
                x="planner",
                y="GB-seconds",
                text=df_summary["GB-seconds"].apply(lambda v: f"{v:.2f}"),
                color="planner",
                category_orders={"planner": sorted_planners},
                labels={"planner": "Planner", "GB-seconds": "Median GB-seconds"},
                title="Resource Usage (per Planner)"
            )
            fig_resource.update_traces(
                textposition="outside",
                textfont=dict(size=14)  # 🔹 Bigger value text
            )
            fig_resource.update_layout(
                xaxis_title="Planner",
                yaxis_title="Median GB-seconds",
                height=650,
                width=650,
                legend_title="Planner",
                xaxis=dict(tickfont=dict(size=14)),  # 🔹 Bigger x-axis planner names
                yaxis=dict(tickfont=dict(size=12)),
                title_font=dict(size=20),
                bargap=0.05,          # 🔹 Decrease gap between bars
                bargroupgap=0.02,      # 🔹 Decrease gap between groups of bars
                showlegend=False
            )

            # === Display stacked vertically in Streamlit ===
            st.plotly_chart(fig_makespan, use_container_width=False)
            st.plotly_chart(fig_resource, use_container_width=False)

            #############

            planner_start_stats = []
            for instance in workflow_types[selected_workflow].instances:
                if not instance.plan:
                    continue
                planner = instance.plan.planner_name

                warm = instance.warm_starts_count
                cold = instance.cold_starts_count
                total = warm + cold
                if total == 0:
                    continue

                warm_pct = warm / total * 100
                cold_pct = cold / total * 100

                planner_start_stats.append({
                    "Planner": planner,
                    "Warm": warm,
                    "Cold": cold,
                    "Warm %": warm_pct,
                    "Cold %": cold_pct
                })

            # Convert to DataFrame
            df_starts = pd.DataFrame(planner_start_stats)

            # Group by planner: take mean counts + mean percentages
            df_agg = df_starts.groupby("Planner").agg({
                "Warm": "mean",
                "Cold": "mean",
                "Warm %": "mean",
                "Cold %": "mean"
            }).reset_index()

            # Melt for stacked bar (percentages)
            df_melted = df_agg.melt(
                id_vars="Planner",
                value_vars=["Warm %", "Cold %"],
                var_name="Start Type",
                value_name="Percentage"
            )

            # Create stacked percentage bar chart
            fig = px.bar(
                df_melted,
                x="Planner",
                y="Percentage",
                color="Start Type",
                barmode="stack",
                text=df_melted["Percentage"].round(1).astype(str) + "%",
                title="Warm vs Cold Starts (per Planner)"
            )

            fig.update_traces(textposition="inside")

            # Layout tweaks
            fig.update_layout(
                xaxis_title="Planner",
                yaxis_title="Mean Percentage of Starts",
                legend_title="Start Type",
                plot_bgcolor="rgba(0,0,0,0)",
                height=650,
                yaxis=dict(range=[0, 115])  # leave space for labels above bars
            )

            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No instance data available for the selected filters.")

if __name__ == "__main__":
    asyncio.run(main())