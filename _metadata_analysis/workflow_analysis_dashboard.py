import seaborn as sns
import colorsys
from datetime import datetime
import hashlib
import os
import sys
from matplotlib import pyplot as plt
import numpy as np
from streamlit_agraph import agraph, Node, Edge, Config
import streamlit as st
import redis
import cloudpickle
import pandas as pd
import plotly.express as px
from dataclasses import dataclass
from typing import Dict, List
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.planning.optimizations.preload import PreLoadOptimization
from src.storage.prefixes import DAG_PREFIX
from src.planning.abstract_dag_planner import AbstractDAGPlanner
from src.storage.metadata.metrics_types import FullDAGPrepareTime, TaskMetrics, WorkerStartupMetrics, UserDAGSubmissionMetrics
from src.dag.dag import FullDAG
from src.dag_task_node import DAGTaskNode
from src.storage.metadata.metadata_storage import MetadataStorage

# Redis connection setup
def get_redis_connection(port: int = 6379):
    return redis.Redis(
        host='localhost',
        port=port,
        password='redisdevpwd123',
        decode_responses=False
    )

def format_bytes(size: float) -> str:
    """Convert bytes to human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size < 1024.0:
            return f"{size:.3f} {unit}"
        size /= 1024.0
    return f"{size:.3f} TB"

def get_function_group(task_id: str, func_name: str) -> str:
    """Extract the function group from task ID and function name"""
    # For tasks like "grayscale_image_part-1e783776-abe4-4e1a-8ec2-81df12192b0b"
    # we want to group them as "grayscale_image_part"
    if '-' in task_id:
        base_part = task_id.split('-')[0]
        if base_part.startswith(func_name.split('_')[0]):
            return base_part
    return func_name

@dataclass
class WorkflowInstanceInfo:
    master_dag_id: str
    dag: FullDAG
    dag_submission_metrics: UserDAGSubmissionMetrics
    planner_type: str | None


def get_workflows_information(metrics_redis: redis.Redis) -> Dict[str, Dict[str, List[WorkflowInstanceInfo]]]:
    """
    Get all workflows grouped by workflow type and planner type.
    Returns: Dict[workflow_type, Dict[planner_type, List[WorkflowInstanceInfo]]]
    """
    workflows: Dict[str, Dict[str, List[WorkflowInstanceInfo]]] = {}
    
    try:
        # Get all DAG keys
        def scan_keys(conn, pattern: str):
            """Efficiently scan Redis keys matching pattern."""
            cursor = 0
            keys = []
            while True:
                cursor, batch = conn.scan(cursor=cursor, match=pattern, count=1000)
                keys.extend(batch)
                if cursor == 0:
                    break
            return keys

        # Scan DAG keys instead of using keys()
        all_dag_keys = scan_keys(metrics_redis, f"{DAG_PREFIX}*")
        for dag_key in all_dag_keys:
            try:
                # Get DAG data
                dag_data = metrics_redis.get(dag_key)
                dag: FullDAG = cloudpickle.loads(dag_data)  # type: ignore
                
                # Get DAG submission metrics
                dag_metrics_data = metrics_redis.get(f"{MetadataStorage.USER_DAG_SUBMISSION_PREFIX}{dag.master_dag_id}")
                if not dag_metrics_data:
                    continue
                    
                dag_submission_metrics: UserDAGSubmissionMetrics = cloudpickle.loads(dag_metrics_data)  # type: ignore
                
                # Get planner type from PlanOutput.planner_name
                planner_type = "unknown"
                plan_data = metrics_redis.get(f"{MetadataStorage.PLAN_KEY_PREFIX}{dag.master_dag_id}")
                if plan_data:
                    try:
                        plan_output = cloudpickle.loads(plan_data)  # type: ignore
                        if hasattr(plan_output, 'planner_name'):
                            planner_type = plan_output.planner_name
                    except Exception as e:
                        print(f"Error extracting planner type for DAG {dag.master_dag_id}: {e}")
                
                # Initialize workflow type if not exists
                if dag.dag_name not in workflows:
                    workflows[dag.dag_name] = {}
                
                # Initialize planner type if not exists
                if planner_type not in workflows[dag.dag_name]:
                    workflows[dag.dag_name][planner_type] = []
                
                # Add workflow instance
                workflows[dag.dag_name][planner_type].append(
                    WorkflowInstanceInfo(
                        master_dag_id=dag.master_dag_id,
                        dag=dag,
                        dag_submission_metrics=dag_submission_metrics,
                        planner_type=planner_type
                    )
                )
                
            except Exception as e:
                print(f"Error processing DAG {dag_key}: {e}")
                continue
                
    except Exception as e:
        print(f"Error accessing Redis: {e}")
    
    return workflows


def main():
    # Configure page layout for wider usage
    st.set_page_config(layout="wide")
    st.title("Workflow Instance Analysis Dashboard")
    
    # Connect to Redis
    metrics_redis = get_redis_connection(6380)
    
    # Get all workflows information
    if 'workflows' not in st.session_state:
        with st.spinner('Loading workflow information...'):
            st.session_state.workflows = get_workflows_information(metrics_redis)
    
    workflows = st.session_state.workflows
    
    if not workflows:
        st.warning("No workflows found in Redis")
        return
    
    # Create three columns for the dropdowns
    col1, col2, col3 = st.columns(3)
    
    # Workflow Type Dropdown
    with col1:
        workflow_types = sorted(workflows.keys())
        selected_workflow = st.selectbox(
            "Workflow Type",
            options=workflow_types,
            index=0,
            key="selected_workflow"
        )
    
    # Planner Type Dropdown
    with col2:
        planner_types = sorted(workflows[selected_workflow].keys()) if selected_workflow in workflows else []
        selected_planner = st.selectbox(
            "Planner Type",
            options=planner_types,
            index=0 if planner_types else None,
            key="selected_planner",
            disabled=not planner_types
        )
    
    # DAG ID Dropdown
    with col3:
        workflow_instances = workflows.get(selected_workflow, {}).get(selected_planner, []) if selected_planner else []
        dag_options = [
            (f"{instance.master_dag_id} (submitted at {instance.dag_submission_metrics.dag_submission_time_ms:.2f} ms)", 
             instance.master_dag_id) 
            for instance in workflow_instances
        ]
        
        selected_dag = st.selectbox(
            "Workflow Instance (DAG ID)",
            options=[opt[1] for opt in dag_options],
            format_func=lambda x: next((opt[0] for opt in dag_options if opt[1] == x), x),
            index=0 if dag_options else None,
            key="selected_dag",
            disabled=not dag_options
        )
    
    # Get the selected DAG
    selected_dag_info = None
    if selected_workflow and selected_planner and selected_dag:
        workflow_instances = workflows.get(selected_workflow, {}).get(selected_planner, [])
        selected_dag_info = next((inst for inst in workflow_instances if inst.master_dag_id == selected_dag), None)
    
    if not selected_dag_info:
        st.warning("No DAG selected or available")
        return
    
    # Reset task selection if DAG changed
    if 'prev_dag_id' not in st.session_state or st.session_state.prev_dag_id != selected_dag:
        if 'selected_task_id' in st.session_state:
            del st.session_state.selected_task_id
        st.session_state.prev_dag_id = selected_dag
    
    # Use the selected DAG for the rest of the dashboard
    dag = selected_dag_info.dag
    dag_submission_metrics = selected_dag_info.dag_submission_metrics

    worker_startup_keys = metrics_redis.keys(f"{MetadataStorage.WORKER_STARTUP_PREFIX}*")
    worker_startup_metrics: list[WorkerStartupMetrics] = [cloudpickle.loads(metrics_redis.get(key)) for key in worker_startup_keys] # type: ignore
    total_workflow_worker_startup_time_s = sum([m.end_time_ms - m.start_time_ms for m in worker_startup_metrics if m.end_time_ms is not None]) / 1000

    # Collect all metrics for this DAG
    dag_metrics: list[TaskMetrics] = []
    total_data_transferred = 0
    total_time_executing_tasks_ms = 0
    total_time_uploading_data_ms = 0
    total_time_downloading_data_ms = 0
    total_time_invoking_tasks_ms = 0
    total_time_updating_dependency_counters_ms = 0
    task_metrics_data = []
    function_groups = set()

    _sink_task_metrics = None
    for task_id in dag._all_nodes.keys():
        metrics_key = f"{MetadataStorage.TASK_MD_KEY_PREFIX}{task_id}+{dag.master_dag_id}"
        metrics_data = metrics_redis.get(metrics_key)
        if not metrics_data: raise Exception(f"Could not find metrics for key: {metrics_key}")
        metrics: TaskMetrics = cloudpickle.loads(metrics_data) # type: ignore
        assert metrics
        dag_metrics.append(metrics)
        
        func_name = dag._all_nodes[task_id].func_name
        function_groups.add(func_name)

        if task_id == dag.sink_node.id.get_full_id():
            _sink_task_metrics = metrics

        total_time_invoking_tasks_ms += metrics.total_invocation_time_ms if metrics.total_invocation_time_ms else 0
        total_time_updating_dependency_counters_ms += metrics.update_dependency_counters_time_ms if metrics.update_dependency_counters_time_ms else 0

        # Calculate data transferred
        task_data = 0
        total_time_downloading_data_ms += sum([input_metrics.time_ms for input_metrics in metrics.input_metrics.input_download_metrics.values() if input_metrics.time_ms])
        if metrics.output_metrics:
            task_data += metrics.output_metrics.serialized_size_bytes
            total_time_uploading_data_ms += metrics.output_metrics.tp_time_ms if metrics.output_metrics.tp_time_ms else 0
        
        total_data_transferred += task_data
        total_time_executing_tasks_ms += metrics.tp_execution_time_ms if metrics.tp_execution_time_ms else 0

        downloadable_input_size_bytes = sum([input_metrics.serialized_size_bytes for input_metrics in metrics.input_metrics.input_download_metrics.values()])
        # Prepare data for visualization
        task_metrics_data.append({
            'task_id': task_id,
            'task_started_at': datetime.fromtimestamp(metrics.started_at_timestamp_s).strftime("%Y-%m-%d %H:%M:%S:%f"),
            'function_name': func_name,
            'execution_time_ms': metrics.tp_execution_time_ms,
            'worker_id': metrics.worker_resource_configuration.worker_id,
            'worker_resource_configuration_cpus': metrics.worker_resource_configuration.cpus,
            'worker_resource_configuration_ram': metrics.worker_resource_configuration.memory_mb,
            'input_size': downloadable_input_size_bytes,
            'output_size': metrics.output_metrics.serialized_size_bytes,
            'downstream_calls': metrics.total_invocations_count
        })
    
    assert _sink_task_metrics

    sink_task_ended_timestamp_ms = (_sink_task_metrics.started_at_timestamp_s * 1000) + (_sink_task_metrics.input_metrics.tp_total_time_waiting_for_inputs_ms or 0) + _sink_task_metrics.tp_execution_time_ms + (_sink_task_metrics.output_metrics.tp_time_ms or 0) + (_sink_task_metrics.total_invocation_time_ms or 0)
    makespan_ms = sink_task_ended_timestamp_ms - dag_submission_metrics.dag_submission_time_ms

    keys = metrics_redis.keys(f'{MetadataStorage.DAG_MD_KEY_PREFIX}{dag.master_dag_id}*')
    total_time_downloading_dag_ms = 0
    dag_prepare_metrics = []
    for key in keys:
        serialized_value = metrics_redis.get(key)
        deserialized = cloudpickle.loads(serialized_value) # type: ignore
        if not isinstance(deserialized, FullDAGPrepareTime): raise Exception(f"Deserialized value is not of type TaskMetrics: {type(deserialized)}")
        total_time_downloading_dag_ms += deserialized.download_time_ms
        dag_prepare_metrics.append({
            "dag_download_time": deserialized.download_time_ms,
            "create_subdag_time": deserialized.create_subdags_time_ms,
            "dag_size": deserialized.serialized_size_bytes
        })

    # Calculate task timing metrics (start times, end times)
    task_timings = {}
    
    # Find critical path (longest path from source to sink)
    def find_critical_path():
        # Perform DFS to find the longest path from source to sink
        max_path = []
        max_length = -1
        
        def dfs(node, path, current_length):
            nonlocal max_path, max_length
            node_id = node.id.get_full_id()
            
            task_metrics = next((t for t in task_metrics_data if t['task_id'] == node_id), None)
            if not task_metrics:
                return
                
            task_time = task_metrics.get('end_time_ms', 0)
            
            # Update path and length
            new_path = path + [node_id]
            new_length = current_length + task_time
            
            # If this is the sink node, check if it's the longest path
            if node_id == dag.sink_node.id.get_full_id():
                if new_length > max_length:
                    max_length = new_length
                    max_path = new_path
                return
            
            # Continue DFS to downstream nodes
            for downstream in node.downstream_nodes:
                dfs(downstream, new_path, new_length)
        
        # Start DFS from all root nodes
        for root in dag.root_nodes:
            dfs(root, [], 0)
        
        # Convert the critical path to a set for O(1) lookups
        critical_nodes = set(max_path)
        critical_edges = set(zip(max_path[:-1], max_path[1:]))
        return critical_nodes, critical_edges
            
    critical_nodes, critical_edges = find_critical_path()

    # First pass: collect all task metrics and find the minimum start time
    dag_start_timestamp_s = dag_submission_metrics.dag_submission_time_ms / 1000

    # Second pass: calculate end times relative to min_start_time
    for task_id, _metrics in zip(dag._all_nodes.keys(), dag_metrics):
        # worker startup time is considered through the {metrics.started_at_timestamp_s}
        relative_start_time_ms = (_metrics.started_at_timestamp_s - dag_start_timestamp_s) * 1000  # Convert to ms
        end_time = relative_start_time_ms + (_metrics.input_metrics.tp_total_time_waiting_for_inputs_ms or 0) + (_metrics.tp_execution_time_ms or 0) + (_metrics.output_metrics.tp_time_ms or 0) + (_metrics.total_invocation_time_ms or 0)

        task_timings[task_id] = {
            'start_time': relative_start_time_ms,
            'end_time': end_time
        }
    
    # Update task_metrics_data with the calculated timing information
    for i, task_data in enumerate(task_metrics_data):
        task_id = task_data['task_id']
        task_metrics_data[i].update({
            'relative_start_time_ms': task_timings[task_id]['start_time'],
            'end_time_ms': task_timings[task_id]['end_time']
        })
    
    # Create tabs for visualization and metrics
    tab_viz, tab_summary, tab_exec, tab_data_transfer, tab_workers, tab_critical_path = st.tabs([
        "Visualization", 
        "Summary", 
        "Execution Times", 
        "Data Transfer", 
        "Worker Distribution",
        "Critical Path Breakdown"
    ])

    # Visualization tab
    with tab_viz:
        # Create columns for graph and task details
        graph_col, details_col, planned_vs_observed_col = st.columns([0.9, 0.8, 1])
        
        def get_color_for_worker(worker_id):
            # Create a hash of the worker_id
            hash_obj = hashlib.md5(worker_id.encode())
            hash_int = int(hash_obj.hexdigest(), 16)
            
            # Generate a hue value that is more spaced out
            hue = (hash_int % 360)  # Full hue spectrum (0-359 degrees)
            saturation = 0.7  # Keep colors vibrant
            lightness = 0.5   # Ensure colors are not too dark or too bright

            # Convert HSL to RGB (values between 0-1)
            r, g, b = colorsys.hls_to_rgb(hue / 360, lightness, saturation)

            # Scale to 0-255 and format as RGB
            return f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})"
            
        with graph_col:
            nodes = []
            edges = []
            node_levels = {}  # Tracks hierarchy levels
            visited = set()
            
            # Get planned critical path if available
            planned_critical_edges = set()
            try:
                plan_key = f"{MetadataStorage.PLAN_KEY_PREFIX}{dag.master_dag_id}"
                plan_data = metrics_redis.get(plan_key)
                if plan_data:
                    # Ensure we have bytes before attempting to deserialize
                    if isinstance(plan_data, bytes):
                        plan: AbstractDAGPlanner.PlanOutput = cloudpickle.loads(plan_data)
                        if hasattr(plan, 'critical_path_node_ids') and plan.critical_path_node_ids:
                            planned_path = list(plan.critical_path_node_ids)
                            if len(planned_path) > 1:
                                planned_critical_edges = set(zip(planned_path[:-1], planned_path[1:]))
                    else:
                        st.warning(f"Expected bytes from Redis, got {type(plan_data).__name__}")
            except Exception as e:
                st.warning(f"Could not load planned critical path: {e}")
            
            def traverse_dag(node: DAGTaskNode, level=0):
                """ Recursively traverse DAG from root nodes """
                node_id = node.id.get_full_id()
                worker_id = [m for m in task_metrics_data if m["task_id"] == node_id][0]['worker_id']

                if node_id in visited:
                    return # Prevents duplicate processing

                visited.add(node_id)
                node_levels[node_id] = level

                # Create node (no special styling for critical path nodes)
                node_color = get_color_for_worker(worker_id)
                
                nodes.append(Node(
                    id=node_id, 
                    label=node.func_name,
                    title=f"task_id: {node_id}\nworker_id: {worker_id}",
                    size=20,
                    color=node_color,
                    shape="dot",
                    font={"color": "white", "size": 10, "face": "Arial"},
                    level=level
                ))
            
                # Process downstream nodes - highlight actual and planned critical paths
                for downstream in node.downstream_nodes:
                    downstream_id = downstream.id.get_full_id()
                    edge = (node_id, downstream_id)
                    is_actual_critical = edge in critical_edges
                    is_planned_critical = edge in planned_critical_edges
                    
                    # Determine edge color and title
                    if is_actual_critical and is_planned_critical:
                        edge_color = "#FF0000"  # Red - both actual and planned critical
                        edge_title = "Critical Path (Actual & Planned)"
                    elif is_actual_critical:
                        edge_color = "#FF0000"  # Red - only actual critical
                        edge_title = "Critical Path (Actual)"
                    elif is_planned_critical:
                        edge_color = "#FFD700"  # Yellow - only planned critical
                        edge_title = "Critical Path (Planned)"
                    else:
                        edge_color = "#888888"  # Gray - not critical
                        edge_title = ""
                    
                    edges.append(Edge(
                        source=node_id, 
                        target=downstream_id, 
                        arrow="to",
                        color=edge_color,
                        width=2 if (is_actual_critical or is_planned_critical) else 1,
                        title=edge_title
                    ))
                    traverse_dag(downstream, level + 1)
                    
            # Start traversal from all root nodes
            assert dag.root_nodes
            for root in dag.root_nodes:
                traverse_dag(root, level=0)

            # Graph configuration
            config = Config(
                width="70%", # type: ignore
                height=600,
                directed=True,
                physics=False,
                hierarchical=True,
                hierarchical_sort_method="directed"
            )

            # Get selected node from graph interaction
            selected_node = agraph(nodes=nodes, edges=edges, config=config)
            
            # Update selected task if a node was clicked
            if selected_node and selected_node in dag._all_nodes:
                st.session_state.selected_task_id = selected_node
        
        # Get the task node
        metrics: TaskMetrics | None = None
        task_node: DAGTaskNode | None = None
        if hasattr(st.session_state, 'selected_task_id'):
            task_node = dag._all_nodes[st.session_state.selected_task_id]
            # Try to find metrics for this task
            metrics_key = f"{MetadataStorage.TASK_MD_KEY_PREFIX}{st.session_state.selected_task_id}+{dag.master_dag_id}"
            metrics_data = metrics_redis.get(metrics_key)
            if not metrics_data: raise Exception(f"Metrics not found for key {metrics_key}")
            metrics = cloudpickle.loads(metrics_data) # type: ignore

        with details_col:
            st.subheader("Selected Task Details")
            
            # Initialize selected_task_id if not set
            if 'selected_task_id' not in st.session_state:
                st.session_state.selected_task_id = list(dag._all_nodes.keys())[0] if dag._all_nodes else None
            
            def small_metric(label, value, help=None):
                help_icon = f'<span title="{help}" style="cursor: help;">&#9432;</span>' if help else ''
                st.markdown(f"""
                    <div style="padding: 4px 0;">
                        <div style="font-size: 12px; color: gray;">{label} {help_icon}</div>
                        <div style="font-size: 18px; font-weight: bold;">{value}</div>
                    </div>
                """, unsafe_allow_html=True)

            if metrics and task_node:
                # Basic task info
                small_metric("Function", task_node.func_name, help=task_node.id.get_full_id())
                small_metric("Worker", metrics.worker_resource_configuration.worker_id)
                col1, col2, col3 = st.columns(3)
                output_data = metrics.output_metrics.serialized_size_bytes
                worker_startups_w_my_task = [m for m in worker_startup_metrics if task_node.id.get_full_id() in m.initial_task_ids]
                worker_startup_metrics_w_my_task = worker_startups_w_my_task[0] if len(worker_startups_w_my_task) > 0 else None
                worker_startup_time_ms = (worker_startup_metrics_w_my_task.end_time_ms - worker_startup_metrics_w_my_task.start_time_ms) if worker_startup_metrics_w_my_task and worker_startup_metrics_w_my_task.end_time_ms else 0
                with col1:
                    total_task_handling_time = worker_startup_time_ms + (metrics.input_metrics.tp_total_time_waiting_for_inputs_ms or 0) + (metrics.tp_execution_time_ms or 0) + (metrics.update_dependency_counters_time_ms or 0) + (metrics.output_metrics.tp_time_ms or 0) + (metrics.total_invocation_time_ms or 0)
                    small_metric("Worker Resources", f"{metrics.worker_resource_configuration.cpus}, {metrics.worker_resource_configuration.memory_mb}", help="vCPUs, Memory (MB)")
                    small_metric("Time Waiting for Dependencies", f"{(metrics.input_metrics.tp_total_time_waiting_for_inputs_ms or 0):.2f} ms")
                    small_metric("Task Execution Time", f"{(metrics.tp_execution_time_ms or 0):.2f} ms")
                    small_metric("Input Size", format_bytes(sum([input_metric.serialized_size_bytes for input_metric in metrics.input_metrics.input_download_metrics.values()]) + metrics.input_metrics.hardcoded_input_size_bytes))
                with col2:
                    small_metric("Total Task Time", f"{total_task_handling_time:.2f} ms")
                    small_metric("Time Downloading Dependencies", f"{sum([input_metric.time_ms for input_metric in metrics.input_metrics.input_download_metrics.values() if input_metric.time_ms]):.2f} ms")
                    small_metric("Output Upload Time", f"{(metrics.output_metrics.tp_time_ms or 0):.2f} ms")
                    small_metric("Output Size", format_bytes(output_data))
                with col3:
                    small_metric(f"Worker Startup Time ({worker_startup_metrics_w_my_task.state if worker_startup_metrics_w_my_task else 'N/A'})", f"{worker_startup_time_ms:.2f} ms")
                    small_metric("DC Updates Time", f"{(metrics.update_dependency_counters_time_ms or 0):.2f} ms")
                    small_metric("Downstream Invocations Time", f"{(metrics.total_invocation_time_ms or 0):.2f} ms")
                    
                    small_metric("Preload ?:", f"{task_node.try_get_optimization(PreLoadOptimization) is not None}")
                    
        with planned_vs_observed_col:
            # Add planned vs observed metrics if available
            st.subheader("Planned vs Observed Metrics")
            plan_key = f"{MetadataStorage.PLAN_KEY_PREFIX}{dag.master_dag_id}"
            plan_data = metrics_redis.get(plan_key)

            if metrics and task_node:
                try:
                    plan: AbstractDAGPlanner.PlanOutput | None = None
                    tp: AbstractDAGPlanner.PlanningTaskInfo | None = None
                    if plan_data:
                        plan = cloudpickle.loads(plan_data) # type: ignore
                        tp = plan.nodes_info.get(st.session_state.selected_task_id) # type: ignore
                    
                    current_task_metrics = next((t for t in task_metrics_data if t['task_id'] == st.session_state.selected_task_id), None)
                    
                    if current_task_metrics:
                        # Calculate all the metrics we want to compare
                        output_size = metrics.output_metrics.serialized_size_bytes if metrics.output_metrics else 0
                        actual_start_time = current_task_metrics['relative_start_time_ms']
                        end_time_ms = current_task_metrics['end_time_ms']
                        
                        # Create columns for comparison
                        col_metric, col_planned, col_observed, col_diff = st.columns([2, 1, 1, 1])

                        # Add header
                        with col_metric:
                            st.markdown("**Metric**")
                        with col_planned:
                            st.markdown("**Planned**")
                        with col_observed:
                            st.markdown("**Observed**")
                        with col_diff:
                            st.markdown("**Difference**")
                        
                        # Comparison fields
                        with col_metric:
                            st.text('Input Size (bytes)')
                            st.text('Output Size (bytes)')
                            st.text('Downloading Deps. (ms)')
                            st.text('Execution Time (ms)')
                            st.text('Upload Time (ms)')
                            st.text('Earliest Start (ms)')
                            st.text('End Time (ms)')
                            st.text('Worker Startup Time (ms)')
                        with col_planned:
                            if tp:
                                st.text(format_bytes(tp.serialized_input_size))
                                st.text(format_bytes(tp.serialized_output_size))
                                st.text(f"{float(tp.total_download_time_ms):.3f} ms")
                                st.text(f"{float(tp.tp_exec_time_ms):.3f} ms")
                                st.text(f"{float(tp.tp_upload_time_ms):.3f} ms")
                                st.text(f"{float(tp.earliest_start_ms):.3f} ms")
                                st.text(f"{float(tp.task_completion_time_ms):.3f} ms")
                                st.text(f"({tp.worker_startup_state or 'N/A'}) {float(tp.tp_worker_startup_time_ms):.3f} ms")
                        
                        with col_observed:
                            st.text(format_bytes(sum([input_metric.serialized_size_bytes for input_metric in metrics.input_metrics.input_download_metrics.values()]) + metrics.input_metrics.hardcoded_input_size_bytes))
                            st.text(format_bytes(output_size))
                            time_downloading_inputs = sum([input_metric.time_ms for input_metric in metrics.input_metrics.input_download_metrics.values() if input_metric.time_ms])
                            st.text(f"{float(time_downloading_inputs):.3f} ms")
                            st.text(f"{float(metrics.tp_execution_time_ms):.3f} ms")
                            st.text(f"{float(metrics.output_metrics.tp_time_ms or 0):.3f} ms")
                            st.text(f"{float(actual_start_time):.3f} ms")
                            st.text(f"{float(end_time_ms):.3f} ms")
                            worker_startups_w_my_task = [m for m in worker_startup_metrics if task_node.id.get_full_id() in m.initial_task_ids]
                            worker_startup_metrics_w_my_task = worker_startups_w_my_task[0] if len(worker_startups_w_my_task) > 0 else None
                            actual_worker_startup_time_ms = (worker_startup_metrics_w_my_task.end_time_ms - worker_startup_metrics_w_my_task.start_time_ms) if worker_startup_metrics_w_my_task and worker_startup_metrics_w_my_task.end_time_ms else 0
                            st.text(f"({worker_startup_metrics_w_my_task.state if worker_startup_metrics_w_my_task else 'N/A'}) {float(actual_worker_startup_time_ms):.2f} ms")
                        
                        # Calculate and display difference
                        def get_diff_style(percentage):
                            """Returns appropriate color style based on percentage difference"""
                            if percentage is None:
                                return ""
                            if abs(percentage) > 70:
                                return "color: red;"
                            return "color: green;"

                        def format_percentage(diff, total):
                            """Calculate and format percentage difference"""
                            if total == 0:
                                return None, "N/A"
                            percentage = (diff / total) * 100
                            return percentage, f"{percentage:+.2f}%"

                        with col_diff:
                            if tp:
                                # Input Size difference
                                planned_input = tp.serialized_input_size
                                observed_input = sum([input_metric.serialized_size_bytes for input_metric in metrics.input_metrics.input_download_metrics.values()]) + metrics.input_metrics.hardcoded_input_size_bytes
                                pct, pct_str = format_percentage(observed_input - planned_input, planned_input)
                                st.markdown(f"<span style='{get_diff_style(pct)}'>{pct_str}</span>", unsafe_allow_html=True)
                                
                                # Output Size difference
                                planned_output = tp.serialized_output_size
                                pct, pct_str = format_percentage(output_size - planned_output, planned_output)
                                st.markdown(f"<span style='{get_diff_style(pct)}'>{pct_str}</span>", unsafe_allow_html=True)

                                # Time Downloading Dependencies difference
                                planned_download = float(tp.total_download_time_ms)
                                observed_download = float(time_downloading_inputs)
                                if planned_download is not None and observed_download is not None and planned_download != 0:
                                    pct = ((observed_download - planned_download) / planned_download) * 100
                                    st.markdown(f"<span style='{get_diff_style(pct)}'>{pct:+.2f}%</span>", unsafe_allow_html=True)
                                else:
                                    st.text("N/A")
                                
                                # Execution Time difference
                                planned_exec = float(tp.tp_exec_time_ms)
                                observed_exec = float(metrics.tp_execution_time_ms)
                                if planned_exec is not None and observed_exec is not None and planned_exec != 0:
                                    pct = ((observed_exec - planned_exec) / planned_exec) * 100
                                    st.markdown(f"<span style='{get_diff_style(pct)}'>{pct:+.2f}%</span>", unsafe_allow_html=True)
                                else:
                                    st.text("N/A")

                                # Upload Time difference
                                planned_upload = float(tp.tp_upload_time_ms)
                                observed_upload = float(metrics.output_metrics.tp_time_ms or 0)
                                if planned_upload is not None and observed_upload is not None and planned_upload != 0:
                                    pct = ((observed_upload - planned_upload) / planned_upload) * 100
                                    st.markdown(f"<span style='{get_diff_style(pct)}'>{pct:+.2f}%</span>", unsafe_allow_html=True)
                                else:
                                    st.text("N/A")
                                
                                # Earliest Start difference
                                planned_start = float(tp.earliest_start_ms)
                                observed_start = float(actual_start_time)
                                if planned_start is not None and observed_start is not None and planned_start != 0:
                                    pct = ((observed_start - planned_start) / planned_start) * 100
                                    st.markdown(f"<span style='{get_diff_style(pct)}'>{pct:+.2f}%</span>", unsafe_allow_html=True)
                                else:
                                    st.text("N/A")
                                
                                # End Time difference
                                planned_end = float(tp.task_completion_time_ms)
                                observed_end = float(end_time_ms)
                                if planned_end is not None and observed_end is not None and planned_end != 0:
                                    pct = ((observed_end - planned_end) / planned_end) * 100
                                    st.markdown(f"<span style='{get_diff_style(pct)}'>{pct:+.2f}%</span>", unsafe_allow_html=True)
                                else:
                                    st.text("N/A")

                                # Worker Startup Time difference
                                planned_startup = float(tp.tp_worker_startup_time_ms)
                                observed_startup = float(actual_worker_startup_time_ms)
                                if planned_startup is not None and observed_startup is not None and planned_startup != 0:
                                    pct = ((observed_startup - planned_startup) / planned_startup) * 100
                                    st.markdown(f"<span style='{get_diff_style(pct)}'>{pct:+.2f}%</span>", unsafe_allow_html=True)
                                else:
                                    st.text("N/A")
                except Exception as e:
                    st.error(f"Error loading plan data: {str(e)}")
                        
    # Metrics tabs (unchanged from original)
    with tab_summary:
        # Create dataframe for visualizations
        metrics_df = pd.DataFrame(task_metrics_data)
        grouped_df = metrics_df.groupby('function_name').agg({
            'execution_time_ms': ['sum', 'mean', 'count']
        }).reset_index()
        
        # Flatten multi-index columns
        grouped_df.columns = ['_'.join(col).strip('_') for col in grouped_df.columns.values]
        
        # Worker distribution data
        worker_df = metrics_df.groupby(['function_name', 'worker_id']).agg({
            'execution_time_ms': 'sum',
            'task_id': 'count'
        }).reset_index()
        worker_df = worker_df.rename(columns={'task_id': 'task_count'})
        
        # DAG Summary Stats in columns
        # Calculate predicted makespan
        plan_key = f"{MetadataStorage.PLAN_KEY_PREFIX}{dag.master_dag_id}"
        plan_data = metrics_redis.get(plan_key)
        plan_output: AbstractDAGPlanner.PlanOutput | None = None
        if plan_data:
            plan_output = cloudpickle.loads(plan_data) # type: ignore

        predicted_makespan = plan_output.nodes_info[dag.sink_node.id.get_full_id()].task_completion_time_ms if plan_output else -1
        predicted_upload_time = sum([tp.tp_upload_time_ms for tp in plan_output.nodes_info.values()]) if plan_output else -1
        predicted_upload_size = sum([tp.serialized_output_size for tp in plan_output.nodes_info.values()]) if plan_output else -1
        predicted_download_time = sum([tp.total_download_time_ms for tp in plan_output.nodes_info.values()]) if plan_output else -1
        predicted_download_size = sum([tp.serialized_input_size for tp in plan_output.nodes_info.values()]) if plan_output else -1

        col1, col2, col3, col4, col5 = st.columns(5)
        worker_startup_metrics_for_this_workflow = [m for m in worker_startup_metrics if m.master_dag_id == dag.master_dag_id]
        worker_startup_times_for_this_workflow = [m.end_time_ms - m.start_time_ms for m in worker_startup_metrics_for_this_workflow if m.end_time_ms is not None]
        total_workflow_worker_startup_time_s = sum(worker_startup_times_for_this_workflow) / 1000
        warm_starts_count = len([m for m in worker_startup_metrics_for_this_workflow if m.state == "warm"])
        cold_starts_count = len([m for m in worker_startup_metrics_for_this_workflow if m.state == "cold"])
        task_execution_time_avg = total_time_executing_tasks_ms / len(dag_metrics) if dag_metrics else 0
        avg_dag_download_time = sum(m['dag_download_time'] for m in dag_prepare_metrics) / len(dag_prepare_metrics)
        total_dag_download_time = sum(m['dag_download_time'] for m in dag_prepare_metrics)
        avg_subdag_create_time = sum(m['create_subdag_time'] for m in dag_prepare_metrics) / len(dag_prepare_metrics)
        avg_dag_size = sum(m['dag_size'] for m in dag_prepare_metrics) / len(dag_prepare_metrics)
        with col1:
            st.metric("Total Tasks", len(dag._all_nodes))
            st.metric(f"Total Time Executing Tasks (avg: {task_execution_time_avg:.2f} ms)", f"{total_time_executing_tasks_ms:.2f} ms")
            if predicted_upload_size > 0:
                percentage_diff = ((total_data_transferred - predicted_upload_size) / predicted_upload_size) * 100
                st.metric(
                    "Total Data Transferred", 
                    format_bytes(total_data_transferred),
                    delta=f"{percentage_diff:+.1f}% vs predicted ({format_bytes(predicted_upload_size)})"
                )
            else:
                st.metric("Total Data Transferred", format_bytes(total_data_transferred))
            st.metric("Avg. DAG Download Time", f"{avg_dag_download_time:.2f} ms")
        with col2:
            if predicted_makespan > 0:
                percentage_diff = ((makespan_ms - predicted_makespan) / predicted_makespan) * 100
                st.metric(
                    "Makespan", 
                    f"{makespan_ms:.2f} ms",
                    delta=f"{percentage_diff:+.1f}% vs predicted ({predicted_makespan:.2f} ms)"
                )
            else:
                st.metric("Makespan", f"{makespan_ms:.2f} ms")
            if predicted_upload_time > 0:
                percentage_diff = ((total_time_uploading_data_ms - predicted_upload_time) / predicted_upload_time) * 100
                st.metric(
                    "Total Upload Time", 
                    f"{total_time_uploading_data_ms:.2f} ms",
                    delta=f"{percentage_diff:+.1f}% vs predicted ({predicted_upload_time:.2f} ms)"
                )
            else:
                st.metric("Total Upload Time", f"{total_time_uploading_data_ms:.2f} ms")
            avg_data = total_data_transferred / len(dag_metrics) if dag_metrics else 0
            st.metric("Data Transferred per Task (avg)", format_bytes(avg_data))
            st.metric("DAG Size", format_bytes(avg_dag_size))
        with col3:
            st.metric("Unique Workers", int(metrics_df['worker_id'].nunique()))
            if predicted_download_time > 0:
                percentage_diff = ((total_time_downloading_data_ms - predicted_download_time) / predicted_download_time) * 100
                st.metric(
                    "Total Download Time", 
                    f"{total_time_downloading_data_ms:.2f} ms",
                    delta=f"{percentage_diff:+.1f}% vs predicted ({predicted_download_time:.2f} ms)"
                )
            else:
                st.metric("Total Download Time", f"{total_time_downloading_data_ms:.2f} ms")
            st.metric("Total Worker Invocations (excludes initial)", f"{int(metrics_df['downstream_calls'].sum())}")
            st.metric("Total Time Downloading DAG", f"{total_time_downloading_dag_ms:.2f} ms")
        with col4:
            st.metric("Unique Tasks", len(function_groups))
            st.metric("Total Invocation Time", f"{total_time_invoking_tasks_ms:.2f} ms")
            st.metric(" ", " ", help="")
            st.metric(" ", " ", help="")
            st.metric("Avg. SubDAG Create Time", f"{avg_subdag_create_time:.2f} ms")
        with col5:
            st.metric("Total DC Update Time", f"{total_time_updating_dependency_counters_ms:.2f} ms")
            if plan_output and plan_output.total_time_waiting_for_worker_startup_ms > 0:
                percentage_diff = ((total_workflow_worker_startup_time_s - (plan_output.total_time_waiting_for_worker_startup_ms / 1000)) / (plan_output.total_time_waiting_for_worker_startup_ms / 1000)) * 100
                st.metric(
                    "Total Worker Startup Time", 
                    f"{total_workflow_worker_startup_time_s:.2f} s ({len(worker_startup_metrics_for_this_workflow)} workers)",
                    delta=f"{percentage_diff:+.1f}% vs predicted ({plan_output.total_time_waiting_for_worker_startup_ms / 1000:.2f} s)",
                    help=f"warm: {warm_starts_count}, cold: {cold_starts_count}"
                )
            else:
                st.metric("Total Worker Startup Time", f"{total_workflow_worker_startup_time_s:.2f} s ({len(worker_startup_metrics_for_this_workflow)} workers)", help=f"warm: {warm_starts_count}, cold: {cold_starts_count}")
            st.metric(" ", " ", help="")

        breakdown_data = {
            "Task Execution": total_time_executing_tasks_ms,
            "Data Download": total_time_downloading_data_ms,
            "Data Upload": total_time_uploading_data_ms,
            "Invocation Time": total_time_invoking_tasks_ms,
            "Waiting for Worker Startup": total_workflow_worker_startup_time_s * 1_000,
            "DC Updates": total_time_updating_dependency_counters_ms,
            "DAG Download Time": total_dag_download_time
        }
        
        # Create pie chart
        breakdown_df = pd.DataFrame({
            "Component": breakdown_data.keys(),
            "Time (ms)": breakdown_data.values()
        })
       
        st.subheader("Breakdown")
        fig = px.pie(
            breakdown_df,
            names="Component",
            values="Time (ms)",
            title="",
            color="Component",
            color_discrete_map={
                "Task Execution": "#636EFA",
                "Data Download": "#EF553B",
                "Data Upload": "#00CC96",
                "Invocation Time": "#AB63FA",
                "Waiting for Worker Startup": "#AB63FA",
                "DC Updates": "#DB61CE",
                "Unknown": "#333333"
            }
        )
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate="%{label}:<br>%{value:.2f} ms<br>%{percent}"
        )
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Metrics by Function/Task type")
        st.dataframe(grouped_df, use_container_width=True)

        st.subheader("Raw Task Metrics")
        raw_task_df = metrics_df.sort_values('function_name').reset_index(drop=True)
        columns_to_exclude = { "function_name" }
        columns_to_show = [col for col in raw_task_df.columns if col not in columns_to_exclude]
        st.dataframe(
            raw_task_df[columns_to_show],
            use_container_width=True,
            column_config={
                "task_id": st.column_config.TextColumn("Task ID"),
                "task_started_at": st.column_config.TextColumn("Started At"),
                "execution_time_ms": st.column_config.NumberColumn("Exec Time (ms)", format="%.2f"),
                "worker_id": st.column_config.TextColumn("Worker"),
                "worker_resource_configuration_cpus": st.column_config.TextColumn("Worker Resources (CPUs)"),
                "worker_resource_configuration_ram": st.column_config.TextColumn("Worker Resources (RAM MBs)"),
                "input_size": st.column_config.NumberColumn("Input Size"),
                "output_size": st.column_config.NumberColumn("Output Size"),
                "downstream_calls": st.column_config.NumberColumn("Downstream Calls")
            }
        )
    
        with tab_exec:
            # Create two columns for the charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Total Execution Time per Function Group
                total_time_df = metrics_df.groupby('function_name')['execution_time_ms'].sum().reset_index()
                fig = px.bar(
                    total_time_df,
                    x='function_name',
                    y='execution_time_ms',
                    labels={
                        'function_name': 'Function Group',
                        'execution_time_ms': 'Total Execution Time (ms)'
                    },
                    title="Total Execution Time by Function",
                    color='function_name',
                    text_auto=True
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Average Execution Time per Function Group
                avg_time_df = metrics_df.groupby('function_name')['execution_time_ms'].mean().reset_index()
                fig = px.bar(
                    avg_time_df,
                    x='function_name',
                    y='execution_time_ms',
                    labels={
                        'function_name': 'Function Group',
                        'execution_time_ms': 'Average Execution Time (ms)'
                    },
                    title="Average Execution Time by Function",
                    color='function_name',
                    text_auto=True
                )
                fig.update_layout(showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
    with tab_data_transfer:
        if dag_metrics:
            # Collect all individual transfer metrics
            download_throughputs_mb_s = []
            upload_throughputs_mb_s = []
            all_transfer_speeds_b_ms = []  # In bytes/ms
            total_data_downloaded = 0
            total_data_uploaded = 0

            for task_metrics in dag_metrics:
                for input_metrics in task_metrics.input_metrics.input_download_metrics.values():
                    # Calculate download throughputs for each input
                    if input_metrics.time_ms is not None:
                        downloadable_input_size_bytes = sum([input_metric.serialized_size_bytes for input_metric in task_metrics.input_metrics.input_download_metrics.values()])
                        throughput_mb = (downloadable_input_size_bytes / (input_metrics.time_ms / 1000)) / (1024 * 1024)  # MB/s
                        speed_bytes_ms = downloadable_input_size_bytes / input_metrics.time_ms  # bytes/ms
                        download_throughputs_mb_s.append(throughput_mb)
                        all_transfer_speeds_b_ms.append(speed_bytes_ms)
                        total_data_downloaded += downloadable_input_size_bytes

                # Calculate upload throughput for output if available
                if task_metrics.output_metrics.tp_time_ms is not None:
                    throughput_mb = (task_metrics.output_metrics.serialized_size_bytes / (task_metrics.output_metrics.tp_time_ms / 1000)) / (1024 * 1024)  # MB/s
                    speed_bytes_ms = task_metrics.output_metrics.serialized_size_bytes / task_metrics.output_metrics.tp_time_ms  # bytes/ms
                    upload_throughputs_mb_s.append(throughput_mb)
                    all_transfer_speeds_b_ms.append(speed_bytes_ms)
                    total_data_uploaded += task_metrics.output_metrics.serialized_size_bytes

            # Calculate average throughputs
            avg_download_throughput_mb_s = sum(download_throughputs_mb_s) / len(download_throughputs_mb_s) if download_throughputs_mb_s else 0
            avg_upload_throughput_mb_s = sum(upload_throughputs_mb_s) / len(upload_throughputs_mb_s) if upload_throughputs_mb_s else 0

            # Display metrics in columns
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                if predicted_download_size > 0:
                    percentage_diff = ((total_data_downloaded - predicted_download_size) / predicted_download_size) * 100
                    st.metric(
                        "Total Data Downloaded",
                        format_bytes(total_data_downloaded),
                        delta=f"{percentage_diff:+.1f}% vs predicted ({format_bytes(predicted_download_size)})"
                    )
                else:
                    st.metric("Total Data Downloaded", format_bytes(total_data_downloaded))
                if predicted_upload_size > 0:
                    percentage_diff = ((total_data_uploaded - predicted_upload_size) / predicted_upload_size) * 100
                    st.metric(
                        "Total Data Uploaded",
                        format_bytes(total_data_uploaded),
                        delta=f"{percentage_diff:+.1f}% vs predicted ({format_bytes(predicted_upload_size)})"
                    )
                else:
                    st.metric("Total Data Uploaded", format_bytes(total_data_uploaded))
            with col2:
                st.metric("Download Throughput (avg)", f"{avg_download_throughput_mb_s:.2f} MB/s")
                st.metric("Upload Throughput (avg)", f"{avg_upload_throughput_mb_s:.2f} MB/s")
            with col3:
                st.metric("Number of Downloads", len(download_throughputs_mb_s))
                st.metric("Number of Uploads", f"{len(upload_throughputs_mb_s)}")
            with col4:
                st.metric("Total Download Time", f"{total_time_downloading_data_ms:.2f} ms")
                st.metric("Total Upload Time", f"{total_time_uploading_data_ms:.2f} ms")

    with tab_workers:
        if dag_metrics:
            col1, col2 = st.columns(2)
            with col1:
                fig = px.bar(
                    worker_df,
                    x='function_name',
                    y='task_count',
                    color='worker_id',
                    title="Task Distribution by Worker and Function Group",
                    labels={
                        'function_name': 'Function Group',
                        'task_count': 'Number of Tasks',
                        'worker_id': 'Worker ID'
                    }
                )
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = px.sunburst(
                    worker_df,
                    path=['worker_id', 'function_name'],
                    values='task_count',
                    title="Worker-Function Group Task Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)

    with tab_critical_path:
        st.header("Critical Path Analysis")

        if not critical_nodes:
            st.warning("Could not determine critical path")
        else:
            st.subheader(f"Critical Path ({len(critical_nodes)} tasks)")
            
            # Initialize accumulated time counters
            total_waiting = 0
            total_executing = 0
            total_uploading = 0
            total_invoking = 0
            total_updating = 0
            total_worker_startup = 0

            # Calculate total times across all critical path tasks
            for task_id in critical_nodes:
                metrics_key = f"{MetadataStorage.TASK_MD_KEY_PREFIX}{task_id}+{dag.master_dag_id}"
                metrics_data = metrics_redis.get(metrics_key)
                if not metrics_data:
                    continue
                    
                metrics: TaskMetrics = cloudpickle.loads(metrics_data)  # type: ignore

                for wsm in worker_startup_metrics:
                    if task_id in wsm.initial_task_ids:
                        total_worker_startup += (wsm.end_time_ms - wsm.start_time_ms) if wsm.end_time_ms is not None else 0
                
                # Accumulate times
                total_waiting += metrics.input_metrics.tp_total_time_waiting_for_inputs_ms or 0
                total_executing += metrics.tp_execution_time_ms or 0
                total_uploading += metrics.output_metrics.tp_time_ms if metrics.output_metrics and metrics.output_metrics.tp_time_ms else 0
                total_invoking += metrics.total_invocation_time_ms or 0
                total_updating += metrics.update_dependency_counters_time_ms or 0   
            
            # Create a summary table
            breakdown_data = [
                ('Waiting for Inputs', total_waiting, 'Time spent waiting for input data to be available'),
                ('Executing', total_executing, 'Time spent executing the task function'),
                ('Uploading Outputs', total_uploading, 'Time spent uploading task outputs to storage'),
                ('Invoking Downstream', total_invoking, 'Time spent notifying and invoking downstream tasks'),
                ('Updating Counters', total_updating, 'Time spent updating dependency counters'),
                ('Worker Startup', total_worker_startup, 'Time spent initializing worker containers')
            ]
            
            # Calculate total and percentages
            total_time = sum(time for _, time, _ in breakdown_data)
            
            # Create a DataFrame for the table
            table_data = []
            for activity, time, description in breakdown_data:
                table_data.append({
                    'Activity': activity,
                    'Time (ms)': f"{time:,.0f}",
                    'Percentage': f"{(time/total_time*100):.1f}%" if total_time > 0 else "0.0%",
                    'Description': description
                })
            
            # Add total row
            if table_data:
                table_data.append({
                    'Activity': 'TOTAL',
                    'Time (ms)': f"{total_time:,.0f} (makespan: {makespan_ms:,.0f})",
                    'Percentage': '100.0%',
                    'Description': 'Sum of all measured components'
                })
            
            # Display the table
            if table_data:
                # Separate the total row if it exists
                total_row = None
                if table_data and table_data[-1]['Activity'] == 'TOTAL':
                    total_row = table_data.pop()
                
                # Create DataFrame and sort by percentage (descending)
                df = pd.DataFrame(table_data)
                df['percentage_value'] = df['Percentage'].str.rstrip('%').astype(float)
                df = df.sort_values('percentage_value', ascending=False).drop('percentage_value', axis=1)
                
                # Add total row back if it existed
                if total_row:
                    df = pd.concat([df, pd.DataFrame([total_row])], ignore_index=True)
                
                st.dataframe(
                    df,
                    use_container_width=True,
                    column_config={
                        'Activity': st.column_config.TextColumn(width='medium'),
                        'Time (ms)': st.column_config.TextColumn(width='small'),
                        'Percentage': st.column_config.TextColumn(width='small'),
                        'Description': st.column_config.TextColumn(width='large')
                    },
                    hide_index=True
                )
    
if __name__ == "__main__":
    main()