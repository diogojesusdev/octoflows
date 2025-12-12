import base64
import os
import signal
import sys
import threading
import time
import uuid
import cloudpickle
from flask import Flask, request, jsonify
from concurrent.futures import ThreadPoolExecutor

from src.task_worker_resource_configuration import TaskWorkerResourceConfiguration
from src.utils.logger import create_logger
import src.docker_workers_gateway.container_pool_executor as container_pool_executor

logger = create_logger(__name__)

DOCKER_WORKER_PYTHON_PATH = "/app/src/docker_worker_handler/worker.py"

MAX_CONCURRENT_WORKERS = 26
DOCKER_IMAGE = os.environ.get('DOCKER_IMAGE', None)
if DOCKER_IMAGE is None:
    logger.warning("Set the DOCKER_IMAGE environment variable to the name of the Docker image to use.")
    sys.exit(3)

DOCKER_IMAGE = DOCKER_IMAGE.strip()
logger.info(f"Using Docker image: '{DOCKER_IMAGE}'")

app = Flask(__name__)
thread_pool = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_WORKERS)
container_pool = container_pool_executor.ContainerPoolExecutor(docker_image=DOCKER_IMAGE, max_containers=MAX_CONCURRENT_WORKERS)

def process_job_async(resource_configuration: TaskWorkerResourceConfiguration, base64_config: str, dag_id: str, base64_task_ids: list[str], base64_fulldag: str | None = None, base64_relevant_cached_results: str | None = None):
    """
    Process a job asynchronously.
    This function will be run in a separate thread.
    """
    job_id = str(uuid.uuid4())
    worker_id = resource_configuration.worker_id

    def get_time_formatted():
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    if base64_fulldag is not None:
        command = f"python {DOCKER_WORKER_PYTHON_PATH} {base64_config} {dag_id} {base64_task_ids} {base64_relevant_cached_results} {base64_fulldag}"
    else:
        command = f"python {DOCKER_WORKER_PYTHON_PATH} {base64_config} {dag_id} {base64_task_ids} {base64_relevant_cached_results}"
        
    logger.info(f"[{get_time_formatted()}] {job_id}) [INFO] Waiting for container for W({worker_id})")

    container_id = container_pool.wait_for_container(cpus=resource_configuration.cpus, memory=resource_configuration.memory_mb, dag_id=dag_id)
    try:
        exit_code = container_pool.execute_command_in_container(container_id, command)
        if exit_code == 2:
            logger.error(f"[{get_time_formatted()}] {job_id}) W({worker_id}) [ERROR] Container {container_id} should be available but another task is using it (this should never happen!)")
        elif exit_code != 0:
            logger.error(f"[{get_time_formatted()}] {job_id}) W({worker_id}) [ERROR] Container {container_id} unexpected exit code={exit_code}")
    except Exception as e:
        logger.error(f"[{get_time_formatted()}] {job_id}) W({worker_id}) [ERROR] Exception: {e}")
    finally:
        container_pool.release_container(container_id)

@app.route('/warmup', methods=['POST'])
def handle_warmup():
    # Parse request data
    if not request.is_json: return jsonify({"error": "JSON data is required"}), 400
    data = request.get_json()

    dag_id = data.get('dag_id', None)
    if dag_id is None: 
        logger.error("'dag_id' field is required")
        return jsonify({"error": "'dag_id' field is required"}), 400

    resource_config_key = data.get('resource_configurations', None)
    if resource_config_key is None: 
        logger.error("'resource_configurations' field is required")
        return jsonify({"error": "'resource_configurations' field is required"}), 400

    resource_configurations: list[TaskWorkerResourceConfiguration] = cloudpickle.loads(base64.b64decode(resource_config_key))

    for resource_configuration in resource_configurations:
        print("Warming up resource configuration: ", resource_configuration)
        container_id = container_pool._launch_container(cpus=resource_configuration.cpus, memory=resource_configuration.memory_mb, dag_id=dag_id, name_prefix="PRE-WARMED_", is_prewarm=True)
        
        if container_id is None:
            logger.error("Max containers reached. Can't launch new container")
            return jsonify({"error": "Max containers reached. Can't launch new container"}), 400

        container_pool.release_container(container_id)
    return "", 202

@app.route('/wait-containers-shutdown', methods=['POST'])
def handle_containers_shutdown():
    # Parse request data
    logger.info("Waiting for all containers to shutdown")
    container_pool._wait_until_there_are_no_more_containers_active()
    logger.info("All containers have shutdown!")
    return "", 200

@app.route('/job', methods=['POST'])
def handle_job():
    """
    Handles POST and GET requests to /job.
    - POST: Accepts the job and immediately returns 202, then processes the job asynchronously.
    - GET: Returns a list of available container IDs grouped by resource configuration. Used for DEBUG
    """
    # Parse request data
    if not request.is_json: return jsonify({"error": "JSON data is required"}), 400
    data = request.get_json()

    resource_config_key = data.get('resource_configuration', None)
    if resource_config_key is None: 
        logger.error("'resource_configuration' field is required")
        return jsonify({"error": "'resource_configuration' field is required"}), 400
    resource_configuration: TaskWorkerResourceConfiguration | None = cloudpickle.loads(base64.b64decode(resource_config_key))
    if resource_configuration is None: 
        logger.error("'resource_configuration' field is required")
        return jsonify({"error": "'resource_configuration' field is required"}), 400
    dag_id = data.get('dag_id', None)
    if dag_id is None: 
        logger.error("'dag_id' field is required")
        return jsonify({"error": "'dag_id' field is required"}), 400
    b64_task_ids = data.get('task_ids', None)
    if b64_task_ids is None: 
        logger.error("'task_id' field is required")
        return jsonify({"error": "'task_id' field is required"}), 400
    b64config = data.get('config', None)
    if b64config is None: 
        logger.error("'config' field is required")
        return jsonify({"error": "'config' field is required"}), 400
    b64_fulldag = data.get('fulldag', None)
    b64_relevant_cached_results = data.get('relevant_cached_results', None)
    if b64_relevant_cached_results is None:
        logger.error("'relevant_cached_results' field is required")
        return jsonify({"error": "'relevant_cached_results' field is required"}), 400

    thread_pool.submit(process_job_async, resource_configuration, b64config, dag_id, b64_task_ids, b64_fulldag, b64_relevant_cached_results)
    
    return "", 202 # Immediately return 202 Accepted

if __name__ == '__main__':
    is_shutting_down_flag = threading.Event()

    def cleanup(signum, frame):
        if is_shutting_down_flag.is_set(): return # avoid executing shutdown more than once
        is_shutting_down_flag.set()
        logger.info("Shutdown. Cleaning up...")
        container_pool.shutdown()
        thread_pool.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)  # Ctrl+C
    signal.signal(signal.SIGTERM, cleanup)  # Termination signal

    app.run(host='0.0.0.0', port=5000)
