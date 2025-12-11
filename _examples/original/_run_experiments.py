import os
import sys
import time
import subprocess
import requests

WORKFLOWS_PATHS = [
    'tree_reduction.py',
    'gemm.py',
    'text_analysis.py',
    'image_transformer.py',
]

# ITERATIONS_PER_ALGORITHM = 10
ITERATIONS_PER_ALGORITHM = 2
# ALGORITHMS = ['uniform', 'non-uniform']
ALGORITHMS = ['wukong', 'wukong-opt', 'uniform', 'uniform-opt', 'non-uniform', 'non-uniform-opt']
SLAS = ['50']
# SLAS = ['50', '75', '90']

DOCKER_FAAS_GATEWAY_IPS = [
    "95.94.148.210",
    "146.193.41.126"
]

failed_instances = 0

def wait_containers_shutdown():
    for gateway_ip in DOCKER_FAAS_GATEWAY_IPS:
        url = f"http://{gateway_ip}:5000/wait-containers-shutdown"
        print("Waiting for all containers to shutdown...")
        try:
            response = requests.post(url)
            if response.status_code == 200:
                print("All containers have shutdown!")
            else:
                print(f"Unexpected response: {response.status_code}, {response.text}")
        except requests.RequestException as e:
            print(f"Error making request: {e}")


def kill_docker_workers():
    """Forcefully kill all running containers that use the 'docker_worker' image."""
    print("Killing all containers using image 'docker_worker'...")
    try:
        # Get all container IDs using the image
        result = subprocess.run(
            ["docker", "ps", "-q", "--filter", "ancestor=docker_worker"],
            stdout=subprocess.PIPE,
            text=True
        )
        container_ids = result.stdout.strip().splitlines()

        if not container_ids:
            print("No running containers found for image 'docker_worker'.")
            return

        # Kill each container
        subprocess.run(["docker", "kill"] + container_ids, check=False)
        print(f"Killed {len(container_ids)} container(s) using image 'docker_worker'.")
    except Exception as e:
        print(f"Error while killing docker_worker containers: {e}")


def run_experiment(script_path: str, algorithm: str, sla: str, iteration: str, current: int, total: int, max_retries: int = 2) -> None:
    global failed_instances
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    full_script_path = os.path.join(script_dir, script_path)

    cmd = [sys.executable, full_script_path, algorithm, sla]

    percentage = (current / total) * 100 if total > 0 else 0
    
    for attempt in range(1, max_retries + 1):
        wait_containers_shutdown()  # sync before each attempt
        
        retry_suffix = f" (retry {attempt}/{max_retries})" if attempt > 1 else ""
        print(f" > [{percentage:5.1f}%] [failed_instances:{failed_instances}] Workflow: {os.path.basename(script_path)} | Planner: {algorithm.upper()} algorithm | SLA: {sla} (iteration: {iteration}){retry_suffix} [{current}/{total}]")

        try:
            subprocess.run(cmd, check=True, cwd=script_dir, timeout=400)
            # Success! Exit the retry loop
            return
            
        except subprocess.TimeoutExpired:
            print(f"Timeout: {script_path} with {algorithm} and SLA {sla} exceeded 6.5 minutes (attempt {attempt}/{max_retries})", file=sys.stderr)
            kill_docker_workers()
            
            if attempt == max_retries:
                # Final attempt failed
                print(f"Failed after {max_retries} attempts. Skipping...", file=sys.stderr)
                failed_instances += 1
            else:
                print(f"Retrying in 3 seconds...", file=sys.stderr)
                time.sleep(3)
                
        except subprocess.CalledProcessError as e:
            print(f"Error running {script_path} with {algorithm} and SLA {sla}: {e} (attempt {attempt}/{max_retries})", file=sys.stderr)
            kill_docker_workers()
            
            if attempt == max_retries:
                # Final attempt failed
                print(f"Failed after {max_retries} attempts. Skipping...", file=sys.stderr)
                failed_instances += 1
            else:
                print(f"Retrying in 3 seconds...", file=sys.stderr)
                time.sleep(3)

def main():
    global failed_instances
    start_time = time.time()
    os.environ['TZ'] = 'UTC-1'  # Set timezone for log timestamps consistency

    script_dir = os.path.dirname(os.path.abspath(__file__))

    total_runs = 0
    for script_name in WORKFLOWS_PATHS:
        script_path = os.path.join(script_dir, script_name)
        if os.path.isfile(script_path):
            total_runs += len(ALGORITHMS) * len(SLAS) * ITERATIONS_PER_ALGORITHM

    current_run = 0

    for script_name in WORKFLOWS_PATHS:
        script_path = os.path.join(script_dir, script_name)
        if not os.path.isfile(script_path):
            print(f"Warning: The file \"{script_path}\" does not exist. Skipping...", file=sys.stderr)
            continue

        print(f"\n{'='*60}")
        print(f"Running experiments for {script_name}")
        print(f"{'='*60}")

        for algorithm in ALGORITHMS:
            for sla in SLAS:
                for i in range(1, ITERATIONS_PER_ALGORITHM + 1):
                    current_run += 1
                    run_experiment(script_name, algorithm, sla, str(i), current_run, total_runs)

    total_time = time.time() - start_time
    print(f"All runs completed in {total_time/60:.2f}mins | Failed instances: {failed_instances}")


if __name__ == "__main__":
    main()
