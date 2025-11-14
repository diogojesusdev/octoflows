# DEPLOYMENT

## Requirements
- Docker
- Python 3.12

## Steps
1) Create **python venv**
    - `python3.12 -m venv venv`
    - Activate it with `. activate_venv.sh`
    - After activating the virtual environment, install the requirements: `pip install -r src/requirements.txt`
2) Install `graphviz`
```bash
sudo apt-get update
sudo apt-get install graphviz
```
3) Install `redis-cli`
```bash
sudo apt-get install redis-tools
```
4) Enable the Docker API, which is used to calculate resource usage metrics. For instructions, see the [official Docker documentation](https://docs.docker.com/engine/daemon/remote-access/).
5) Run `create_redis_docker.sh`. This script creates two password-protected Redis containers with persistence enabled.
6) Start the Docker gateway (FaaS emulator): `bash build_docker_worker_image.sh && bash start_gateway_docker.sh`