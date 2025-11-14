# DEPLOYMENT

## Requirements
- Docker
- Python 3.12

## Steps
1) Create **python venv**
    - `python3.12 -m venv venv`
    - use `. activate_venv.sh` to activate it
    - install requirements (first **activate venv**): `pip install -r src/requirements.txt`
2) Install `graphviz`
```bash
sudo apt-get update
sudo apt-get install graphviz
```
3) Install `redis-cli`
```bash
sudo apt-get install redis-tools
```
4) Enable Docker API (used to calculate resource usage metrics): [https://docs.docker.com/engine/daemon/remote-access/](https://docs.docker.com/engine/daemon/remote-access/)
5) Run `create_redis_docker.sh` (creates 2 redis containers, password protected and with persistence enabled)
6) Start the docker gateway (FaaS "emulator"): `bash build_docker_worker_image.sh && bash start_gateway_docker.sh`