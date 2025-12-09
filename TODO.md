- ERROR: Cannot connect to host 95.94.148.210:5000
    Docker worker can't talk to gateway on same machine via public IP?
- Make the dashboard scripts merge data from multiple IPs/DBs
- Capability to use multiple machines as workers
    - Implementation
        - user provides list of available Docker gateway addresses
        - when client delegates INITIAL tasks, it randomly chooses one of the gateways
        - workers will send requests to random gateway
    - Setup
        - Windows machine
            1 gateway + Client program
            Expose ports: 5000 + 2375
            Issue: here, local workers can't talk to own gateway via remote ip because router doesn't allow loopback?
                solution: windows hosts always talk to self
        - Remote Ubuntu machine
            1 gateway + 2 Databases
            Expose ports: 5000 + 6379 + 6380
    - Notes:
        - `DockerContainerUsageMonitor` class would need to receive a list of Docker API endpoints
        - warmup requests could warm a container on a diff. gateway than the task will run

[KNOWN_ISSUES]
- In simulation {worker_active_periods} (`abstract_dag_planner.py`) are not being calculated accuratelly