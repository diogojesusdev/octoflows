- Capability to use multiple machines as workers
    - Implementation
        - user provides list of available Docker gateway addresses
        - when user delegates tasks it randomly chooses one of the gateways
        - workers will always send requests to the gateway running on their
    - Setup
        - Windows machine
            1 gateway + Client program
            Expose ports: 5000 + 2375
            Issue: here, local workers can't talk to own gateway via remote ip because router doesn't allow loopback?
                solution: windows hosts always talk to self
        - Remote Ubuntu machine
            1 gateway + 2 Databases
            Expose ports: 5000 + 2375 + 6379 + 6380
                Use SSH tunnels?
    - Notes:
        - `DockerContainerUsageMonitor` class would need to receive a list of Docker API endpoints
        - warmup requests could warm a container on a diff. gateway than the task will run

[KNOWN_ISSUES]
- In simulation {worker_active_periods} (`abstract_dag_planner.py`) are not being calculated accuratelly