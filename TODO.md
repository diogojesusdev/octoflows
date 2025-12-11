- Capability to use multiple machines as workers
    - Implementation
        - user provides list of available Docker gateway addresses
        - when client delegates INITIAL tasks, it randomly chooses one of the gateways
            note: random is better than round-robin
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

- Run ALL experiments 2 times on 1 local + 1 remote machine
- Check metrics dashboard for each type of workflow
- Deploy on another machine
- Test with 2 remote machines
- Run ALL experiments 3 times on 2 remote machines

[KNOWN_ISSUES]
- In simulation {worker_active_periods} (`abstract_dag_planner.py`) are not being calculated accuratelly