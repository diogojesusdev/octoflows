 Notes:
    - warmup requests could warm a container on a diff. gateway than the task will run

- Test workflows with: My vagrant (gateway) + 126 (gateway + DBs)
- Deploy on 1 cluster machine using Vagrant
- Try running tree reduction multiple times and with diff. planner algorithms

- Test with 2 remote machines
- Run ALL experiments 3 times on 2 remote machines

[KNOWN_ISSUES]
- In simulation, {worker_active_periods} (`abstract_dag_planner.py`) are not being calculated accuratelly