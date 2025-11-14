# OctoFlows: A Decentralized Serverless Workflow Execution Engine with Predictive Planning

Research project for the MSc thesis, "Serverless Dataflows: A Decentralized Workflow Execution Engine with Predictive Planning"

This project tackles the core inefficiencies of running complex, multi-stage data workflows (DAGs) on Function-as-a-Service (FaaS) platforms. Traditional serverless execution suffers from high overhead due to the stateless nature of functions and a heavy reliance on external storage for communication and data exchange.

Our solution is a modular **decentralized workflow execution engine** that introduces a **predictive planning** layer. Instead of making scheduling decisions one step at a time, it leverages historical metadata from previous runs to generate a complete execution plan *before* the workflow starts.

## Key Features

* **Predictive Static Planning:** The engine uses historical metrics (like execution time, I/O size, and startup latency) to simulate and create an optimal static plan for the entire workflow, using a user-specified planning algorithm (*Planner*). This plan dictates which tasks run on which worker, allowing Planners control over *data locality*.

* **Decentralized, Choreographed Execution:** There is **no central scheduler**. Once the plan is created on the client, FaaS workers execute it in a choreographed manner, intelligently delegating tasks and launching new workers as needed. This eliminates central bottlenecks and reduces coordination overhead.

* **Non-Uniform Resource Allocation:** Our solution can assign different memory resource configurations (CPUs proportionally calculated) to different tasks within the same workflow. Most comparable systems use "one-size-fits-all" homogeneous configurations.

* **Advanced Optimizations:** The planning layer can apply optimizations to minimize latency:
    * **Pre-warming:** Proactively warms workers *before* they are needed, hiding cold-start latencies.
    * **Pre-loading:** Proactively downloads task dependencies in the background as soon as they become available (rather than when they are needed), masking data transfer latency.

* **Performance & Efficiency:** Our evaluation shows the effectiveness of our solution, showing gains on both performance and resource usage when compared to a similar state-of-the-art solution (WUKONG).
    * **Resource-Efficient:** Our `Uniform` planner (our most resource-efficient) reduces resource consumption by **36%** and execution time by **12.6%** compared to WUKONG.
    * **Speed-Optimized:** Our Optimized `Non-Uniform` planner (our fastest) achieves a **57.5%** reduction in makespan compared to our most resource-efficient planner (Uniform), but this performance gain requires a **113.5%** increase in resource consumption.