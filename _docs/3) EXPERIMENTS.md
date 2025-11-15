# EXPERIMENTS

After configuring the environment as described in `DEPLOYMENT.md`, the set of experiments used in the paper can be run using the Python script `_examples/original/_run_experiments.py`. This script allows you to configure the workflows to run, the SLAs to use, the Planners, and the number of instances for each possible combination. The script then runs the experiments sequentially. 

> Note: Before starting a new experiment, the script waits for all containers to shut down gracefully, ensuring that the next workflow experiences cold starts for its initial workers.

```bash
. activate_venv.sh
cd _examples/original
python _run_experiments.py
```

After the experiments are complete, you can visualize the results using the Planners Analysis Streamlit dashboard:

```bash
cd _metadata_analysis
streamlit run planners_analysis_dashboard.py
```

A Redis snapshot of the metrics database used to build the thesis and paper visualizations can be found in: `_metadata_analysis/snapshots/experiments_metrics_db_snapshot.rdb`.