# EXPERIMENTS

After configuring the environment as described in `DEPLOYMENT.md`, the set of experiments used in the paper can be ran using the Python script `_examples/original/_run_experiments.py`. Here it's possible to configure the workflows to run, SLAs to use, Planners, and number of instances for each possible combination. The script then runs the experiments sequentially. 

> Note: Before starting a new experiment, it waits for all containers to shutdown gracefully, ensuring next workflow will experience cold starts for initial workers.

```bash
. activate_venv.sh
cd _examples/original
python _run_experiments.py
```

Afterwards, you can visualize the results using the Planners Analysis Streamlit dashboard:

```bash
cd _metadata_analysis
streamlit run planners_analysis_dashboard.py
```
