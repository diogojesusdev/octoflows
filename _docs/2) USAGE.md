# USAGE

Workflow examples can be found in the `_examples` directory.

## Running an "Image Transformation" workflow

The workflow is defined at `_examples/original/image_transformer.py`. To generate a graph representation of the workflow:

1) Activate the Python virtual environment and navigate to the workflow directory: 
```bash
. activate_venv.sh
cd _examples/original
```
> Note: You can generate a visual graph representation of workflows by calling `visualize_dag()` on their **sink node** (e.g., `final_img.visualize_dag()`)

2) Run the Workflow (using the Uniform planner)
```bash
python image_transformer.py uniform
```
> Note: You can monitor the workflow's progress in real time by setting `open_dashboard=False` in the `.compute()` function that triggers the workflow.

3) Checking workflow metrics
To visualize workflow metrics, two dashboards are available:

- **Planner Analysis Dashboard**: Shows aggregated metrics across different workflows for all planners, which is ideal for debugging the performance of planner algorithms.
```bash
cd _metadata_analysis
streamlit run planners_analysis_dashboard.py
```

- **Workflow Analysis Dashboard**: Shows metrics for specific workflow instances, which is ideal for debugging workflow performance.
```bash
cd _metadata_analysis
streamlit run workflow_analysis_dashboard.py
```
