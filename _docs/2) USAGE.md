# USAGE

Workflow Examples can be found under `_examples`

## Running an "Image Transformation" workflow

Workflow Defined at: `_examples/original/image_transformer.py`. To generate an image of the a graph representation of the workflow

1) Activate the Python Virtual Environment and go to the workflow folder 
```bash
. activate_venv.sh
cd _examples/original
```
> Note: You can generate a visual graph representation of workflows by calling `visualize_dag()` on their **sink node** (e.g., `final_img.visualize_dag()`)

2) Run the Workflow (using the Uniform planner)
```bash
python image_transformer.py uniform
```
> Note: You can see workflow progression in real-time by setting `open_dashboard=False` on the `.compute()` function that triggers the workflow

3) Checking workflow metrics
To visualize workflow metrics, you have 2 dashboards:

- **Planner Analysis Dashboard** (shows aggregated metrics across different workflows for all planners, ideal for debugging planner algorithms performance)
```bash
cd _metadata_analysis
streamlit run planners_analysis_dashboard.py
```

- **Workflow Analysis Dashboard** (shows metrics for a specific workflow instances, ideal for debugging workflow performance)
```bash
cd _metadata_analysis
streamlit run workflow_analysis_dashboard.py
```
