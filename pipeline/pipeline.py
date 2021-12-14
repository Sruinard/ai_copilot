from os import name
from azureml.core import Workspace, Datastore, Dataset, Environment, Experiment, Run
from azureml.data import OutputFileDatasetConfig
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import Pipeline, PipelineRun
from azureml.core.runconfig import RunConfiguration
from datetime import datetime

from azureml.core.environment import Environment
from azureml.core.conda_dependencies import CondaDependencies

# Adds dependencies to PythonSection of myenv


ws = Workspace.from_config()
run = Run.get_context()
datastore = ws.datastores['aicopilot_pipelines_datastore']
compute_target = ws.compute_targets["copilotgpt"]
environment = Environment.get(ws, 'aicopilot_curated')


aml_run_config = RunConfiguration()
aml_run_config.target = compute_target
aml_run_config.environment = environment

run = Run.get_context()
timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
ingestion_dataset = Dataset.get_by_name(ws, name='code_to_text_python')

example_gen = OutputFileDatasetConfig('example_gen', destination=(datastore, f'{timestamp}/example_gen/')).as_upload()
transform = OutputFileDatasetConfig('transform', destination=(datastore, f'{timestamp}/transform/')).as_upload()
model_artifacts = OutputFileDatasetConfig('model_artifacts', destination=(datastore, f'{timestamp}/serving_dir/')).as_upload()

example_gen_step = PythonScriptStep(
    name='example_gen',
    script_name="./utils/build_dataset.py",
    compute_target=compute_target,
    runconfig=aml_run_config,
    arguments = ['--code-to-text-dataset', ingestion_dataset.as_named_input("raw_data").as_download(), '--example-gen-output', example_gen],
    allow_reuse=False
)

preprocessing_step = PythonScriptStep(
    name="preprocessing",
    script_name="./trainer/preprocessing.py",
    compute_target=compute_target,
    runconfig=aml_run_config,
    arguments = ['--example-gen', example_gen.as_input(), '--transform', transform]
)

training_step = PythonScriptStep(
    name="train",
    script_name="./trainer/task.py",
    compute_target=compute_target,
    runconfig=aml_run_config,
    arguments = ['--transform', transform.as_input(), '--serving-dir', model_artifacts]
)

pipeline_object = Pipeline(ws, steps=[example_gen_step, preprocessing_step, training_step])

# preprocessing_step = PythonScriptStep(
#     name="preprocessing",
#     script_name="./trainer/check_if_read_output.py",
#     compute_target=compute_target,
#     runconfig=aml_run_config,
# )

# pipeline_object = Pipeline(ws, steps=[preprocessing_step])

experiment = Experiment(ws, "ai_copilot").submit(pipeline_object)
experiment.wait_for_completion()
