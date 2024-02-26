# Databricks notebook source
# MAGIC %pip install -r ../requirements.txt

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 2

# COMMAND ----------

from huggingface_hub import notebook_login, login

notebook_login()

# COMMAND ----------

import os

os.environ["HF_HOME"] = "/local_disk0/hf"
os.environ["HF_DATASETS_CACHE"] = "/local_disk0/hf"
os.environ["TRANSFORMERS_CACHE"] = "/local_disk0/hf"
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_DEBUG"] = "INFO"

# COMMAND ----------

# import logging

# logging.basicConfig(
#     format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
#     level=logging.INFO,
#     datefmt="%Y-%m-%d %H:%M:%S",
# )
# logging.getLogger("py4j").setLevel(logging.ERROR)
# logging.getLogger("sh.command").setLevel(logging.ERROR)

# COMMAND ----------

from databricks_llm.notebook_utils import get_dbutils

# COMMAND ----------

DEFAULT_INPUT_MODEL = "meta-llama/Llama-2-7b-chat-hf"
SUPPORTED_INPUT_MODELS = [
    "mosaicml/mpt-30b-instruct",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "mosaicml/mpt-7b-instruct",
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-2-13b-chat-hf",
    "meta-llama/Llama-2-70b-chat-hf",
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-13b-hf",
    "meta-llama/Llama-2-70b-hf",
    "codellama/CodeLlama-7b-hf",
    "codellama/CodeLlama-13b-hf",
    "codellama/CodeLlama-34b-hf",
    "codellama/CodeLlama-7b-Instruct-hf",
    "codellama/CodeLlama-13b-Instruct-hf",
    "codellama/CodeLlama-34b-Instruct-hf",
    "tiiuae/falcon-7b-instruct",
    "tiiuae/falcon-40b-instruct",
    "HuggingFaceH4/starchat-beta",
]

# COMMAND ----------

get_dbutils().widgets.text("num_gpus", "4", "num_gpus")
get_dbutils().widgets.text("dbfs_output_location", "/dbfs/llm/", "dbfs_output_location")
get_dbutils().widgets.combobox(
    "pretrained_name_or_path",
    DEFAULT_INPUT_MODEL,
    SUPPORTED_INPUT_MODELS,
    "pretrained_name_or_path",
)
get_dbutils().widgets.text(
    "dataset",
    "/dbfs/pj/llm/datasets/e2e_nlg",
    "dataset",
)

# COMMAND ----------

num_gpus = get_dbutils().widgets.get("num_gpus")
pretrained_name_or_path = get_dbutils().widgets.get("pretrained_name_or_path")
dataset = get_dbutils().widgets.get("dataset")
dbfs_output_location = get_dbutils().widgets.get("dbfs_output_location")

# COMMAND ----------

!echo {dataset}

# COMMAND ----------

!mkdir -p {dbfs_output_location}

# COMMAND ----------

# MAGIC %load_ext tensorboard
# MAGIC %tensorboard --logdir '/local_disk0/output/runs'

# COMMAND ----------

import os
os.environ['DATABRICKS_TOKEN'] = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
os.environ['DATABRICKS_HOST'] = "https://" + spark.conf.get("spark.databricks.workspaceUrl")
os.environ['MLFLOW_EXPERIMENT_NAME'] = "/Users/puneet.jain@databricks.com/mlflow_blog"
os.environ['MLFLOW_FLATTEN_PARAMS'] = "true"
os.environ['HF_MLFLOW_LOG_ARTIFACTS'] = "true" 
os.environ['MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING'] = "true"

# COMMAND ----------

!export DATABRICKS_TOKEN && export DATABRICKS_HOST && export MLFLOW_EXPERIMENT_NAME && export MLFLOW_FLATTEN_PARAMS &&  \
 export HF_MLFLOW_LOG_ARTIFACTS && export MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING && cd .. && deepspeed \
--num_gpus='{num_gpus}' \
--module databricks_llm.fine_tune \
--final_model_output_path='{dbfs_output_location}' \
--output_dir="/local_disk0/output" \
--dataset={dataset} \
--model={pretrained_name_or_path} \
--tokenizer={pretrained_name_or_path} \
--use_lora=false \
--use_4bit=false \
--deepspeed_config="ds_configs/ds_zero_3_cpu_offloading.json" \
--fp16=false \
--bf16=true \
--per_device_train_batch_size=16 \
--per_device_eval_batch_size=48 \
--gradient_checkpointing=true \
--gradient_accumulation_steps=1 \
--learning_rate=1e-6 \
--lr_scheduler_type="cosine" \
--warmup_steps=50 \
--evaluation_strategy="steps" \
--save_strategy="steps" \
--save_steps=100 \
--num_train_epochs=2\
--run_name="{pretrained_name_or_path}-blog-run"

# COMMAND ----------

print(dbfs_output_location)
