# Databricks notebook source
# MAGIC %pip install torch==2.0.1

# COMMAND ----------

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
    "mlabonne/guanaco-llama2",
    "dataset",
)

# COMMAND ----------

num_gpus = get_dbutils().widgets.get("num_gpus")
pretrained_name_or_path = get_dbutils().widgets.get("pretrained_name_or_path")
dataset = get_dbutils().widgets.get("dataset")
dbfs_output_location = get_dbutils().widgets.get("dbfs_output_location")

# COMMAND ----------

!echo {num_gpus}

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
--num_train_epochs=3\
--run_name="{pretrained_name_or_path}-blog-run"

# COMMAND ----------

print(dbfs_output_location)

# COMMAND ----------

import mlflow

loaded_model = mlflow.pyfunc.load_model(f"runs:/4b32ccd588794530bd66fd3557c4d42c/checkpoint-900")

# Make a prediction using the loaded model
loaded_model.predict(
    {"prompt": "check out Gaucho , near convent garden cost $50 for 2, nice columbian dishes",
     "temperature": 0.4,
     "max_tokens": 128,
    }
)


# COMMAND ----------

import transformers,torch
from transformers import AutoModelForCausalLM, AutoTokenizer
model = "/tmp/tmpdhdct314/checkpoint-200/artifacts/checkpoint-200"
# revision = "08751db2aca9bf2f7f80d2e516117a53d7450235"

tokenizer = AutoTokenizer.from_pretrained(model,trust_remote_code=True)

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    # revision=revision,
    return_full_text=False
)

# Required tokenizer setting for batch inference
pipeline.tokenizer.pad_token_id = tokenizer.eos_token_id

# COMMAND ----------

# Define prompt template to get the expected features and performance for the chat versions. See our reference code in github for details: https://github.com/facebookresearch/llama/blob/main/llama/generation.py#L212

DEFAULT_SYSTEM_PROMPT = """\
Extract entities from the text below."""
PROMPT_FOR_GENERATION_FORMAT = """
<s>[INST]<<SYS>>
{system_prompt}
<</SYS>>
{instruction}
[/INST]
""".format(
    system_prompt=DEFAULT_SYSTEM_PROMPT,
    instruction="{instruction}"
)

# COMMAND ----------

# Define parameters to generate text
def gen_text(prompts, use_template=True, **kwargs):
    if use_template:
        full_prompts = [
            PROMPT_FOR_GENERATION_FORMAT.format(instruction=prompt)
            for prompt in prompts
        ]
    else:
        full_prompts = prompts

    if "batch_size" not in kwargs:
        kwargs["batch_size"] = 1
    
    # the default max length is pretty small (20), which would cut the generated output in the middle, so it's necessary to increase the threshold to the complete response
    if "max_new_tokens" not in kwargs:
        kwargs["max_new_tokens"] = 128

    # configure other text generation arguments, see common configurable args here: https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
    kwargs.update(
        {
            "pad_token_id": tokenizer.eos_token_id,  # Hugging Face sets pad_token_id to eos_token_id by default; setting here to not see redundant message
            "eos_token_id": tokenizer.eos_token_id,
        }
    )
    # kwargs["temperature"] = 0.4
    # kwargs["repetition_penalty"] = 2.0
    # kwargs["do_sample"] = True
    print(full_prompts)
    print(kwargs)
    outputs = pipeline(full_prompts, **kwargs)
    outputs = [out[0]["generated_text"] for out in outputs]

    return outputs

# COMMAND ----------


# COMMAND ----------

results = gen_text(["The Restaurant Gymkhana near Marlybenone station has a high customer star rating and offers a unique Indian cuisines"])
# results = gen_text(["check out Gaucho , near convent garden cost $50 for 2, nice columbian dishes"])
print(results)

# COMMAND ----------

class c():
  def __init__(
        self,loc =None
    ):
    self.artifacts =  {"repository":loc}

cont = c("/tmp/tmp_txt_k02/checkpoint-900/artifacts/checkpoint-900")


# COMMAND ----------

# 
  
class LLMPyFuncModel(mlflow.pyfunc.PythonModel):
    def __init__(
        self,context
    ):
        pass

    def load_context(self, context):
        """
        This method initializes the tokenizer and language model
        using the specified model repository.
        """
        import transformers,torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        # Initialize tokenizer and language model
        self.tokenizer = AutoTokenizer.from_pretrained(
            context.artifacts["repository"],
            trust_remote_code=True
        )
        self.pipeline = transformers.pipeline(
                "text-generation",
                model=context.artifacts["repository"],
                tokenizer=self.tokenizer,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map="auto",
              return_full_text=False)
        # Required tokenizer setting for batch inference
        self.pipeline.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.DEFAULT_SYSTEM_PROMPT = """Extract entities from the text below."""
        self.PROMPT_FOR_GENERATION_FORMAT = """"\n<s>[INST]<<SYS>>\n{system_prompt}\n<</SYS>>\n{instruction}\n[/INST]""".format(
            system_prompt=self.DEFAULT_SYSTEM_PROMPT,
            instruction='{instruction}')

    def predict(self, context, model_input):
        """
        This method generates prediction for the given input.
        """

        kwargs ={}
        # kwargs['temperature'] = model_input.get("temperature", [1.0])[0]
        kwargs['max_new_tokens'] = model_input.get("max_new_tokens", [100])[0]
        # top_p = model_input.get("top_p", [100])[0]
        for key in model_input:
            if key not in ['prompt','use_template']:
                kwargs[key] = model_input[key][0]

        if "use_template" in model_input.keys():
            use_template =  model_input.get("use_template", [True])[0]
        else : 
            use_template = True

        if use_template:
            prompts = [self.PROMPT_FOR_GENERATION_FORMAT.format(instruction=p) for p in model_input.get("prompt") ]


        return self.gen_text(prompts,kwargs)
    
    def gen_text(self,prompts, kwargs):

        if "batch_size" not in kwargs:
            kwargs["batch_size"] = 1

        # configure other text generation arguments, see common configurable args here: https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig
        kwargs.update(
            {
                "pad_token_id": self.tokenizer.eos_token_id,  # Hugging Face sets pad_token_id to eos_token_id by default; setting here to not see redundant message
                "eos_token_id": self.tokenizer.eos_token_id,
            })
        outputs = self.pipeline(prompts, **kwargs)

        outputs = [out[0]["generated_text"] for out in outputs]

        return outputs

# COMMAND ----------

p =LLMPyFuncModel(cont)

# COMMAND ----------

p.predict( model_input={"prompt": ["check out Gaucho , near convent garden cost $50 for 2, nice columbian dishes",
                                   "The Restaurant Gymkhana near Marlybenone station has a high customer star rating and offers a unique Indian cuisines"],
     "max_new_tokens": [128],
    },
    context =cont
)

# COMMAND ----------

import glob, os,shutil
for f in glob.glob("/tmp/tmp*"):
  shutil.rmtree(f)

# COMMAND ----------


