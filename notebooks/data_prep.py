# Databricks notebook source
# MAGIC %md 
# MAGIC # This is the Data Prep to download the dataset and provide it in the correct prompt structure
# MAGIC
# MAGIC ## Dataset used : e2e_nlg

# COMMAND ----------

from datasets import load_dataset,DatasetDict,Dataset

ds = load_dataset("e2e_nlg_cleaned")

# COMMAND ----------

from datasets import load_dataset


def process(v):
    human_reference = v["human_reference"]
    meaning_representation = v["meaning_representation"]
    return {
        "text": f"""[INST] <<SYS>>Extract entities from the text given below.<</SYS>> {human_reference} [/INST]{meaning_representation}</s>"""
    }


ds = (
    ds.filter(
        lambda v: len(v["human_reference"]) > 1 and len(v["meaning_representation"]) > 1
    )
    .map(process, remove_columns=["meaning_representation", "human_reference"])
    .shuffle()
    
)

# COMMAND ----------

# MAGIC  %sh rm -rf /dbfs/pj/llm/datasets/e2e_nlg

# COMMAND ----------

# ds_limit = DatasetDict({"test" :ds["test"].select(range(1000)),
#                         "train" : ds["train"].select(range(1000)),
#                         "validation" : ds["validation"].select(range(1000))})

# COMMAND ----------

ds.save_to_disk("/dbfs/pj/llm/datasets/e2e_nlg")

# COMMAND ----------


