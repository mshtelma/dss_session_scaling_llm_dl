import json
import logging

import os
from typing import Tuple

import torch

from datasets import Dataset, load_dataset, load_from_disk, DatasetDict
from transformers.integrations import MLflowCallback

from huggingface_hub import login

from transformers import (
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    PreTrainedTokenizer,
)

from databricks_llm.model_utils import get_model_and_tokenizer, get_tokenizer
from databricks_llm.callbacks import CustomMLflowCallback
from databricks_llm.utils import ExtendedTrainingArguments

logger = logging.getLogger(__name__)


def load_training_dataset(
    tokenizer,
    path_or_dataset: str,
    split: str,
    dataset_text_field: str = "text",
    max_seq_len: int = 512,
    formatting_func=None,
) -> Dataset:
    logger.info(f"Loading dataset from {path_or_dataset}")
    if path_or_dataset.startswith("/"):
        dataset = load_from_disk(path_or_dataset)
        if isinstance(dataset, DatasetDict):
            dataset = dataset[split]
            print(
                f"Loaded dataset {path_or_dataset} from disk for split {split} with {len(dataset)} rows."
            )
    else:
        dataset = load_dataset(path_or_dataset, split=split)
        print(
            f"Loaded dataset {path_or_dataset} from HF Hub for split {split} with {len(dataset)} rows."
        )
    logger.info("Found %d rows", dataset.num_rows)
    logger.info("Found %d rows", len(dataset))

    use_formatting_func = formatting_func is not None and dataset_text_field is None

    # Inspired from: https://huggingface.co/learn/nlp-course/chapter7/6?fw=pt
    def tokenize(element):
        input_batch = []
        attention_masks = []

        outputs = tokenizer(
            element[dataset_text_field]
            if not use_formatting_func
            else formatting_func(element),
            truncation=True,
            padding=True,
            max_length=max_seq_len,
            return_overflowing_tokens=False,
            return_length=True,
        )

        for length, input_ids, attention_mask in zip(
            outputs["length"], outputs["input_ids"], outputs["attention_mask"]
        ):
            # if length == max_seq_len:
            input_batch.append(input_ids)
            attention_masks.append(attention_mask)

        return {"input_ids": input_batch, "attention_mask": attention_masks}

    tokenized_dataset = dataset.map(
        tokenize, batched=True, remove_columns=dataset.column_names
    )

    print(len(tokenized_dataset))

    return tokenized_dataset


def setup_hf_trainer(
    train_dataset, eval_dataset=None, **config
) -> Tuple[Trainer, AutoModelForCausalLM, PreTrainedTokenizer]:
    args: ExtendedTrainingArguments = config["args"]

    torch.backends.cuda.matmul.allow_tf32 = True

    training_args = TrainingArguments(
        local_rank=args.local_rank,
        output_dir=args.output_dir,
        run_name=args.run_name,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        optim=args.optim,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        adam_epsilon=args.adam_epsilon,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        logging_strategy=args.logging_strategy,
        evaluation_strategy=args.evaluation_strategy,
        save_strategy=args.save_strategy,
        fp16=args.fp16,
        bf16=args.bf16,
        deepspeed=config.get("deepspeed_config_dict", None),
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        push_to_hub=False,
        disable_tqdm=True,
        report_to=["tensorboard"],
        # group_by_length=True,
        ddp_find_unused_parameters=False,
        # fsdp=["full_shard", "offload"],
    )

    model, tokenizer = get_model_and_tokenizer(
        args.model, use_4bit=args.use_4bit, load_in_8bit=False, use_lora=args.use_lora
    )
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    return trainer, model, tokenizer


def train(args: ExtendedTrainingArguments):
    tokenizer = get_tokenizer(args.tokenizer)
    train_dataset = load_training_dataset(
        tokenizer, args.dataset, "train", "text", 256, formatting_func=None
    )
    # train_dataset = train_dataset.select(range(2000))
    eval_dataset = load_training_dataset(
        tokenizer, args.dataset, "test", "text", 256, formatting_func=None
    )
    # eval_dataset = eval_dataset.select(range(2000))
    if args.deepspeed_config:
        with open(args.deepspeed_config) as json_data:
            deepspeed_config_dict = json.load(json_data)
    else:
        deepspeed_config_dict = None
    trainer, model, tokenizer = setup_hf_trainer(
        train_dataset,
        eval_dataset,
        args=args,
        deepspeed_config_dict=deepspeed_config_dict,
    )
    trainer.add_callback(CustomMLflowCallback)
    trainer.train()
    
    trainer.save_model(args.final_model_output_path)
    tokenizer.save_pretrained(args.final_model_output_path)


def main():
    import pathlib

    parser = HfArgumentParser(ExtendedTrainingArguments)

    parsed = parser.parse_args_into_dataclasses()
    args: ExtendedTrainingArguments = parsed[0]

    if args.token is not None and len(args.token):
        login(args.token)
    elif pathlib.Path("/root/.cache/huggingface/token").exists():
        login(pathlib.Path("/root/.cache/huggingface/token").read_text())

    train(args)


if __name__ == "__main__":
    os.environ["HF_HOME"] = "/local_disk0/hf"
    os.environ["HF_DATASETS_CACHE"] = "/local_disk0/hf"
    os.environ["TRANSFORMERS_CACHE"] = "/local_disk0/hf"
    os.environ["NCCL_P2P_DISABLE"] = "1"
    os.environ["NCCL_DEBUG"] = "INFO"

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    try:
        main()
    except Exception:
        logger.exception("main failed")
        raise
