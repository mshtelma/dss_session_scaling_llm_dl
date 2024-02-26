import importlib.util
import json
import os
import pathlib
import shutil,glob

from transformers.integrations import MLflowCallback
from transformers.trainer_callback import TrainerCallback
from transformers.utils.generic import flatten_dict
from transformers.utils import logging 
from transformers.utils.import_utils import ENV_VARS_TRUE_VALUES
from transformers import AutoModelForCausalLM, AutoTokenizer


logger = logging.get_logger(__name__)

DEFAULT_SYSTEM_PROMPT = """Extract entities from the text below"""

def build_prompt(instruction):
    return f"""<s>[INST]<<SYS>>{DEFAULT_SYSTEM_PROMPT}<</SYS>>{instruction}[/INST]"""


import mlflow
from mlflow.models import infer_signature

# Define model signature including params
input_example = {"prompt": build_prompt("The Vaults pub near Café Adriatic has a 5 star rating. Prices start at £30")}
inference_config = {
    "temperature": 0.5,
    "max_new_tokens": 100,
    "do_sample": True,
}
signature = infer_signature(
    model_input=input_example,
    model_output="name[The Vaults] ....",
    params=inference_config
)

def is_mlflow_available():
    if os.getenv("DISABLE_MLFLOW_INTEGRATION", "FALSE").upper() == "TRUE":
        return False
    return importlib.util.find_spec("mlflow") is not None


class CustomMLflowCallback(MLflowCallback):
    """
    A [`TrainerCallback`] that sends the logs to [MLflow](https://www.mlflow.org/). Can be disabled by setting
    environment variable `DISABLE_MLFLOW_INTEGRATION = TRUE`.
    """

    def __init__(self):
        if not is_mlflow_available():
            raise RuntimeError("MLflowCallback requires mlflow to be installed. Run `pip install mlflow`.")
        import mlflow

        self._MAX_PARAM_VAL_LENGTH = mlflow.utils.validation.MAX_PARAM_VAL_LENGTH
        self._MAX_PARAMS_TAGS_PER_BATCH = mlflow.utils.validation.MAX_PARAMS_TAGS_PER_BATCH

        self._initialized = False
        self._auto_end_run = False
        self._log_artifacts = False
        self._ml_flow = mlflow

    def on_save(self, args, state, control,model,tokenizer, **kwargs):
        if self._initialized and state.is_world_process_zero and self._log_artifacts:
            ckpt_dir = f"checkpoint-{state.global_step}"
            artifact_path = os.path.join(args.output_dir, ckpt_dir)
            log_path = os.path.join(args.output_dir, f"tmp/{ckpt_dir}")
            logger.info(f"Loading model from checkpoint")
            # model_load = model.to("cpu")
            # model.save_pretrained(os.path.join(args.output_dir, f"tmp/{ckpt_dir}"))
            shutil.copytree(artifact_path, log_path,ignore =shutil.ignore_patterns("global_step*"),dirs_exist_ok=True)
            # tokenizer.save_pretrained(os.path.join(args.output_dir, f"tmp/{ckpt_dir}"))
            logger.info(f"Logging checkpoint artifacts in {ckpt_dir}. This may take time.")

        
            # self._ml_flow.transformers.log_model(
            #                 transformers_model={
            #                     "model": model_load,
            #                     "tokenizer": tokenizer,
            #                 },
            #                 task="text-generation",
            #                 artifact_path=ckpt_dir,
            #                 pip_requirements=["torch", "transformers", "accelerate"],
            #                 input_example=input_example,
            #                 signature=signature,
            #                 # Add the metadata task so that the model serving endpoint created later will be optimized
            #                 metadata={"task": "llm/v1/completions"}
            #             )



            self._ml_flow.pyfunc.log_model(
                            ckpt_dir,
                            python_model=LLMPyFuncModel(),
                            artifacts={"repository": os.path.join(args.output_dir, f"tmp/{ckpt_dir}")},
                            pip_requirements=[
                                "torch==2.0.1",
                                "transformers",
                                "accelerate",
                                "einops",
                                "sentencepiece",
                            ],
                            input_example=input_example,
                            signature=signature,
                        )
            shutil.rmtree(os.path.join(args.output_dir, f"tmp/{ckpt_dir}"))
            for f in glob.glob("/tmp/tmp*"):
                try:
                    shutil.rmtree(f)
                except: 
                    logger.info(f"Could not delete file {f}")

class LLMPyFuncModel(mlflow.pyfunc.PythonModel):
    def __init__(
        self
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
