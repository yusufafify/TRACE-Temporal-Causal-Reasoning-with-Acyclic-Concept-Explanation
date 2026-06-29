import os
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

from env import HUGGINGFACEHUB_TOKEN


def get_llm_access():
    access_token = HUGGINGFACEHUB_TOKEN
    try:
      login(access_token)
      print("Successfully logged in to the Hugging Face Hub.")
    except Exception as e:
        raise ValueError("Failed to access the Hugging Face Hub.") from e


def get_pretrained_llm(cfg):
    """
    Load a pretrained language model and its tokenizer from the Hugging Face Hub with specified quantization settings.

    Args:
        llm_name (str): The name of the pretrained language model to load.
        load_in_fourbit (bool): Flag indicating whether to load the model in 4-bit precision. Default is True.
        **kwargs: Additional keyword arguments for quantization settings, which may include:
            - double_quant (bool): Flag indicating whether to use double quantization for 4-bit models. Default is True.
            - quant_type (str): The quantization type to use. Default is "nf4".
            - bit_compute_dtype (torch.dtype): The data type to use for 4-bit computation. Default is torch.bfloat16.

    Returns:
        tuple: A tuple containing the pretrained model and its tokenizer.
    
    Raises:
        ValueError: If the model ID is invalid or if loading fails.
    """
    # if the file exist in the 'pretrained_llms' directory, load it
    # if os.path.exists(f'pretrained_llms/{cfg.name}'):
    #     model = AutoModelForCausalLM.from_pretrained(f'pretrained_llms/{cfg.name}_model', device_map="auto")
    #     tokenizer = AutoTokenizer.from_pretrained(f'pretrained_llms/{cfg.name}_tokenizer')
    # else:
    # print(f"The model {cfg.name} is not found in the 'pretrained_llms' directory.")
    
    # Load the model with or without quantization
    if not cfg.load_in_fourbit:
        model = AutoModelForCausalLM.from_pretrained(cfg.llm_name, 
                                                       **cfg.llm_model_kwargs)
    else:
        # ------------------------------------------------
        # TODO: CHECK (also check dtype)
        # ------------------------------------------------
        # Extract quantization settings from kwargs, with defaults
        double_quant = cfg.fourbit_kwargs.get('double_quant', True)
        quant_type = cfg.fourbit_kwargs.get('quant_type', "nf4")
        bit_compute_dtype = torch.bfloat16
        
        print(f"The model is quantized with standard parameters: double quantization: {double_quant}, "
    f"quantization type: {quant_type}, bit compute dtype: {bit_compute_dtype}.")

        # Define quantization configuration
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=double_quant,
            bnb_4bit_quant_type=quant_type,
            bnb_4bit_compute_dtype=bit_compute_dtype
        )
        model = AutoModelForCausalLM.from_pretrained(cfg.llm_name, 
                                                     **cfg.llm_model_kwargs,
                                                     quantization_config=bnb_config)
        # ------------------------------------------------
        # ------------------------------------------------

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.llm_name)

        # save the model and tokenizer
        # model.save_pretrained(f'pretrained_llms/{cfg.name}_model')
        # tokenizer.save_pretrained(f'pretrained_llms/{cfg.name}_tokenizer')

    return model, tokenizer