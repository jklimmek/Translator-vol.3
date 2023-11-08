import re
import random
import numpy as np
import torch
import yaml


def read_txt(path, skip_blank=True):
    """
    Reads a text file and returns a list of lines.

    Parameters:
        path (str): Path to text file.
        skip_blank (bool): Whether to skip blank lines.

    Returns:
        lines (list): List of lines.
    """

    lines = []
    with open(path, 'r', encoding="utf-8") as file:
        for line in file:
            if skip_blank and line == '\n':
                continue
            lines.append(line.strip())
    return lines


def save_txt(txt, path, mode="w", skip_blank=True):
    """
    Saves a list of lines to a text file.

    Parameters:
        txt (list): List of lines.
        path (str): Path to text file.
        mode (str): Mode to open file.
        skip_blank (bool): Whether to skip blank lines.
    """
    
    with open(path, mode=mode, encoding='utf-8') as file:
        for line in txt:
            if skip_blank and line == '':
                continue
            file.write(line + '\n')


def read_yaml(path):
    """
    Reads a YAML file converts values to float if they are in scientific notation.

    Parameters:
        path (str): Path to YAML file.

    Returns:
        config (dict): Dictionary of YAML file.
    """
    with open(path, 'r') as file:
        params = yaml.safe_load(file)
    
    for key, value in params.items():
        if isinstance(value, str):
            if re.match(r"\d+\.?\d*e[-+]?\d+", value):
                params[key] = float(value)
    return params
            


def get_model_size(model):
    """
    Returns the model size in millions of parameters.

    Parameters:
        model (nn.Module): Model to get size of.

    Returns:
        model_size (str): Model size in millions of parameters.
    """
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    million_params = num_params // 1_000_000
    return f"{model.__class__.__name__}_{million_params}M"


def params_check(architecture):
    """
    Checks if the architecture params are consistent.

    Parameters:
        architecture (dict): Dictionary of architecture params.
    """
    
    enc_vocab_size = architecture["encoder"]["position_encoding_config"]["vocab_size"]
    dec_vocab_size = architecture["decoder"]["position_encoding_config"]["vocab_size"]
    assert enc_vocab_size == dec_vocab_size == architecture["vocab_size"], "Vocab size mismatch."

    enc_dim_model = architecture["encoder"]["dim_model"]
    dec_dim_model = architecture["decoder"]["dim_model"]
    assert enc_dim_model == dec_dim_model == architecture["dim"], "Embedding dim mismatch."


def seed_all(seed):
    """
    Sets the seed for reproducibility.
    """
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)