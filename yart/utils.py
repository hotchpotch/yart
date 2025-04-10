"""
YART: Your Another Reranker Trainer
Utility functions.
"""

import logging
import os
import random
from pathlib import Path
from typing import Dict, Union

import numpy as np
import torch
import yaml
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from .arguments import (
    DataArguments,
    ModelArguments,
    RankerTrainingArguments,
    RunArguments,
)

logger = logging.getLogger(__name__)


def seed_everything(seed: int = 42):
    """
    Set random seed for reproducibility.

    Args:
        seed: Random seed
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore
    logger.info(f"Random seed set to {seed}")


def setup_logging(verbose: bool = False):
    """
    Setup logging configuration.

    Args:
        verbose: Whether to use verbose logging
    """
    logging_level = logging.DEBUG if verbose else logging.INFO

    # Configure root logger
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging_level,
    )

    # Set transformers logging
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.ERROR if not verbose else logging.INFO)


def setup_wandb(project_name: str = "ranker"):
    """
    Setup Weights & Biases logging.

    Args:
        project_name: W&B project name
    """
    if "WANDB_PROJECT" not in os.environ:
        os.environ["WANDB_PROJECT"] = project_name


def load_config_from_yaml(config_path: str) -> Dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML config file

    Returns:
        Dictionary containing configuration
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def parse_config_to_args(
    config: Dict,
) -> tuple[ModelArguments, DataArguments, RankerTrainingArguments, RunArguments]:
    """
    Parse configuration dictionary into argument objects.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (model_args, data_args, training_args, run_args)
    """
    model_args_dict = config.get("model_args", {})
    data_args_dict = config.get("data_args", {})
    training_args_dict = config.get("training_args", {})
    run_args_dict = config.get("run_args", {})

    # Create output_dir if specified
    if "output_dir" in training_args_dict:
        os.makedirs(training_args_dict["output_dir"], exist_ok=True)

    model_args = ModelArguments(**model_args_dict)
    data_args = DataArguments(**data_args_dict)
    training_args = RankerTrainingArguments(**training_args_dict)
    run_args = RunArguments(**run_args_dict)

    return model_args, data_args, training_args, run_args


def load_model_and_tokenizer(
    model_args: ModelArguments, training_args: RankerTrainingArguments
) -> tuple[PreTrainedModel, PreTrainedTokenizer | PreTrainedTokenizerFast]:
    """
    Load model and tokenizer based on arguments.

    Args:
        model_args: Model arguments
        training_args: Training arguments

    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model: {model_args.model_name_or_path}")

    # Load tokenizer
    tokenizer_name = model_args.tokenizer_name or model_args.model_name_or_path
    if not tokenizer_name:
        raise ValueError(
            "Either model_name_or_path or tokenizer_name must be provided."
        )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        use_fast=True,
        trust_remote_code=model_args.trust_remote_code,
    )

    # Load model
    model_config = {
        "num_labels": 1,  # Regression problem
        "problem_type": "regression",
        "classifier_dropout": model_args.classifier_dropout,
    }

    if not model_args.config_name:
        raise ValueError("Either model_name_or_path or config_name must be provided.")

    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_args.model_name_or_path),
        **model_config,
        trust_remote_code=model_args.trust_remote_code,
    )

    logger.info(f"Model loaded: {model.__class__.__name__}")

    return model, tokenizer


def remove_checkpoints(output_dir: Union[str, Path]):
    """
    Remove checkpoint directories after training.

    Args:
        output_dir: Output directory containing checkpoints
    """
    import shutil

    if isinstance(output_dir, str):
        output_dir = Path(output_dir)

    for checkpoint_dir in output_dir.glob("checkpoint-*"):
        if checkpoint_dir.is_dir():
            logger.info(f"Removing checkpoint directory: {checkpoint_dir}")
            shutil.rmtree(checkpoint_dir)
