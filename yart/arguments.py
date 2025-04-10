"""
YART: Your Another Reranker Trainer
Arguments module for YART.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from transformers import TrainingArguments


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    classifier_dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout rate for the classifier layer"},
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Trust remote code for model loading"},
    )


@dataclass
class RankerTrainingArguments(TrainingArguments):
    """
    Arguments for training a cross-encoder ranker.
    """

    loss_name: str = field(
        default="cross_entropy",
        metadata={
            "help": "Loss function to use. Options: cross_entropy, mse, margin_ranking_loss"
        },
    )
    train_group_size: int = field(
        default=16, metadata={"help": "Number of examples per group for training"}
    )
    neg_shuffle: bool = field(
        default=False, metadata={"help": "Shuffle negative examples during training"}
    )
    disable_tqdm: bool = field(
        default=False, metadata={"help": "Disable tqdm progress bar"}
    )


@dataclass
class DataArguments:
    train_data: Any = field(
        default=None, metadata={"help": "Path or hf dataset to corpus"}
    )  # type: ignore
    train_group_size: int = field(default=16)
    train_max_positive_size: int = field(default=1)
    max_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input document length after tokenization for input text. "
        },
    )
    max_query_length: int = field(
        default=64,
        metadata={
            "help": "The maximum total input query length after tokenization for input text."
        },
    )
    dataset_options: dict = field(
        default_factory=dict, metadata={"help": "Additional options for the dataset"}
    )

    def __post_init__(self):
        # validation
        pass


@dataclass
class RunArguments:
    """
    Arguments for experiment run.
    """

    debug: bool = field(default=False, metadata={"help": "Enable debug mode"})
    output_prefix: str = field(
        default="", metadata={"help": "Prefix for output directory"}
    )
    remove_checkpoints: bool = field(
        default=False, metadata={"help": "Remove checkpoints after training"}
    )
