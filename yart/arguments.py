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


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_data: Any = field(
        default=None, metadata={"help": "Path or hf dataset to corpus"}
    )
    train_size: Optional[int] = field(
        default=None, metadata={"help": "Number of training examples to use"}
    )
    test_size: int = field(
        default=1000, metadata={"help": "Number of test examples to use"}
    )
    max_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization for input text."
        },
    )
    dataset_options: Dict[str, Any] = field(
        default_factory=dict, metadata={"help": "Additional options for the dataset"}
    )
    shuffle_ds: bool = field(
        default=True, metadata={"help": "Shuffle dataset before training"}
    )
    target_ds: Optional[Dict[str, Optional[int]]] = field(
        default=None,
        metadata={"help": "Target datasets to use with optional sample sizes"},
    )
    pick_top_100: int = field(
        default=0, metadata={"help": "Number of negatives to pick from top 100 results"}
    )
    slice_top_100_k: int = field(
        default=50, metadata={"help": "Start index for slicing top 100 IDs"}
    )


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
