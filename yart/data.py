"""
YART: Your Another Reranker Trainer
Data loading and preprocessing module.
"""

import logging
import math
import random
import warnings
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Type, cast

import torch
from datasets import Dataset, concatenate_datasets, load_dataset, load_from_disk
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    BatchEncoding,
    DataCollatorWithPadding,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from .arguments import DataArguments

logger = logging.getLogger(__name__)


class DatasetForCrossEncoder(TorchDataset):
    """
    Base dataset class for cross-encoder training.
    """

    def __init__(
        self,
        args: DataArguments,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        dataset: Dataset | None = None,
    ):
        if not dataset:
            train_data = args.train_data  # list or str
            if isinstance(train_data, list):
                datasets = []
                for target in train_data:
                    logger.info(f"Loading {target}")
                    datasets.append(self.load_dataset(target))
                self.dataset = concatenate_datasets(datasets)
            else:
                logger.info(f"Loading {train_data}")
                self.dataset = self.load_dataset(train_data)
        else:
            self.dataset = dataset

        self.dataset: Dataset = cast(Dataset, self.dataset)
        self.tokenizer = tokenizer
        self.args = args
        self.total_len = len(self.dataset)
        self.max_length = args.max_length
        # self.shuffle_ds = args.shuffle_ds

        # if self.shuffle_ds:
        #     self.dataset = self.dataset.shuffle(seed=42)

    def load_dataset(self, target_name: str) -> Dataset:
        """
        Load a dataset from a path or identifier.
        """
        if target_name.endswith(".arrow") or any(
            [":" in target_name, "/" in target_name]
        ):
            logger.info(f"Loading dataset from {target_name}")
            return load_from_disk(target_name)  # type: ignore
        else:
            logger.info(f"Loading dataset {target_name} with split='train'")
            return load_dataset(target_name, split="train")  # type: ignore

    def create_one_example(
        self, query: str, document: str, label: float
    ) -> BatchEncoding:
        """
        Create one example by encoding a query-document pair.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Suppress warnings
            if len(query) == 0:
                logger.warning("Empty query detected")
                query = "none"
            if len(document) == 0:
                logger.warning("Empty document detected")
                document = "none"
            item = self.tokenizer.encode_plus(
                query,
                document,
                truncation="only_second",
                max_length=self.max_length,
                padding=False,
            )
            item["labels"] = label
            return item

    def __len__(self):
        return self.total_len


class TrainDatasetForCE(DatasetForCrossEncoder):
    def __init__(
        self,
        args: DataArguments,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        dataset: Dataset | None = None,
        train_group_size: int = 16,
        neg_shuffle: bool = False,
    ):
        super().__init__(args, tokenizer, dataset)
        self.train_group_size = train_group_size
        self.neg_shuffle = neg_shuffle

    def __getitem__(self, item) -> List[BatchEncoding]:
        pos_key = "positive"
        neg_key = "negatives"

        query = self.dataset[item]["anchor"]
        pos = self.dataset[item][pos_key]

        # Default scores if not provided
        pos_score = self.dataset[item].get("positive_score", 1.0)
        if isinstance(pos_score, list) and len(pos_score) > 0:
            pos_score = pos_score[0]

        target_negs = self.dataset[item][neg_key]
        negs_max_size = self.train_group_size - 1

        negs_score = self.dataset[item].get("negatives_score", [0.0] * negs_max_size)
        if len(negs_score) > negs_max_size:
            negs_score = negs_score[:negs_max_size]
        elif len(negs_score) < negs_max_size:
            negs_score = negs_score + [0.0] * (negs_max_size - len(negs_score))

        # Check if target_negs and negs_score size matches
        if len(target_negs) != len(negs_score):
            logger.warning(
                f"Warning: target_negs ({len(target_negs)}) and negs_score ({len(negs_score)}) size is not match, override scores"
            )
            pos_score = 1.0
            negs_score = [0.0 for _ in range(negs_max_size)]

        if len(target_negs) < negs_max_size:
            # If not enough negative samples, repeat what we have
            logger.warning(
                f"Warning: not enough negative samples for query: {query}, target_neg len: {len(target_negs)}"
            )
            num = math.ceil((negs_max_size) / len(target_negs))
            if self.neg_shuffle:
                negs = random.sample(target_negs * num, negs_max_size)
            else:
                negs = (target_negs * num)[0:negs_max_size]
        else:
            if self.neg_shuffle:
                negs = random.sample(target_negs, negs_max_size)
            else:
                negs = target_negs[0:negs_max_size]

        batch_data = []
        batch_data.append(self.create_one_example(query, pos, pos_score))  # type: ignore
        for neg, neg_score in zip(negs, negs_score):
            batch_data.append(self.create_one_example(query, neg, neg_score))

        return batch_data


@dataclass
class GroupCollator(DataCollatorWithPadding):
    """
    Data collator that collates examples into batches, flattening nested lists.
    """

    def __call__(self, features):
        if isinstance(features[0], list):
            features = sum(features, [])  # type: ignore
        return super().__call__(features)  # type: ignore


def detect_dataset_klass(dataset_path: str) -> Type[DatasetForCrossEncoder]:
    """
    Dynamically import and return a dataset class from a path.
    """
    import importlib

    module_path, class_name = dataset_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    dataset_class = getattr(module, class_name)
    return dataset_class


def create_dateset_from_args(
    args: DataArguments, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast
) -> DatasetForCrossEncoder:
    train_data = args.train_data
    if isinstance(train_data, str):
        target_ds = DatasetForCrossEncoder(args, tokenizer)
    elif isinstance(train_data, list):
        target_ds_list = []
        for target_train_data in train_data:
            dataset_class_args = deepcopy(args)
            if isinstance(target_train_data, str):
                dataset_class_args.train_data = target_train_data
                target_ds_list.append(
                    DatasetForCrossEncoder(dataset_class_args, tokenizer)
                )
            elif isinstance(target_train_data, dict):
                dataset_class_name = target_train_data.get("dataset_class")
                dataset_options = target_train_data.get("dataset_options", {})
                # merge dataset_options
                dataset_class_args.dataset_options.update(dataset_options)

                if not dataset_class_name:
                    raise ValueError(f"dataset_class is required, {target_train_data}")
                dataset_class_train_data = target_train_data.get("train_data")
                if dataset_class_train_data:
                    dataset_class_args.train_data = dataset_class_train_data
                dataset_klass = detect_dataset_klass(dataset_class_name)
                target_ds_list.append(dataset_klass(dataset_class_args, tokenizer))
            else:
                raise ValueError(f"Invalid type {target_train_data}")
        target_ds = torch.utils.data.ConcatDataset(target_ds_list)
    else:
        raise ValueError(
            f"Invalid type {type(train_data)}, expected str or list of str"
        )
    return target_ds  # type: ignore
