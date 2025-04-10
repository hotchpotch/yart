"""
YART: Your Another Reranker Trainer
GOTCHA dataset implementation for cross-encoder training.
"""

import logging
import random
from typing import Dict, List, Optional, cast

from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from ..arguments import DataArguments
from ..data import DatasetForCrossEncoder

logger = logging.getLogger(__name__)


class GotchaDataset(DatasetForCrossEncoder):
    """
    Dataset implementation for GOTCHA positive-negative pairs.
    """

    def __init__(
        self,
        args: DataArguments,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        dataset: Optional[Dataset] = None,
    ):
        """
        Initialize the GOTCHA dataset.

        Args:
            args: Data arguments
            tokenizer: Tokenizer for encoding text
            dataset: Optional pre-loaded dataset
        """
        self.tokenizer = tokenizer
        self.args = args
        self.train_group_size = args.dataset_options.get("train_group_size", 16)
        self.max_length = args.max_length
        # self.shuffle_ds = args.shuffle_ds

        # Load dataset if not provided
        if dataset is None:
            dataset_name = args.dataset_options.get(
                "dataset_name", "hotchpotch/gotcha_pos_negs_v1"
            )
            logger.info(f"Loading GOTCHA dataset: {dataset_name}")
            dataset = load_dataset(dataset_name, split="train")  # type: ignore
            dataset = cast(Dataset, dataset)

            # Filter by target datasets if specified
            if args.target_ds:
                target_ds = args.target_ds
                logger.info(f"Filtering to specific datasets: {list(target_ds.keys())}")

                filtered_datasets = []
                for source_name, sample_size in target_ds.items():
                    # Skip if None (disabled)
                    if source_name is None:
                        continue

                    # Filter the dataset by source
                    source_ds = dataset.filter(
                        lambda example: example["source"] == source_name, num_proc=11
                    )

                    # Sample if sample size is specified
                    if sample_size is not None:
                        if len(source_ds) > sample_size:
                            source_ds = source_ds.shuffle(seed=42).select(
                                range(sample_size)
                            )
                        logger.info(
                            f"Selected {len(source_ds)} examples from {source_name}"
                        )

                    filtered_datasets.append(source_ds)

                if filtered_datasets:
                    dataset = concatenate_datasets(filtered_datasets)

            # Apply train_size limit if specified
            if args.train_size is not None and len(dataset) > args.train_size:
                dataset = dataset.shuffle(seed=42).select(range(args.train_size))

        # Shuffle dataset if requested
        if self.shuffle_ds:
            dataset = dataset.shuffle(seed=42, num_proc=11)  # type: ignore

        self._dataset = dataset
        self.total_len = len(self.dataset)
        logger.info(f"Initialized GOTCHA dataset with {self.total_len} examples")

    def __getitem__(self, item) -> List[Dict]:
        """Get an item at the specified index with multiple negative examples."""
        example = self._dataset[item]  # type: ignore

        query = example["query"]
        positive = example["positive"]

        # Get all available negatives
        negatives = example["negatives"]

        # Ensure we have enough negatives
        required_negs = self.train_group_size - 1
        if len(negatives) < required_negs:
            # Repeat negatives if needed
            negatives = (negatives * ((required_negs // len(negatives)) + 1))[
                :required_negs
            ]
        elif len(negatives) > required_negs:
            # Sample random negatives
            negatives = random.sample(negatives, required_negs)

        # Create batch
        batch_data = []
        batch_data.append(self.create_one_example(query, positive, 1.0))
        for neg in negatives:
            batch_data.append(self.create_one_example(query, neg, 0.0))

        return batch_data
