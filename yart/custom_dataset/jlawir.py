"""
YART: Your Another Reranker Trainer
Japanese Large QA Dataset for Information Retrieval (JLaWIR) Dataset implementation.
"""

import logging
import random
from typing import Optional, cast

import datasets
from datasets import Dataset as HFDataset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from ..arguments import DataArguments
from ..data import DatasetForCrossEncoder

logger = logging.getLogger(__name__)


class JLaWIRDataset(DatasetForCrossEncoder):
    """
    PyTorch Dataset implementation for JLaWIR that returns anchor, positive, negatives, and label.
    """

    def __init__(
        self,
        args: DataArguments,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        dataset: Optional[HFDataset] = None,
    ):
        """
        Initialize the JLaWIRDataset.

        Args:
            args: Data arguments
            tokenizer: Tokenizer for encoding text
            dataset: Optional pre-loaded dataset
        """
        # Load the datasets manually instead of using parent class's loader
        jlawir_ds_path = args.dataset_options.get(
            "jlawir_ds_path", "/path/to/japanese-ir-qa-large/dataset/train/"
        )
        jlawir_score_ds_path = args.dataset_options.get(
            "jlawir_score_ds_path", "/path/to/japanese-ir-qa-large/hard_negs/dataset/"
        )

        self.random_seed = args.dataset_options.get("random_seed", 42)
        self.train_size = args.dataset_options.get("train_size", None)
        self.test_size = args.dataset_options.get("test_size", 1000)
        self.slice_top_100_k = args.dataset_options.get("slice_top_100_k", 50)
        self.pick_top_100 = args.dataset_options.get("pick_top_100", 0)

        # Train group size is the total number of examples per group
        self.train_group_size = args.train_group_size
        self.total_negs = self.train_group_size - 1
        self.pick_top_1000 = self.total_negs - self.pick_top_100

        self.tokenizer = tokenizer
        self.args = args
        self.max_length = args.max_length

        # Initialize random number generator
        self.rnd = random.Random(self.random_seed)

        # Load datasets
        logger.info(f"Loading JLaWIR base dataset from: {jlawir_ds_path}")
        self.jlawir_ds = cast(HFDataset, datasets.load_from_disk(jlawir_ds_path))

        logger.info(f"Loading JLaWIR score dataset from: {jlawir_score_ds_path}")
        jlawir_ds_score = cast(
            HFDataset, datasets.load_from_disk(jlawir_score_ds_path)["train"]
        )
        jlawir_ds_score = jlawir_ds_score.shuffle(seed=self.random_seed)

        if self.train_size is not None:
            # Select specified number of examples
            total_size = self.train_size + self.test_size
            logger.info(f"Selecting {total_size} examples from {len(jlawir_ds_score)}")
            jlawir_ds_score = jlawir_ds_score.select(range(total_size))

        # Split into train/test
        score_dataset_dict = jlawir_ds_score.train_test_split(
            test_size=self.test_size, seed=self.random_seed
        )

        self.jlawir_score_ds = score_dataset_dict["train"]
        self.test_score_ds = score_dataset_dict["test"]

        # self.dataset = self  # Make dataset point to self for compatibility
        self.total_len = len(self.jlawir_score_ds)

        logger.info(f"Initialized JLaWIRDataset with {self.total_len} examples")

    @property
    def column_names(self) -> list[str]:
        """Return the column names."""
        return ["anchor", "positive", "negatives", "label"]

    def __len__(self):
        """Return the number of items in the dataset."""
        return self.total_len

    def __getitem__(self, idx):
        """Get an item at the specified index."""
        # Get score item
        score_item = self.jlawir_score_ds[idx]
        score_item_columns = score_item.keys()

        # Get positive ID from either original_row_id or pos_id
        if "original_row_id" in score_item_columns:
            pos_id = score_item["original_row_id"]
        elif "pos_id" in score_item_columns:
            pos_id = score_item["pos_id"]
        else:
            raise ValueError("No pos_id or original_row_id found in score_item")

        # Get negative IDs
        top_100_ids = score_item["top_100_ids"][self.slice_top_100_k :]
        top_1000_ids = score_item["top_1000_ids"]

        # Sample negative IDs
        neg_ids = self.rnd.sample(top_100_ids, self.pick_top_100) + self.rnd.sample(
            top_1000_ids, self.pick_top_1000
        )
        assert len(neg_ids) == self.total_negs

        # Get positive data
        pos_data = self.jlawir_ds[pos_id]

        # Get negative texts
        negative_texts = [self.jlawir_ds[i]["text"].strip() for i in neg_ids]

        # Create batch examples
        query = pos_data["question"].strip()
        positive = pos_data["text"].strip()

        batch_data = []
        batch_data.append(self.create_one_example(query, positive, 1.0))
        for neg in negative_texts:
            batch_data.append(self.create_one_example(query, neg, 0.0))

        return batch_data

    def get_test_dataset(self) -> "JLaWIRDataset":
        """Get the test dataset."""
        # Create a copy of current instance with test dataset
        test_dataset = JLaWIRDataset(self.args, self.tokenizer, None)
        test_dataset.jlawir_score_ds = self.test_score_ds
        test_dataset.total_len = len(self.test_score_ds)

        return test_dataset
