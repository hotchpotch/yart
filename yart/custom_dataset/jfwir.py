"""
YART: Your Another Reranker Trainer
Japanese Large QA Dataset for Information Retrieval (jfwir) Dataset implementation.
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

NUM_PROC = 15


class JFWIRDataset(DatasetForCrossEncoder):
    def __init__(
        self,
        args: DataArguments,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        dataset: Optional[HFDataset] = None,
    ):
        """
        Initialize the jfwirDataset.

        Args:
            args: Data arguments
            tokenizer: Tokenizer for encoding text
            dataset: Optional pre-loaded dataset
        """
        # Load the datasets manually instead of using parent class's loader
        jfwir_ds_path = args.dataset_options.get(
            "jfwir_ds_path", "/path/to/japanese-ir-qa-large/dataset/train/"
        )
        jfwir_score_ds_path = args.dataset_options.get(
            "jfwir_score_ds_path", "/path/to/japanese-ir-qa-large/hard_negs/dataset/"
        )

        self.random_seed = args.dataset_options.get("random_seed", 42)
        self.train_size = args.dataset_options.get("train_size", None)
        self.test_size = args.dataset_options.get("test_size", 1000)
        self.neg_th = args.dataset_options.get(
            "neg_th", None
        )  # このスコア以下を negative とする
        self.pos_th = args.dataset_options.get(
            "pos_th", None
        )  # このスコア以上を positive とする

        # Train group size is the total number of examples per group
        self.train_group_size = args.train_group_size
        self.total_negs = self.train_group_size - 1

        self.tokenizer = tokenizer
        self.args = args
        self.max_length = args.max_length

        # Initialize random number generator
        self.rnd = random.Random(self.random_seed)

        # Load datasets
        logger.info(f"Loading jfwir base dataset from: {jfwir_ds_path}")
        self.jfwir_ds = cast(HFDataset, datasets.load_from_disk(jfwir_ds_path))
        # もし ['train'] が存在する場合はそれを使用
        if "train" in self.jfwir_ds:
            self.jfwir_ds = self.jfwir_ds["train"]
            # cast
            self.jfwir_ds = cast(HFDataset, self.jfwir_ds)

        logger.info(f"Loading jfwir score dataset from: {jfwir_score_ds_path}")
        jfwir_ds_score = cast(HFDataset, datasets.load_from_disk(jfwir_score_ds_path))
        # logger.info("Loaded jfwir_ds_score", jfwir_ds_score)
        # # もし ['train'] が存在する場合はそれを使用
        # if "train" in jfwir_ds_score:
        #     jfwir_ds_score = jfwir_ds_score["train"]
        #     # cast
        #     jfwir_ds_score = cast(HFDataset, jfwir_ds_score)

        # shuffle
        jfwir_ds_score = jfwir_ds_score.shuffle(seed=self.random_seed)

        if self.train_size is not None:
            # Select specified number of examples
            total_size = self.train_size + self.test_size
            logger.info(f"Selecting {total_size} examples from {len(jfwir_ds_score)}")
            jfwir_ds_score = jfwir_ds_score.select(range(total_size))

        print(f"Loaded jfwir_ds: {len(self.jfwir_ds)} examples")
        jfwir_ds_score = self.filter_scores(jfwir_ds_score)
        print(f"Filtered jfwir_ds_score: {len(jfwir_ds_score)} examples")

        # Split into train/test
        score_dataset_dict = jfwir_ds_score.train_test_split(
            test_size=self.test_size, seed=self.random_seed
        )

        self.jfwir_score_ds = score_dataset_dict["train"]
        self.test_score_ds = score_dataset_dict["test"]

        # self.dataset = self  # Make dataset point to self for compatibility
        self.total_len = len(self.jfwir_score_ds)

        logger.info(f"Initialized jfwirDataset with {self.total_len} examples")

    def filter_scores(self, score_dataset: HFDataset) -> HFDataset:
        """
        Filter the score dataset to only include items with valid scores.

        Args:
            score_dataset: The dataset containing scores

        Returns:
            Filtered dataset with valid scores
        """
        # pos_id:int, pos_score:float, neg_ids:List[int], neg_scores:List[float] をもとにフィルターする
        pos_th = self.pos_th
        neg_th = self.neg_th
        if pos_th is None or neg_th is None:
            raise ValueError("pos_th and neg_th must be set in dataset options")

        # まずは neg_th より小さいスコアを持つものを map する
        def neg_map_fn(example):
            neg_scores = example.get("neg_scores", [])
            neg_ids = example.get("neg_ids", [])
            # neg_idsもmapするため、zipして処理する
            res_neg_scores = []
            res_neg_ids = []
            for neg_id, neg_score in zip(neg_ids, neg_scores):
                if neg_score <= neg_th:
                    # スコアが neg_th 以下のものだけを残す
                    res_neg_scores.append(neg_score)
                    res_neg_ids.append(neg_id)
            return {
                "neg_ids": res_neg_ids,
                "neg_scores": res_neg_scores,
            }

        neg_size_min = self.total_negs

        # neg_size_min より、サイズが小さかったら、filter する
        def filter_fn(example):
            pos_score = example.get("pos_score", 0.0)
            neg_scores = example.get("neg_scores", [])
            if len(neg_scores) < neg_size_min:
                # neg_scoresのサイズが neg_size_min より小さい
                return False
            if pos_score < self.pos_th:
                # スコアが posに満たない
                return False
            return True

        # Apply the map and filter functions
        score_dataset = score_dataset.map(neg_map_fn, num_proc=NUM_PROC)
        score_dataset = score_dataset.filter(filter_fn, num_proc=NUM_PROC)
        return score_dataset

    @property
    def column_names(self) -> list[str]:
        """Return the column names."""
        return ["query", "text"]

    def __len__(self):
        """Return the number of items in the dataset."""
        return self.total_len

    def __getitem__(self, idx):
        """Get an item at the specified index."""
        # Get score item
        score_item = self.jfwir_score_ds[idx]
        score_item_columns = score_item.keys()

        # Get positive ID from either original_row_id or pos_id
        if "original_row_id" in score_item_columns:
            pos_id = score_item["original_row_id"]
        elif "pos_id" in score_item_columns:
            pos_id = score_item["pos_id"]
        else:
            raise ValueError("No pos_id or original_row_id found in score_item")

        neg_ids = self.rnd.sample(score_item["neg_ids"], self.total_negs)
        assert len(neg_ids) == self.total_negs

        # Get positive data
        pos_data = self.jfwir_ds[pos_id]

        # Get negative texts
        negative_texts = [self.jfwir_ds[i]["text"].strip() for i in neg_ids]

        # Create batch examples
        query = pos_data["query"].strip()
        positive = pos_data["text"].strip()

        batch_data = []
        batch_data.append(self.create_one_example(query, positive, 1.0))
        for neg in negative_texts:
            batch_data.append(self.create_one_example(query, neg, 0.0))

        return batch_data

    def get_test_dataset(self) -> "JFWIRDataset":
        """Get the test dataset."""
        # Create a copy of current instance with test dataset
        test_dataset = JFWIRDataset(self.args, self.tokenizer, None)
        test_dataset.jfwir_score_ds = self.test_score_ds
        test_dataset.total_len = len(self.test_score_ds)

        return test_dataset
