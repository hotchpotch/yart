import logging
import math
import random
from functools import cache
from typing import Optional, cast

import numpy as np
from datasets import Dataset, concatenate_datasets, load_dataset
from huggingface_hub import HfApi
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from ..arguments import DataArguments
from ..data import DatasetForCrossEncoder

logger = logging.getLogger(__name__)

HPPRC_EMB_DS = "hpprc/emb"
SCORE_DS = "hotchpotch/hpprc_emb-scores"


@cache
def get_dataset_subsets(dataset_name):
    logger.info(f"Fetching dataset subsets for dataset: {dataset_name}")
    api = HfApi()
    dataset_info = api.dataset_info(dataset_name)

    if card_data := dataset_info.card_data:
        if "configs" in card_data:
            subsets = [config["config_name"] for config in card_data["configs"]]
            logger.info(f"Found configs: {subsets}")
            return subsets
        elif "dataset_info" in card_data:
            subsets = [info["config_name"] for info in card_data["dataset_info"]]
            logger.info(f"Found dataset_info configs: {subsets}")
            return subsets
    else:
        logger.warning(f"No subsets found for dataset: {dataset_name}")
        return []


def get_datasets(target_name: str):
    logger.info(f"Retrieving datasets for target name: {target_name}")
    subsets = get_dataset_subsets(SCORE_DS)
    target_subsets = [subset for subset in subsets if subset.startswith(target_name)]  # type: ignore
    if not target_subsets:
        logger.error(f"Subset not found: {target_name}")
        raise ValueError(f"Subset not found: {target_name}")
    target_subset = target_subsets[0]
    target_base_name, revision = target_subset.rsplit("-dataset__", 1)
    logger.info(
        f"Loading score dataset: {SCORE_DS}, subset: {target_subset}, revision: {revision}"
    )
    score_ds = load_dataset(
        SCORE_DS,
        name=target_subset,
        split="train",
    )
    if target_name.startswith("quiz-") or target_name.startswith("mkqa"):
        collection_name = "qa-collection"
    else:
        collection_name = f"{target_base_name}-collection"
    logger.info(
        f"Loading embedding dataset: {HPPRC_EMB_DS}, collection: {collection_name}, revision: {revision}"
    )
    hpprc_emb_ds = load_dataset(
        HPPRC_EMB_DS,
        name=collection_name,
        split="train",
        revision=revision,
    )
    return hpprc_emb_ds, score_ds


TARGET_POS_ID_KEYS = [
    "pos_ids",
    "pos_ids.original",
    "pos_ids.me5-large",
    "pos_ids.bm25",
]
TARGET_NEG_ID_KEYS = [
    "neg_ids",
    "neg_ids.original",
    "neg_ids.me5-large",
    "neg_ids.bm25",
]


def map_data(
    example,
    target_score_keys: list[str] = ["ruri-reranker-large", "bge-reranker-v2-m3"],
    pos_id_score_threshold: float = 0.7,
    neg_id_score_threshold: float = 0.3,
):
    target_id_keys = TARGET_POS_ID_KEYS + TARGET_NEG_ID_KEYS
    # 対象の target_score の平均値を取得
    target_score_dict = {}
    for id_key in target_id_keys:
        scores_key_values = []
        for score_key in target_score_keys:
            full_score_key = f"score.{score_key}.{id_key}"
            scores = example.get(full_score_key, [])
            if len(scores) > 0:
                scores_key_values.append(np.array(scores))
        if len(scores_key_values) > 0:
            if len(scores_key_values) != len(target_score_keys):
                logger.error(
                    f"len(scores_key_values) != len(target_score_keys): {len(scores_key_values)} != {len(target_score_keys)}"
                )
                raise ValueError(
                    f"len(scores_key_values) != len(target_score_keys): {len(scores_key_values)} != {len(target_score_keys)}"
                )
            mean_scores = np.array(scores_key_values).T.mean(axis=1)
            target_score_dict[id_key] = mean_scores.tolist()

    filtered_target_ids_dict = {}
    # 閾値でフィルタリング
    for id_key in target_score_dict.keys():
        target_score_ids = example[id_key]
        target_scores = target_score_dict[id_key]
        if "pos_ids" in id_key:
            filtered_target_scores_indexes = [
                i
                for i, score in enumerate(target_scores)
                if score >= pos_id_score_threshold
            ]
        else:
            filtered_target_scores_indexes = [
                i
                for i, score in enumerate(target_scores)
                if score <= neg_id_score_threshold
            ]
        filtered_target_ids = [
            target_score_ids[i] for i in filtered_target_scores_indexes
        ]
        filtered_target_ids_dict[id_key] = filtered_target_ids
        target_score_dict[id_key] = [
            target_scores[i] for i in filtered_target_scores_indexes
        ]
    result_pos_ids = []
    result_pos_ids_score = []
    result_neg_ids = []
    result_neg_ids_score = []
    for id_key in target_score_dict.keys():
        # pos_ids, neg_ids ともに重複IDがあるので、その場合は追加しない
        if "pos_ids" in id_key and id_key not in result_pos_ids:
            result_pos_ids += filtered_target_ids_dict[id_key]
            result_pos_ids_score += target_score_dict[id_key]
        elif "neg_ids" in id_key and id_key not in result_neg_ids and id_key:
            result_neg_ids += filtered_target_ids_dict[id_key]
            result_neg_ids_score += target_score_dict[id_key]
    # ログメッセージに置き換え
    logger.debug(f"pos_ids: {result_pos_ids}")
    logger.debug(f"neg_ids: {result_neg_ids}")
    logger.debug(f"result_pos_ids_score: {result_pos_ids_score}")
    logger.debug(f"result_neg_ids_score: {result_neg_ids_score}")
    return {
        "anc": example["anc"],
        "pos_ids": result_pos_ids,
        "pos_ids.score": result_pos_ids_score,
        "neg_ids": result_neg_ids,
        "neg_ids.score": result_neg_ids_score,
    }


def filter_data(example):
    neg_ids = example["neg_ids"]
    if len(neg_ids) < 8:
        return False
    pos_ids = example["pos_ids"]
    if len(pos_ids) == 0:
        return False
    return True


class HpprcEmbScoresDataset(DatasetForCrossEncoder):
    def __init__(
        self,
        args: DataArguments,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        dataset: Optional[Dataset] = None,
        seed: int = 42,
    ):
        """
        Initialize the HpprcEmbScoresDataset.

        Args:
            args: Data arguments
            tokenizer: Tokenizer for encoding text
            dataset: Optional pre-loaded dataset
            seed: Random seed for sampling
        """
        logger.info("Initializing HpprcEmbScoresDataset")
        self.tokenizer = tokenizer
        self.args = args
        self.max_length = args.max_length

        # Get dataset options
        dataset_options = args.dataset_options
        self.binarize_label: bool = dataset_options.get("binarize_label", False)

        # Train group size is the total number of examples per group (1 positive + N negatives)
        self.train_group_size = args.train_group_size
        self.total_negs = self.train_group_size - 1

        # Initialize random number generator
        self.random_seed = seed
        self.rnd = random.Random(self.random_seed)

        # Load datasets if not provided
        if dataset is None:
            train_data = args.train_data
            # train data must be a list
            if not isinstance(train_data, list):
                logger.error("train_data must be a list")
                raise ValueError("train_data must be a list")

            all_ds = []
            self.target_emb_ds = {}

            for target in train_data:
                if not isinstance(target, dict):
                    logger.error("train_data must be a list of dictionaries")
                    raise ValueError("train_data must be a list of dictionaries")

                subset = target["subset"]
                logger.info(f"Processing subset: {subset}")
                n = target.get("n", None)
                emb_ds, score_ds = get_datasets(subset)
                score_ds = cast(Dataset, score_ds)

                logger.info(f"Mapping data for subset: {subset}")
                score_ds = score_ds.map(
                    map_data,
                    num_proc=11,
                    remove_columns=score_ds.column_names,
                    fn_kwargs={
                        "target_score_keys": target.get(
                            "target_score_keys",
                            ["ruri-reranker-large", "bge-reranker-v2-m3"],
                        )
                    },
                )

                self.target_emb_ds[subset] = emb_ds
                aug_factor = target.get("aug_factor", 1.0)

                # Handle dataset augmentation
                if aug_factor != 1.0:
                    if n is not None:
                        logger.warning(
                            f"aug_factor is ignored because n is specified, skipping aug_factor args for subset: {subset}"
                        )
                    else:
                        n = int(len(score_ds) * aug_factor)
                        logger.info(
                            f"Augmenting dataset: {subset} with aug_factor: {aug_factor}"
                        )

                # Handle dataset size adjustment
                if n is not None:
                    if n > len(score_ds):
                        logger.info(
                            f"Expanding dataset: {subset} from {len(score_ds)} to {n}"
                        )
                        score_ds_expand = []
                        c = n // len(score_ds)
                        r = n % len(score_ds)
                        for _ in range(c):
                            score_ds_expand.append(score_ds.shuffle(seed=seed))
                        score_ds_expand.append(
                            score_ds.shuffle(seed=seed).select(range(r))
                        )
                        score_ds = concatenate_datasets(score_ds_expand)
                        assert len(score_ds) == n
                    else:
                        logger.info(
                            f"Shuffling and selecting first {n} samples from dataset: {subset}"
                        )
                        score_ds = score_ds.shuffle(seed=seed).select(range(n))

                # Filter dataset
                before_filter_len = len(score_ds)
                logger.info(
                    f"Filtering dataset: {subset}, original size: {before_filter_len}"
                )
                score_ds = score_ds.filter(filter_data, num_proc=11)
                after_filter_len = len(score_ds)
                logger.info(
                    f"Filtered dataset size: {subset}, before: {before_filter_len}, after: {after_filter_len}, ratio: {after_filter_len / before_filter_len:.2f}"
                )

                # Add subset column
                subsets_column = [subset] * len(score_ds)
                score_ds = score_ds.add_column("subset", subsets_column)  # type: ignore
                all_ds.append(score_ds)
                logger.info(f"Loaded subset: {subset}, size: {len(score_ds)}")

            # Concatenate all datasets
            self.dataset = concatenate_datasets(all_ds)
            logger.info(f"Total concatenated dataset size: {len(self.dataset)}")
        else:
            self.dataset = dataset

        self.total_len = len(self.dataset)
        logger.info(f"Initialized HpprcEmbScoresDataset with {self.total_len} examples")

    def get_text_by_subset(self, subset: str, idx: int, max_len: int = 1024) -> str:
        """Get text from the embedding dataset by subset and index."""
        emb_ds = self.target_emb_ds[subset]
        row = emb_ds[idx]
        text = row["text"]
        title = row.get("title", None)
        if title:
            text = title + " " + text
        return text[0:max_len]

    def __len__(self):
        """Return the number of items in the dataset."""
        return self.total_len

    def __getitem__(self, item) -> list:
        """Get an item at the specified index following JLaWIR pattern."""
        subset = self.dataset[item]["subset"]
        query = self.dataset[item]["anc"]
        pos_ids = self.dataset[item]["pos_ids"]
        pos_ids_score = self.dataset[item]["pos_ids.score"]
        neg_ids = self.dataset[item]["neg_ids"]
        neg_ids_score = self.dataset[item]["neg_ids.score"]

        # Binarize labels if requested
        if self.binarize_label:
            pos_ids_score = [1.0] * len(pos_ids_score)
            neg_ids_score = [0.0] * len(neg_ids_score)

        # Sample one positive example
        if len(pos_ids) == 0:
            logger.warning(f"No positive examples found for query: {query}")
            # This shouldn't happen due to filtering, but just in case
            return []

        # Sample one positive randomly
        pos_idx = self.rnd.randrange(len(pos_ids))
        pos_id = pos_ids[pos_idx]
        pos_score = pos_ids_score[pos_idx]

        # Get positive text
        pos_text = self.get_text_by_subset(subset, pos_id)

        # Check if we have enough negatives
        if len(neg_ids) < self.total_negs:
            logger.warning(
                f"Not enough negative samples for query: {query}, neg_ids len: {len(neg_ids)}"
            )
            # If not enough negative samples, repeat what we have
            num = math.ceil(self.total_negs / len(neg_ids))
            sampled_neg_ids = self.rnd.sample(neg_ids * num, self.total_negs)
            sampled_neg_scores = []
            for neg_id in sampled_neg_ids:
                # Find the original index to get the score
                orig_idx = neg_ids.index(neg_id)
                sampled_neg_scores.append(neg_ids_score[orig_idx])
        else:
            # Sample the requested number of negatives
            combined = list(zip(neg_ids, neg_ids_score))
            sampled_combined = self.rnd.sample(combined, self.total_negs)
            sampled_neg_ids, sampled_neg_scores = zip(*sampled_combined)

        # Get negative texts
        neg_texts = [
            self.get_text_by_subset(subset, neg_id) for neg_id in sampled_neg_ids
        ]

        # Create batch data in JLaWIR format
        batch_data = []
        batch_data.append(self.create_one_example(query, pos_text, pos_score))
        for neg_text, neg_score in zip(neg_texts, sampled_neg_scores):
            batch_data.append(self.create_one_example(query, neg_text, neg_score))

        return batch_data

    def get_test_dataset(self) -> "HpprcEmbScoresDataset":
        """
        Create a test dataset by sampling from the current dataset.

        This is a simplified version of JLaWIR's test dataset creation.
        You might want to implement a more sophisticated version based on your needs.
        """
        # Create a simple test split (10% of the data)
        test_size = min(1000, int(self.total_len * 0.1))

        # Randomly select indices for test
        test_indices = self.rnd.sample(range(self.total_len), test_size)

        # Create test dataset
        test_dataset = self.dataset.select(test_indices)

        # Create a new instance with the test dataset
        test_instance = HpprcEmbScoresDataset(
            self.args, self.tokenizer, test_dataset, self.random_seed
        )
        return test_instance
