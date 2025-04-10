import logging
import math
import random
from typing import List, Optional, cast

import joblib
from datasets import Dataset, load_dataset
from huggingface_hub import hf_hub_download
from transformers import BatchEncoding, PreTrainedTokenizer, PreTrainedTokenizerFast

from ..arguments import DataArguments
from ..data import DatasetForCrossEncoder

logger = logging.getLogger(__name__)

MMARCO_DATASET = "unicamp-dl/mmarco"
HADR_NEGATIVE_SCORE_DS = "hotchpotch/mmarco-hard-negatives-reranker-score"

NEG_SCORE_TH = 0.3
POS_SCORE_TH = 0.7
NEG_FILTER_COUNT = 8


def _map_filter_score(example, neg_score_th: float, pos_score_th: float):
    neg_score: list[float] = example["neg.score"]
    neg_score_filtered_index = [
        i for i, score in enumerate(neg_score) if score < neg_score_th
    ]
    # same pos_score
    pos_score = example["pos.score"]
    pos_score_filtered_index = [
        i for i, score in enumerate(pos_score) if score > pos_score_th
    ]
    return {
        **example,
        "neg.score": [neg_score[i] for i in neg_score_filtered_index],
        "neg": [example["neg"][i] for i in neg_score_filtered_index],
        "pos.score": [pos_score[i] for i in pos_score_filtered_index],
        "pos": [example["pos"][i] for i in pos_score_filtered_index],
    }


def _filter_score(example, net_filter_count: int):
    # neg のカウントがN以上で、pos のカウントが1以上のものを返す
    return len(example["neg"]) >= net_filter_count and len(example["pos"]) >= 1


class MMarcoHardNegatives(DatasetForCrossEncoder):
    def __init__(
        self,
        args: DataArguments,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        dataset: Optional[Dataset] = None,
        seed: int = 42,
    ):
        """
        Initialize the MMarcoHardNegatives dataset.

        Args:
            args: Data arguments
            tokenizer: Tokenizer for encoding text
            dataset: Optional pre-loaded dataset
            seed: Random seed for sampling
        """
        logger.info("Initializing MMarcoHardNegatives dataset")

        self.tokenizer = tokenizer
        self.args = args
        self.max_length = args.max_length

        # Initialize dataset options
        dataset_options = args.dataset_options
        self.binarize_label: bool = dataset_options.get("binarize_label", False)
        self.query_max_len = dataset_options.get("query_max_len", 256)
        self.doc_max_len = dataset_options.get("doc_max_len", 1024)

        # Initialize random number generator
        self.random_seed = seed
        self.rnd = random.Random(self.random_seed)

        # Set train group size
        self.train_group_size = args.train_group_size
        self.total_negs = self.train_group_size - 1

        # Load dataset if not provided
        if dataset is None:
            train_data = args.train_data

            if not isinstance(train_data, dict):
                logger.error("train_data must be a dictionary")
                raise ValueError("train_data must be a dictionary")

            # Extract configuration
            lang = train_data["lang"]
            reranker_name = train_data["reranker"]
            neg_score_th = train_data.get("neg_score_th", NEG_SCORE_TH)
            pos_score_th = train_data.get("pos_score_th", POS_SCORE_TH)
            net_filter_count = train_data.get("net_filter_count", NEG_FILTER_COUNT)
            subset = f"{lang}_{reranker_name}"

            mapping = f"mappings/{lang}_joblib.pkl.gz"

            logger.info(f"Downloading mapping file from Hugging Face Hub: {mapping}")
            mapping_file = hf_hub_download(
                repo_type="dataset", repo_id=HADR_NEGATIVE_SCORE_DS, filename=mapping
            )

            logger.info(f"Loading mapping file: {mapping_file}")
            index_mapping_dict = joblib.load(mapping_file)

            self.query_id_dict = index_mapping_dict["query_id_dict"]
            self.collection_id_dict = index_mapping_dict["collection_id_dict"]

            logger.info(f"Loading queries dataset for language: {lang}")
            self.queries_ds = cast(
                Dataset,
                load_dataset(
                    MMARCO_DATASET,
                    "queries-" + lang,
                    split="train",
                    trust_remote_code=True,
                ),
            )

            logger.info(f"Loading collection dataset for language: {lang}")
            self.collection_ds = cast(
                Dataset,
                load_dataset(
                    MMARCO_DATASET,
                    "collection-" + lang,
                    split="collection",
                    trust_remote_code=True,
                ),
            )

            logger.info(f"Loading hard negatives dataset subset: {subset}")
            ds = load_dataset(HADR_NEGATIVE_SCORE_DS, subset, split="train")
            ds = cast(Dataset, ds)

            logger.info("Filtering and mapping dataset based on scores")
            ds = ds.map(
                _map_filter_score,
                num_proc=11,
                fn_kwargs={"neg_score_th": neg_score_th, "pos_score_th": pos_score_th},
            )

            ds = ds.filter(
                _filter_score,
                num_proc=11,
                fn_kwargs={"net_filter_count": net_filter_count},
            )

            logger.info(f"Filtered dataset size: {len(ds)}")
            self.dataset = ds
        else:
            self.dataset = dataset

        self.total_len = len(self.dataset)
        logger.info(f"Initialized MMarcoHardNegatives with {self.total_len} examples")

        # Initialize parent class with our dataset
        super().__init__(args, tokenizer, self.dataset)

    def get_query_text(self, query_id: int) -> str:
        """Get query text from the queries dataset by ID."""
        idx = self.query_id_dict[query_id]
        return self.queries_ds[idx]["text"][0 : self.query_max_len]

    def get_collection_text(self, doc_id: int) -> str:
        """Get document text from the collection dataset by ID."""
        idx = self.collection_id_dict[doc_id]
        return self.collection_ds[idx]["text"][0 : self.doc_max_len]

    # def create_one_example(
    #     self, query: str, document: str, label: float
    # ) -> BatchEncoding:
    #     """
    #     Create one example by encoding a query-document pair.
    #     """
    #     with warnings.catch_warnings():
    #         warnings.simplefilter("ignore")  # Suppress warnings
    #         item = self.tokenizer.encode_plus(
    #             query,
    #             document,
    #             truncation="only_second",
    #             max_length=self.max_length,
    #             padding=False,
    #         )
    #         item["labels"] = label
    #         return item

    def __len__(self):
        """Return the number of items in the dataset."""
        return self.total_len

    def __getitem__(self, item) -> List[BatchEncoding]:
        """Get an item at the specified index."""
        qid = self.dataset[item]["qid"]
        query = self.get_query_text(qid)

        # Get positive examples
        pos_ids = self.dataset[item]["pos"]
        pos_ids_score = self.dataset[item]["pos.score"]

        # Get negative examples
        neg_ids = self.dataset[item]["neg"]
        neg_ids_score = self.dataset[item]["neg.score"]

        # Sample one positive example randomly
        if len(pos_ids) == 0:
            logger.warning(f"No positive examples found for query ID: {qid}")
            return []

        pos_idx = self.rnd.randrange(len(pos_ids))
        pos_id = pos_ids[pos_idx]
        pos_score = pos_ids_score[pos_idx]

        # Get positive text
        pos_text = self.get_collection_text(pos_id)

        # Check if we have enough negatives
        if len(neg_ids) < self.total_negs:
            logger.warning(
                f"Not enough negative samples for query ID: {qid}, neg_ids len: {len(neg_ids)}"
            )
            # If not enough negative samples, repeat what we have
            num = math.ceil(self.total_negs / len(neg_ids))
            combined = list(zip(neg_ids * num, neg_ids_score * num))
            sampled_combined = combined[: self.total_negs]
            sampled_neg_ids, sampled_neg_scores = zip(*sampled_combined)
        else:
            # Sample the requested number of negatives
            combined = list(zip(neg_ids, neg_ids_score))
            sampled_combined = self.rnd.sample(combined, self.total_negs)
            sampled_neg_ids, sampled_neg_scores = zip(*sampled_combined)

        # Get negative texts
        neg_texts = [self.get_collection_text(neg_id) for neg_id in sampled_neg_ids]

        # Binarize labels if requested
        if self.binarize_label:
            pos_score = 1.0
            sampled_neg_scores = [0.0] * len(sampled_neg_scores)

        # Create batch data in CrossEncoder format
        batch_data = []
        batch_data.append(self.create_one_example(query, pos_text, pos_score))
        for neg_text, neg_score in zip(neg_texts, sampled_neg_scores):
            batch_data.append(self.create_one_example(query, neg_text, neg_score))

        return batch_data

    def get_test_dataset(self) -> "MMarcoHardNegatives":
        """
        Create a test dataset by sampling from the current dataset.
        """
        # Create a simple test split (10% of the data)
        test_size = min(1000, int(self.total_len * 0.1))

        # Randomly select indices for test
        test_indices = self.rnd.sample(range(self.total_len), test_size)

        # Create test dataset
        test_dataset = self.dataset.select(test_indices)

        # Create a new instance with the test dataset
        test_instance = MMarcoHardNegatives(
            self.args, self.tokenizer, test_dataset, self.random_seed
        )

        # Share the resource datasets
        test_instance.queries_ds = self.queries_ds
        test_instance.collection_ds = self.collection_ds
        test_instance.query_id_dict = self.query_id_dict
        test_instance.collection_id_dict = self.collection_id_dict

        return test_instance
