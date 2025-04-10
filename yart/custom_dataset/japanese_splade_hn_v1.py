import logging
import math
import random
from typing import List, Optional, cast

from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import BatchEncoding, PreTrainedTokenizer, PreTrainedTokenizerFast

from ..arguments import DataArguments
from ..data import DatasetForCrossEncoder

logger = logging.getLogger(__name__)

HADR_NEGATIVE_SCORE_DS = "hotchpotch/japanese-splade-v1-hard-negatives"
DS_SPIT = "train"

NEG_SCORE_TH = 0.3
POS_SCORE_TH = 0.7
NEG_FILTER_COUNT = 15
NEG_POS_SCORE_TH = 0.95

TOP_100_SAMPLING_COUNT = 8  # top100 から hard negative としてサンプリングする数


def _map_score_with_hard_positives(
    example,
    neg_score_th: float,
    pos_score_th: float,
    neg_pos_score_th: float,
    hard_positives: bool,
    target_model_name: str = "japanese-splade-base-v1-mmarco-only",
):
    neg_score_top100 = example[
        f"score.bge-reranker-v2-m3.neg_ids.{target_model_name}.top100"
    ]
    neg_score_other100 = example[
        f"score.bge-reranker-v2-m3.neg_ids.{target_model_name}.other100"
    ]
    pos_score = example["score.bge-reranker-v2-m3.pos_ids"]
    neg_score_top100_filtered_index = [
        i for i, score in enumerate(neg_score_top100) if score < neg_score_th
    ]
    neg_score_other100_filtered_index = [
        i for i, score in enumerate(neg_score_other100) if score < neg_score_th
    ]
    pos_score_filtered_index = [
        i for i, score in enumerate(pos_score) if score > pos_score_th
    ]

    # hard positives はまずは、neg.other100 から取得する
    hard_positives_ids = example[f"neg_ids.{target_model_name}.other100"]
    hard_positives_scores = neg_score_other100
    hard_positives_score_filtered_index = [
        i for i, score in enumerate(hard_positives_scores) if score > neg_pos_score_th
    ]

    data = {
        **example,
        "neg.score.top100": [
            neg_score_top100[i] for i in neg_score_top100_filtered_index
        ],
        "neg.top100": [
            example[f"neg_ids.{target_model_name}.top100"][i]
            for i in neg_score_top100_filtered_index
        ],
        "neg.score.other100": [
            neg_score_other100[i] for i in neg_score_other100_filtered_index
        ],
        "neg.other100": [
            example[f"neg_ids.{target_model_name}.other100"][i]
            for i in neg_score_other100_filtered_index
        ],
        "pos.score": [pos_score[i] for i in pos_score_filtered_index],
        "pos": [example["pos_ids"][i] for i in pos_score_filtered_index],
    }
    if hard_positives and len(hard_positives_score_filtered_index) > 0:
        # hard_positives flag がある
        # かつ hard_positives としてふさわしいスコアがある場合、pos.score, pos を neg に置き換える
        data["pos.score"] = [
            hard_positives_scores[i] for i in hard_positives_score_filtered_index
        ]
        data["pos"] = [
            hard_positives_ids[i] for i in hard_positives_score_filtered_index
        ]
    elif len(pos_score_filtered_index) == 0:
        # neg_score_top100 の最大値と、その index を取得
        max_score = max(neg_score_top100)
        max_score_index = neg_score_top100.index(max_score)
        if max_score >= neg_pos_score_th:
            # pos が閾値以上のものがなく、かつ十分なスコアが neg にある場合は、それを pos とする
            data["pos"] = [
                example[f"neg_ids.{target_model_name}.top100"][max_score_index]
            ]
            data["pos.score"] = [max_score]
        elif len(hard_positives_score_filtered_index) > 0:
            # neg_score_top100 にも hard_positives にも該当するスコアがない場合、
            # hard_positives_score_filtered_index から pos を1つランダムに追加する
            hard_positve_index = random.choice(hard_positives_score_filtered_index)
            max_score = hard_positives_scores[hard_positve_index]
            data["pos.score"] = [max_score]
            data["pos"] = [hard_positives_ids[hard_positve_index]]
    return data


def _filter_score(example, net_filter_count: int):
    # neg のカウントがN以上で、pos のカウントが1以上のものを返す
    return (
        len(example["neg.other100"] + example["neg.top100"]) >= net_filter_count
        and len(example["pos"]) >= 1
    )


class JapaneseSpladeHardNegativesV1(DatasetForCrossEncoder):
    def __init__(
        self,
        args: DataArguments,
        tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
        dataset: Optional[Dataset] = None,
        seed: int = 42,
    ):
        """
        Initialize the JapaneseSpladeHardNegativesV1 dataset.

        Args:
            args: Data arguments
            tokenizer: Tokenizer for encoding text
            dataset: Optional pre-loaded dataset
            seed: Random seed for sampling
        """
        # Initialize tokenizer and args before super().__init__
        self.tokenizer = tokenizer
        self.args = args
        self.max_length = args.max_length

        # Get dataset options
        dataset_options = args.dataset_options
        self.binarize_label: bool = dataset_options.get("binarize_label", False)
        self.hard_positives: bool = dataset_options.get("hard_positives", False)
        self.target_model_name: str = dataset_options.get(
            "target_model_name", "japanese-splade-base-v1-mmarco-only"
        )
        self.query_column_name: str = dataset_options.get("query_column_name", "anc")
        self.doc_column_name: str = dataset_options.get("doc_column_name", "text")

        # Train group size is the total number of examples per group
        self.train_group_size = args.train_group_size
        self.total_negs = self.train_group_size - 1

        # Initialize random number generator
        self.random_seed = seed
        self.rnd = random.Random(self.random_seed)

        # Get sampling parameters
        self.top_100_sampling_count = dataset_options.get(
            "top_100_sampling_count", TOP_100_SAMPLING_COUNT
        )

        # Load dataset if not provided
        if dataset is None:
            dataset_name = dataset_options.get("dataset_name", "mmarco")
            logger.info(f"Initializing {dataset_name} hard_negative dataset")
            logger.info(f"binarize_label: {self.binarize_label}")
            logger.info(f"hard_positives: {self.hard_positives}")
            logger.info(f"target_model_name: {self.target_model_name}")
            logger.info(f"query_column_name: {self.query_column_name}")
            logger.info(f"doc_column_name: {self.doc_column_name}")

            query_ds_name = f"{dataset_name}-dataset"
            collection_ds_name = f"{dataset_name}-collection"

            neg_score_th = dataset_options.get("neg_score_th", NEG_SCORE_TH)
            pos_score_th = dataset_options.get("pos_score_th", POS_SCORE_TH)
            neg_pos_score_th = dataset_options.get(
                "neg_pos_thcore_th", NEG_POS_SCORE_TH
            )
            net_filter_count = dataset_options.get("net_filter_count", NEG_FILTER_COUNT)

            logger.info(
                f"Loading dataset from {HADR_NEGATIVE_SCORE_DS}/{query_ds_name}"
            )
            ds = load_dataset(HADR_NEGATIVE_SCORE_DS, query_ds_name, split=DS_SPIT)
            ds = cast(Dataset, ds)

            logger.info(
                f"Loading collection from {HADR_NEGATIVE_SCORE_DS}/{collection_ds_name}"
            )
            self.collection_ds = cast(
                Dataset,
                load_dataset(HADR_NEGATIVE_SCORE_DS, collection_ds_name, split=DS_SPIT),
            )

            logger.info("Mapping scores with hard positives")
            ds = ds.map(
                _map_score_with_hard_positives,
                num_proc=11,  # type: ignore
                fn_kwargs={
                    "neg_score_th": neg_score_th,
                    "pos_score_th": pos_score_th,
                    "neg_pos_score_th": neg_pos_score_th,
                    "hard_positives": self.hard_positives,
                    "target_model_name": self.target_model_name,
                },
            )

            logger.info("Filtering dataset")
            ds = ds.filter(
                _filter_score,
                num_proc=11,
                fn_kwargs={"net_filter_count": net_filter_count},
            )
            logger.info(f"Filtered dataset size: {len(ds)}")

            # Handle dataset augmentation if needed
            aug_factor = dataset_options.get("aug_factor", 1.0)
            n = int(dataset_options.get("n", 0))
            if aug_factor != 1.0:
                n = int(len(ds) * (aug_factor))
                logger.info(
                    f"Augmenting dataset with factor aug_factor={aug_factor}, n={n}"
                )

            if n > len(ds):
                logger.info(f"Expanding dataset from {len(ds)} to {n}")
                ds_expand = []
                c = n // len(ds)
                r = n % len(ds)
                for _ in range(c):
                    ds_expand.append(ds.shuffle(seed=self.random_seed))
                ds_expand.append(ds.shuffle(seed=self.random_seed).select(range(r)))
                ds = concatenate_datasets(ds_expand)
                assert len(ds) == n
            elif n > 0:
                logger.info(f"Shuffling and selecting first {n} samples from dataset")
                ds = ds.shuffle(seed=self.random_seed).select(range(n))

            self.dataset = ds
        else:
            self.dataset = dataset
            # If dataset is provided, we assume collection_ds is already set correctly
            # or will be set by the caller

        self.total_len = len(self.dataset)
        logger.info(
            f"Initialized JapaneseSpladeHardNegativesV1 with {self.total_len} examples"
        )

        # Call parent's init with our dataset
        super().__init__(args, tokenizer, self.dataset)

    def get_collection_text(self, doc_id: int) -> str:
        """Get document text from the collection dataset by ID."""
        text = self.collection_ds[doc_id][self.doc_column_name]
        return text

    def __len__(self):
        """Return the number of items in the dataset."""
        return self.total_len

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

    def __getitem__(self, item) -> List[BatchEncoding]:
        """Get an item at the specified index."""
        query = self.dataset[item][self.query_column_name]

        # Handle positives
        pos_ids = self.dataset[item]["pos"]
        pos_ids_score = self.dataset[item]["pos.score"]

        # Sample one positive example randomly
        if len(pos_ids) == 0:
            logger.warning(f"No positive examples found for query: {query}")
            # This shouldn't happen due to filtering, but just in case
            return []

        pos_idx = self.rnd.randrange(len(pos_ids))
        pos_id = pos_ids[pos_idx]
        pos_score = pos_ids_score[pos_idx]

        # Get positive text
        pos_text = self.get_collection_text(pos_id)

        # Handle negatives from top100 and other100
        neg_ids_top100 = self.dataset[item]["neg.top100"]
        neg_ids_score_top100 = self.dataset[item]["neg.score.top100"]
        neg_ids_other100 = self.dataset[item]["neg.other100"]
        neg_ids_score_other100 = self.dataset[item]["neg.score.other100"]

        # Sample from top100 (prioritize these for hard negatives)
        top_100_count = min(self.top_100_sampling_count, len(neg_ids_top100))
        if top_100_count > 0:
            top100_indices = list(range(len(neg_ids_top100)))
            sampled_indices = self.rnd.sample(top100_indices, top_100_count)
            neg_ids_top100_sampled = [neg_ids_top100[i] for i in sampled_indices]
            neg_ids_score_top100_sampled = [
                neg_ids_score_top100[i] for i in sampled_indices
            ]
        else:
            neg_ids_top100_sampled = []
            neg_ids_score_top100_sampled = []

        # Sample from other100 to fill the remainder
        other_100_count = self.total_negs - len(neg_ids_top100_sampled)
        other_100_count = min(other_100_count, len(neg_ids_other100))

        if other_100_count > 0 and len(neg_ids_other100) > 0:
            other100_indices = list(range(len(neg_ids_other100)))
            sampled_indices = self.rnd.sample(other100_indices, other_100_count)
            neg_ids_other100_sampled = [neg_ids_other100[i] for i in sampled_indices]
            neg_ids_score_other100_sampled = [
                neg_ids_score_other100[i] for i in sampled_indices
            ]
        else:
            neg_ids_other100_sampled = []
            neg_ids_score_other100_sampled = []

        # Combine negatives
        neg_ids = neg_ids_top100_sampled + neg_ids_other100_sampled
        neg_ids_score = neg_ids_score_top100_sampled + neg_ids_score_other100_sampled

        # Check if we have enough negatives
        if len(neg_ids) < self.total_negs:
            logger.warning(
                f"Not enough negative samples for query: {query}, neg_ids len: {len(neg_ids)}"
            )
            # If not enough negative samples, we'll need to repeat what we have
            if len(neg_ids) > 0:
                # Repeat existing negatives
                num = math.ceil(self.total_negs / len(neg_ids))
                neg_ids_extended = neg_ids * num
                neg_ids_score_extended = neg_ids_score * num
                neg_ids = neg_ids_extended[: self.total_negs]
                neg_ids_score = neg_ids_score_extended[: self.total_negs]
            else:
                # No negatives at all, which shouldn't happen due to filtering
                logger.error(f"No negatives available for query: {query}")
                return []

        # Get negative texts
        neg_texts = [self.get_collection_text(neg_id) for neg_id in neg_ids]

        # Binarize labels if requested
        if self.binarize_label:
            pos_score = 1.0
            neg_ids_score = [0.0] * len(neg_ids_score)

        # Create batch inputs in CrossEncoder format
        batch_data = []
        batch_data.append(self.create_one_example(query, pos_text, pos_score))
        for neg_text, neg_score in zip(neg_texts, neg_ids_score):
            batch_data.append(self.create_one_example(query, neg_text, neg_score))

        return batch_data

    def get_test_dataset(self) -> "JapaneseSpladeHardNegativesV1":
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
        test_instance = JapaneseSpladeHardNegativesV1(
            self.args, self.tokenizer, test_dataset, self.random_seed
        )
        test_instance.collection_ds = self.collection_ds  # Share the collection dataset
        return test_instance
