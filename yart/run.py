"""
YART: Your Another Reranker Trainer
Main script for training cross-encoder rankers.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))  # noqa: F821

import wandb
from yart.data import GroupCollator, create_dateset_from_args
from yart.losses import get_loss_fn
from yart.trainer import CrossEncoderModel, RankerTrainer
from yart.utils import (
    load_config_from_yaml,
    load_model_and_tokenizer,
    parse_config_to_args,
    remove_checkpoints,
    seed_everything,
    setup_logging,
    setup_wandb,
)

logger = logging.getLogger(__name__)
# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a cross-encoder ranker")
    parser.add_argument(
        "--config",
        "-c",  # Added short form -c
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def main() -> None:
    # Parse command line arguments
    args = parse_args()

    # Setup logging
    setup_logging(args.verbose)
    # Load and parse configuration
    config = load_config_from_yaml(args.config)
    model_args, data_args, training_args, run_args = parse_config_to_args(config)

    # Override debug mode if specified
    if args.debug:
        run_args.debug = True

    # Setup output directory
    if run_args.debug:
        run_args.output_prefix = "debug_" + run_args.output_prefix

    # Update output dir with prefix if needed
    if (
        run_args.output_prefix
        and training_args.output_dir
        and not training_args.output_dir.startswith(run_args.output_prefix)
    ):
        training_args.output_dir = os.path.join(
            os.path.dirname(training_args.output_dir),
            f"{run_args.output_prefix}{os.path.basename(training_args.output_dir)}",
        )
    elif run_args.output_prefix and not training_args.output_dir:
        training_args.output_dir = run_args.output_prefix

    # If output_dir is not specified, create it based on config filename
    if not training_args.output_dir:
        # Extract experiment name from config filename (e.g., 'exp303' from 'exp303.yaml')
        config_basename = os.path.basename(args.config)
        experiment_name = os.path.splitext(config_basename)[0]  # Remove file extension
        training_args.output_dir = os.path.join("./outputs", experiment_name)
        # log message
        logger.info(
            f"Output directory not specified. Using -> {training_args.output_dir}"
        )

    output_dir = training_args.output_dir
    logger.info(f"Output directory: {output_dir}")

    # Set seed for reproducibility
    seed_everything(training_args.seed)

    training_args.remove_unused_columns = False  # override

    # Setup wandb if not in debug mode
    if (
        not run_args.debug
        and training_args.report_to
        and "wandb" in training_args.report_to
    ):
        setup_wandb("cross-encoder-reranker")
        wandb.init(
            project="cross-encoder-reranker",
            name=os.path.basename(output_dir),
        )
    else:
        training_args.report_to = ["none"]

    # Log arguments
    logger.info(f"Model arguments: {model_args}")
    logger.info(f"Data arguments: {data_args}")
    logger.info(f"Training arguments: {training_args}")

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_args, training_args)

    # Create dataset
    train_dataset = create_dateset_from_args(
        data_args,
        tokenizer,
    )

    # Create loss function
    loss_fn = get_loss_fn(training_args.loss_name)

    # Create cross-encoder model wrapper
    cross_encoder = CrossEncoderModel(
        model,
        training_args.train_group_size,
        training_args.per_device_train_batch_size,
        loss_fn=loss_fn,
    )

    # Create data collator
    data_collator = GroupCollator(tokenizer)

    # Create trainer
    trainer = RankerTrainer(
        model=cross_encoder,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Train the model
    logger.info("Starting training")
    trainer.train()

    # Save the final model
    logger.info(f"Saving final model to {training_args.output_dir}")
    trainer.save_model()

    # Remove checkpoints if requested
    if run_args.remove_checkpoints:
        logger.info("Removing checkpoints")
        remove_checkpoints(output_dir)

    logger.info("Training complete")

    # Run evaluation if available
    if not run_args.debug and os.path.exists("eval/all.py"):
        logger.info("Running evaluation")
        import subprocess

        eval_script = os.path.join(os.getcwd(), "eval/all.py")
        eval_cmd = ["python", eval_script, "-m", training_args.output_dir]

        if run_args.debug:
            eval_cmd.append("-d")

        logger.info(f"Running evaluation command: {' '.join(eval_cmd)}")
        subprocess.run(eval_cmd)


if __name__ == "__main__":
    main()
