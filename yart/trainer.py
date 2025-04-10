"""
YART: Your Another Reranker Trainer
Trainer class for cross-encoder models.
"""

import logging
import os

import torch
import torch.nn as nn
from transformers import Trainer

# from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from .arguments import RankerTrainingArguments
from .log_metrics import LogMetrics

# from .losses import get_loss_fn

logger = logging.getLogger(__name__)


class CrossEncoderModel(nn.Module):
    """
    Cross-encoder model wrapper with support for multi-sample loss functions.
    """

    def __init__(
        self,
        hf_model: nn.Module,
        train_group_size: int,
        per_device_batch_size: int,
        loss_fn: nn.Module = nn.CrossEntropyLoss(reduction="mean"),
    ):
        super().__init__()
        self.hf_model = hf_model
        self.config = getattr(hf_model, "config", None)
        self.train_group_size = train_group_size
        self.per_device_batch_size = per_device_batch_size
        self.loss_fn = loss_fn

    def gradient_checkpointing_enable(self, **kwargs):
        """Enable gradient checkpointing."""
        if hasattr(self.hf_model, "gradient_checkpointing_enable"):
            self.hf_model.gradient_checkpointing_enable(**kwargs)  # type: ignore

    def forward(self, batch):
        """
        Forward pass of the model.

        Args:
            batch: Batch of inputs

        Returns:
            Model outputs with loss
        """
        labels = batch.pop("labels")
        outputs = self.hf_model(**batch, return_dict=True)
        logits = outputs.logits

        if self.training:
            try:
                # Reshape for group-wise loss computation
                scores = logits.view(
                    self.per_device_batch_size,
                    self.train_group_size,
                )
                labels = labels.view(self.per_device_batch_size, self.train_group_size)
                loss = self.loss_fn(scores, labels)
            except (RuntimeError, ValueError) as e:
                # Handle uneven batch sizes (last batch may be smaller)
                logger.warning(f"Reshaping error: {e}, trying flexible reshaping")
                batch_size = logits.size(0) // self.train_group_size
                scores = logits.view(batch_size, self.train_group_size)
                labels = labels.view(batch_size, self.train_group_size)
                loss = self.loss_fn(scores, labels)

            # Add loss to outputs
            outputs.loss = loss
        else:
            loss = None

        return loss, outputs

    def save_pretrained(self, output_dir: str):
        """
        Save model to output directory.

        Args:
            output_dir: Output directory
        """
        if hasattr(self.hf_model, "save_pretrained"):
            state_dict = self.hf_model.state_dict()
            state_dict = type(state_dict)(
                {k: v.clone().cpu() for k, v in state_dict.items()}
            )
            self.hf_model.save_pretrained(output_dir, state_dict=state_dict)  # type: ignore
        else:
            # Fallback to regular torch save
            torch.save(
                self.hf_model.state_dict(),
                os.path.join(output_dir, "pytorch_model.bin"),
            )


class RankerTrainer(Trainer):
    """
    Custom trainer for ranking models.
    """

    def __init__(self, args: RankerTrainingArguments, **kwargs):
        super().__init__(args=args, **kwargs)
        self.args: RankerTrainingArguments = args
        self.log_metrics = LogMetrics()

    def compute_loss(
        self, model: CrossEncoderModel, inputs, return_outputs=False, **kwargs
    ):
        """
        Compute loss from model outputs.

        Args:
            model: Model to compute loss for
            inputs: Model inputs
            return_outputs: Whether to return outputs along with loss

        Returns:
            Loss value or tuple of (loss, outputs)
        """
        loss, outputs = model.forward(inputs)

        # Log additional metrics if available
        # if hasattr(outputs, "logits"):
        #     self.log_metrics.add("logits_mean", outputs.logits.mean())
        #     self.log_metrics.add("logits_std", outputs.logits.std())

        # print(f"Loss: {loss}")

        self.log_metrics.add("loss", loss)  # type: ignore
        return (loss, outputs) if return_outputs else loss

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        logs["step"] = self.state.global_step
        if self.state.epoch is not None:
            logs["epoch"] = round(self.state.epoch, 2)

        current_metrics = self.log_metrics.mean()
        self.log_metrics.clear()
        logs.update(current_metrics)

        output = {**logs, "step": self.state.global_step}
        self.state.log_history.append(output)
        self.control = self.callback_handler.on_log(
            self.args, self.state, self.control, logs
        )

    def _save(self, output_dir: str, state_dict=None):
        """
        Save model, tokenizer, and training arguments.

        Args:
            output_dir: Directory to save to
            state_dict: Model state dictionary
        """
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        # Save model
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(output_dir)  # type: ignore
        else:
            logger.warning(
                f"Model {self.model.__class__.__name__} does not support save_pretrained, "
                "saving model state dict instead"
            )
            torch.save(
                self.model.state_dict(), os.path.join(output_dir, "pytorch_model.bin")
            )

        # Save tokenizer
        if self.tokenizer is not None and self.is_world_process_zero():
            self.tokenizer.save_pretrained(output_dir)

        # Save training arguments
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
