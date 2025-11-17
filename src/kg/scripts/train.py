import argparse
import json
from pathlib import Path

import torch
from datasets import concatenate_datasets, load_dataset
from transformers import Trainer, TrainingArguments

import wandb
from kg.train.callbacks import LoggingCallback
from kg.train.model_factory import model_factory
from kg.utils.constants import (
    DATA_DIR,
    LOGGER,
    MODEL_TO_HFID,
    TIMESTAMP,
    TRAINED_MODELS_DIR,
    TRAINING_CONFIG_DIR,
)
from kg.utils.utils_io import load_training_config, namespace_to_dict


def create_dataset(cfg, preprocess_data, val_split=0.2):
    dataset_dir = (
        DATA_DIR
        / cfg.data_options.dataset_name
        / cfg.data_options.dataset_dir
        / "dataset"
    )

    if cfg.data_options.dataset_type == "A2B":
        data_files = {
            "train": [str(f) for f in dataset_dir.glob("*.jsonl") if "A2B" in f.name]
        }
        LOGGER.info(f"Loading custom dataset: {data_files}...")
        dataset = load_dataset("json", data_files=data_files)
        dataset = dataset.map(preprocess_data, batched=True)
    elif cfg.data_options.dataset_type == "B2A":
        data_files = {
            "train": [str(f) for f in dataset_dir.glob("*.jsonl") if "B2A" in f.name]
        }
        LOGGER.info(f"Loading custom dataset: {data_files}...")
        dataset = load_dataset("json", data_files=data_files)
        dataset = dataset.map(preprocess_data, batched=True)
    elif cfg.data_options.dataset_type == "all":
        LOGGER.info(f"Loading custom dataset: {dataset_dir}...")
        dataset = load_dataset("json", data_dir=dataset_dir)
        dataset = dataset.map(preprocess_data, batched=True)

    # Pick up val_split from config if present, otherwise use default
    data_options = getattr(cfg, "data_options", None)
    if data_options is not None and hasattr(data_options, "val_split"):
        val_split = data_options.val_split
    LOGGER.info(f"Using val_split: {val_split}")

    dataset = dataset["train"].train_test_split(test_size=val_split)
    dataset["validation"] = dataset.pop("test")

    return dataset


def train(cfg):
    smoke_test = cfg.smoke_test
    freeze_embeddings = cfg.training.freeze_embeddings
    freeze_unembeddings = cfg.training.freeze_unembeddings

    dataset_name = cfg.data_options.dataset_name
    n_supplemental_train_examples = cfg.data_options.n_supplemental_train_examples

    model = cfg.model
    hf_id = MODEL_TO_HFID[model]
    model, tokenizer, preprocess_data = model_factory(hf_id)

    if cfg.model_checkpoint_parent is None:
        run_dir = cfg.data_options.dataset_type + "_" + TIMESTAMP
        run_dir = run_dir if not smoke_test else f"{run_dir}_smoke_test"
        output_dir = TRAINED_MODELS_DIR / hf_id / dataset_name / run_dir
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = cfg.model_checkpoint_parent

    # Save the training configuration as JSON
    config_path = Path(output_dir) / "training_config.json"
    with open(config_path, "w") as f:
        config_dict = namespace_to_dict(cfg)
        json.dump(config_dict, f, indent=2)
    LOGGER.info(f"Saved training configuration to {config_path}")

    ### WANDB & LOGGING ###
    wandb.init(
        project="kg",
        name=str(output_dir),
    )

    dataset = create_dataset(cfg, preprocess_data)

    ### ADD OPENWEBTEXT & IMDB TO TRAIN SET ###
    if n_supplemental_train_examples > 0:
        n_openwebtext = int(
            (1 - cfg.data_options.imdb_split) * n_supplemental_train_examples
        )
        n_imdb = n_supplemental_train_examples - n_openwebtext

        LOGGER.info(
            f"Loading {n_openwebtext} openwebtext and {n_imdb} IMDB examples..."
        )

        openwebtext = load_dataset("openwebtext", trust_remote_code=True)["train"]
        openwebtext = openwebtext.select(range(n_openwebtext))
        openwebtext = openwebtext.map(lambda x: {**x, "entity": ""})
        openwebtext = openwebtext.map(preprocess_data, batched=True)

        imdb = load_dataset("imdb", trust_remote_code=True)["train"]
        imdb = imdb.select(range(n_imdb))
        imdb = imdb.remove_columns(
            "label"
        )  # Remove label column, doing language modeling
        imdb = imdb.map(lambda x: {"text": x["text"], "entity": ""})
        imdb = imdb.map(preprocess_data, batched=True)

        dataset["train"] = concatenate_datasets([dataset["train"], openwebtext, imdb])

    smoke_test_limit = (
        min(20, len(dataset["train"]), len(dataset["validation"]))
        if smoke_test
        else None
    )
    dataset["train"] = (
        dataset["train"]
        if not smoke_test
        else dataset["train"].select(range(smoke_test_limit))
    )
    dataset["validation"] = (
        dataset["validation"]
        if not smoke_test
        else dataset["validation"].select(range(smoke_test_limit))
    )

    num_training_examples = len(dataset["train"])
    train_batch_size = cfg.training.per_device_train_batch_size
    steps_per_epoch = num_training_examples // train_batch_size
    halfway_steps = max(steps_per_epoch // 2, 1)

    callbacks = [LoggingCallback]

    # TODO: Set this up to handle other models besides gemma
    if freeze_embeddings:
        if "gemma" in hf_id:
            LOGGER.info("Freezing input embeddings...")
            for param in model.get_input_embeddings().parameters():
                param.requires_grad = False

    if freeze_unembeddings:
        if "gemma" in hf_id:
            LOGGER.info("Freezing unembeddings...")
            for param in model.get_output_embeddings().parameters():
                param.requires_grad = False

    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=float(cfg.training.learning_rate),
        weight_decay=float(cfg.training.weight_decay),
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
        num_train_epochs=cfg.training.num_train_epochs if not smoke_test else 2,
        eval_strategy=cfg.training.eval_strategy,
        eval_steps=halfway_steps,
        save_strategy=cfg.training.save_strategy,
        save_total_limit=cfg.training.save_total_limit,
        load_best_model_at_end=cfg.training.load_best_model_at_end,
        fp16=cfg.training.fp16 and torch.cuda.is_available(),
        report_to="none",  # "none" to disable logging, "wandb" to log to wandb
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        callbacks=callbacks,
    )

    ### TRAINING ###
    if cfg.model_checkpoint_parent is not None:
        LOGGER.info(
            f"Resuming training from checkpoint: {cfg.model_checkpoint_parent}..."
        )
        trainer.train(resume_from_checkpoint=True)
    else:
        LOGGER.info("Evaluating before training for baseline metrics...")
        trainer.evaluate()
        LOGGER.info("Starting fresh training run...")
        trainer.train()
    LOGGER.info("Training complete!")

    trainer.save_model(output_dir)


if __name__ == "__main__":
    # Note: Use argparse to allow submission of config file via slurm
    parser = argparse.ArgumentParser(description="Scoring script")
    parser.add_argument(
        "--config",
        type=str,
        default="config_train.yaml",
        help="Path to the config file",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Override config entries with KEY=VALUE pairs",
    )
    args = parser.parse_args()

    yaml_path = TRAINING_CONFIG_DIR / args.config
    LOGGER.info(f"Training with config: {yaml_path}")
    LOGGER.info(f"Overrides: {args.override}")
    cfg = load_training_config(yaml_path, overrides=args.override)

    train(cfg)
