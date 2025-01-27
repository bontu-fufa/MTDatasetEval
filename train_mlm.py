import os
import math
import logging
from datasets import load_dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def setup_logging(log_file=None):
    """
    Set up logging to print messages to the console and optionally save to a log file.
    """
    log_handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        log_handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        handlers=log_handlers,
    )
    logging.info("Logging setup complete.")


def load_and_tokenize_dataset(language, tokenizer, text_file, max_length=512, split_ratio=0.8):
    """
    Load and tokenize datasets for MLM training, with a train-test split.
    Args:
        language (str): Language being processed (e.g., am, om, ti).
        tokenizer (AutoTokenizer): Pre-trained tokenizer for the model.
        text_file (str): Path to the training text file.
        max_length (int): Maximum sequence length for tokenized inputs.
        split_ratio (float): Ratio of data to use for training (default: 0.8 for 80% train, 20% test).
    Returns:
        Tuple: Tokenized train dataset, Tokenized test dataset.
    """
    logging.info(f"Loading dataset for language: {language}")

    # Load the dataset from the text file
    dataset = load_dataset("text", data_files={"data": text_file})["data"]

    # Perform train-test split
    logging.info(f"Splitting dataset into train ({int(split_ratio*100)}%) and test ({int((1-split_ratio)*100)}%)...")
    dataset_split = dataset.train_test_split(test_size=1 - split_ratio)

    def tokenize_function(examples):
        """
        Tokenizes inputs and prepares labels for MLM.
        """
        inputs = tokenizer(
            examples["text"], max_length=max_length, truncation=True, padding="max_length"
        )
        inputs["labels"] = inputs["input_ids"]
        return inputs

    # Tokenize the train and test datasets
    tokenized_datasets = dataset_split.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )
    logging.info(f"Training examples: {len(tokenized_datasets['train'])}")
    logging.info(f"Testing examples: {len(tokenized_datasets['test'])}")

    return tokenized_datasets["train"], tokenized_datasets["test"]


def compute_metrics(eval_pred):
    """
    Compute perplexity from the evaluation loss.
    Args:
        eval_pred (tuple): Tuple containing predictions and labels (ignored for perplexity).
    Returns:
        dict: Dictionary containing the perplexity metric.
    """
    loss = eval_pred.metrics["eval_loss"]  # The Trainer automatically calculates eval_loss
    perplexity = math.exp(loss) if loss < float("inf") else float("inf")
    return {"perplexity": perplexity}


def train_mlm(language, dataset, train_dataset, test_dataset, tokenizer, model_name, output_dir, args):
    """
    Train a Masked Language Model (MLM) using HuggingFace's Trainer with dataset-aware output folder structure.
    Args:
        language (str): Language being processed (e.g., am, om, ti).
        dataset (str): Name of the dataset (e.g., hornMT, nllb).
        train_dataset (Dataset): Tokenized training dataset.
        test_dataset (Dataset): Tokenized validation dataset (optional).
        tokenizer (AutoTokenizer): Pre-trained tokenizer for the model.
        model_name (str): Name of the pre-trained model to use (e.g., mBERT, XLM-RoBERTa).
        output_dir (str): Directory to save the trained model.
        args (Namespace): Training arguments.
    """
    logging.info(f"Starting MLM fine-tuning for {language} on dataset {dataset} using {model_name}...")

    # Create a subdirectory for the dataset, language, and model
    dataset_language_output_dir = os.path.join(output_dir, dataset, language, model_name.replace("/", "_"))
    os.makedirs(dataset_language_output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=dataset_language_output_dir,
        evaluation_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        save_total_limit=2,
        logging_steps=args.logging_steps,
        logging_dir=os.path.join(dataset_language_output_dir, "logs"),
    )

    # Initialize HuggingFace Trainer
    trainer = Trainer(
        model=AutoModelForMaskedLM.from_pretrained(model_name),
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,  # Compute perplexity
    )

    # Train the model
    trainer.train()

    # Save the trained model and tokenizer
    trainer.save_model(dataset_language_output_dir)
    tokenizer.save_pretrained(dataset_language_output_dir)
    logging.info(f"Fine-tuned model saved to: {dataset_language_output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a Masked Language Model (MLM) with dataset-aware folders.")
    parser.add_argument("--language", type=str, required=True, help="Language code (e.g., am, om, ti).")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., hornMT, nllb).")
    parser.add_argument("--train_file", type=str, required=True, help="Path to the training text file.")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the trained tokenizer.")
    parser.add_argument("--output_dir", type=str, default="./output/mlm", help="Directory to save trained models.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the pre-trained model (e.g., bert-base-multilingual-cased, xlm-roberta-base).")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate for training.")
    parser.add_argument("--logging_steps", type=int, default=500, help="Log progress every N steps.")
    parser.add_argument("--split_ratio", type=float, default=0.8, help="Ratio of data to use for training (default: 0.8 for 80% train, 20% test).")
    parser.add_argument("--log_file", type=str, help="Optional log file for saving logs.")
    args = parser.parse_args()

    # Set up logging
    setup_logging(args.log_file)

    # Load the tokenizer
    logging.info(f"Loading tokenizer from: {args.tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    # Load and tokenize the dataset
    train_dataset, test_dataset = load_and_tokenize_dataset(
        args.language, tokenizer, args.train_file, max_length=512, split_ratio=args.split_ratio
    )

    # Train the MLM with dataset-aware folders
    train_mlm(
        args.language,
        args.dataset,
        train_dataset,
        test_dataset,
        tokenizer,
        args.model_name,
        args.output_dir,
        args,
    )
