import os
import logging
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from transformers import PreTrainedTokenizerFast
import argparse

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

def train_custom_tokenizer(language, input_file, output_dir, vocab_size=30000):
    """
    Train a custom tokenizer for a specific language and save it in HuggingFace-compatible format.

    Args:
        language (str): Language code (e.g., "am", "om", "ti").
        input_file (str): Path to the training text file for the language.
        output_dir (str): Directory to save the trained tokenizer.
        vocab_size (int): Vocabulary size for the tokenizer.
    """
    logging.info(f"Training tokenizer for language: {language}")
    
    # Ensure the output directory exists
    language_output_dir = os.path.join(output_dir, language)
    os.makedirs(language_output_dir, exist_ok=True)

    # Initialize the tokenizer (WordPiece model)
    tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.WordPieceTrainer(
        vocab_size=vocab_size,
        special_tokens=["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"],
    )

    # Read the input file
    if not os.path.exists(input_file):
        logging.error(f"Input file not found: {input_file}")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Train the tokenizer
    tokenizer.train_from_iterator(lines, trainer)

    # Save the tokenizer to the output directory in HuggingFace-compatible format
    logging.info(f"Saving tokenizer in HuggingFace-compatible format at: {language_output_dir}")
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        mask_token="[MASK]"
    )
    hf_tokenizer.save_pretrained(language_output_dir)
    logging.info(f"Tokenizer saved for {language} at: {language_output_dir}")

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train a custom tokenizer for a specific language.")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., hornMT, nllb).")
    parser.add_argument("--language", type=str, required=True, choices=["am", "om", "ti"], help="Language code (e.g., am, om, ti).")
    parser.add_argument("--vocab_size", type=int, default=30000, help="Vocabulary size for the tokenizer.")
    parser.add_argument("--output_dir", type=str, default="output", help="Base directory to save the tokenizer.")
    parser.add_argument("--log_file", type=str, help="Optional log file to save training logs.")
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_file)

    # Define the input file based on dataset and language
    input_file = os.path.join("data", args.dataset, f"{args.language}.txt")
    output_dir = os.path.join(args.output_dir, args.dataset)

    logging.info(f"Starting tokenizer training for {args.language} in dataset {args.dataset}.")
    logging.info(f"Input file: {input_file}")
    logging.info(f"Output directory: {output_dir}")

    # Train tokenizer
    train_custom_tokenizer(args.language, input_file, output_dir, args.vocab_size)