Custom Tokenizer Training
This project provides a script to train custom tokenizers for multiple languages using HuggingFace's Tokenizers library. The tokenizer is trained on language-specific datasets and saved for later use in tasks like fine-tuning a Masked Language Model (MLM).

Requirements
Python Version
Python 3.7 or higher.
Dependencies
Install the required Python libraries using pip:
pip install transformers tokenizers


Usage
1. Train a Tokenizer for a Single Language
Use the train_tokenizer.py script to train a tokenizer for a specific language and dataset.


python train_tokenizer.py --dataset <DATASET_NAME> --language <LANGUAGE_CODE> --output_dir <OUTPUT_DIR> [OPTIONS]

Arguments
Argument	Required?	Description
--dataset	Yes	Name of the dataset folder (e.g., hornMT, nllb).
--language	Yes	Language code to process (am, om, or ti).
--output_dir	No	Base output directory (default: output).
--vocab_size	No	Vocabulary size for the tokenizer (default: 30000).
--log_file	No	Optional log file to save detailed logs (default: logs to console only).
Examples
Train Tokenizer for Amharic in hornMT Dataset:

python train_tokenizer.py --dataset hornMT --language am --output_dir output
Output: output/hornMT/am/custom_tokenizer.json

Train Tokenizer for Tigrinya in nllb Dataset:

python train_tokenizer.py --dataset nllb --language ti --output_dir output

Specify a Log File:
python train_tokenizer.py --dataset hornMT --language om --output_dir output --log_file logs/hornMT_om.log

Expected Outputs
After training, the tokenizer will be saved in the following format:

output/
├── <DATASET_NAME>/
│   ├── <LANGUAGE_CODE>/
│       ├── custom_tokenizer.json   # Trained tokenizer file
│       ├── vocab.txt               # Vocabulary file (if applicable)

For example:
output/hornMT/am/custom_tokenizer.json
output/nllb/om/custom_tokenizer.json

Input Files
Each input file (e.g., am.txt, om.txt, ti.txt) should contain plain text sentences. Each line in the file is treated as a separate sentence for training.


Logging
Logs are printed to the console by default. To save logs to a file, use the --log_file argument. For example:

python train_tokenizer.py --dataset hornMT --language am --output_dir output --log_file logs/hornMT_am.log
