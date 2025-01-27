# **Custom Tokenizer Training**

This project provides a script to train custom tokenizers for multiple languages using HuggingFace's **Tokenizers** library. The tokenizer is trained on language-specific datasets and saved for later use in tasks like fine-tuning a Masked Language Model (MLM).

---

## **Requirements**

- Python 3.7 or higher.

### **Dependencies**
Install the required Python libraries using pip:
```bash
pip install transformers tokenizers
```

## Usage
- 1. Train a Tokenizer for a Single Language
Use the train_tokenizer.py script to train a tokenizer for a specific language and dataset:

```bash
python train_tokenizer.py \
    --dataset <DATASET_NAME> \
    --language <LANGUAGE_CODE> \
    --output_dir <OUTPUT_DIR> \
    [OPTIONS]
```
## Arguments

| Argument       | Required? | Description                                                                 |
|----------------|-----------|-----------------------------------------------------------------------------|
| `--dataset`    | Yes       | Name of the dataset folder (e.g., `hornMT`, `nllb`).                        |
| `--language`   | Yes       | Language code to process (e.g., `am`, `om`, `ti`).                          |
| `--output_dir` | No        | Base output directory (default: `output`).                                  |
| `--vocab_size` | No        | Vocabulary size for the tokenizer (default: `30000`).                       |
| `--log_file`   | No        | Optional log file to save detailed logs (default: logs to console only).     |


### **Examples**

#### **Train Tokenizer for Amharic in `hornMT` Dataset**

```bash
python train_tokenizer.py --dataset hornMT --language am --output_dir output
```

- **output: output/hornMT/am/custom_tokenizer.json**


Specify a Log File
```bash
python train_tokenizer.py --dataset hornMT --language om --output_dir output --log_file logs/hornMT_om.lo
```


# **Masked Language Model (MLM) Fine-Tuning**

This project provides a flexible script to fine-tune **Masked Language Models (MLMs)** such as **mBERT** and **XLM-RoBERTa** on custom datasets for specific languages. The fine-tuning process integrates perplexity as a key evaluation metric, allowing better monitoring of model performance.

---

## **Features**
- **Multiple Model Support**:
  Fine-tune popular models like:
  - `bert-base-multilingual-cased` (mBERT)
  - `xlm-roberta-base` (XLM-RoBERTa)
- **Perplexity Metric Integration**:
  Automatically computes **perplexity** at the end of each epoch to evaluate model performance during fine-tuning.
- **Train-Test Splitting**:
  Automatically splits the dataset into training and testing sets based on a configurable ratio (default: 80/20 split).
- **Pre-Trained Tokenizers**:
  Supports custom pre-trained tokenizers for compatibility with the fine-tuning process.
- **Flexible Configuration**:
  Easily configure batch size, learning rate, number of epochs, and more via command-line arguments.
- **Organized Outputs**:
  Fine-tuned models and tokenizers are saved in language- and model-specific directories for easy retrieval.

---

## **Setup**

### **Requirements**
- Python 3.7 or higher
- Required libraries:
  ```bash
  pip install transformers datasets tokenizers



A brief description of what this project does and who it's for
## Arguments

| Argument          | Required? | Description                                                                                       |
|-------------------|-----------|---------------------------------------------------------------------------------------------------|
| `--language`      | Yes       | Language code (e.g., `am`, `om`, `ti`).                                                           |
| `--dataset`       | Yes       | Name of the dataset (e.g., `hornMT`, `nllb`).                                                     |
| `--train_file`    | Yes       | Path to the training text file for the language.                                                  |
| `--tokenizer_path`| Yes       | Path to the pre-trained tokenizer directory (e.g., `output/hornMT/am/`).                          |
| `--model_name`    | Yes       | Name of the pre-trained model (e.g., `bert-base-multilingual-cased`, `xlm-roberta-base`).         |
| `--output_dir`    | No        | Directory to save fine-tuned models (default: `output/mlm/`).                                     |
| `--epochs`        | No        | Number of training epochs (default: `3`).                                                        |
| `--batch_size`    | No        | Batch size for training and evaluation (default: `8`).                                           |
| `--learning_rate` | No        | Learning rate for the optimizer (default: `2e-5`).                                               |
| `--logging_steps` | No        | Log progress every N steps (default: `500`).                                                     |
| `--split_ratio`   | No        | Ratio of data to use for training (default: `0.8` for 80% train, 20% test).                      |
| `--log_file`      | No        | Optional log file to save logs (default: logs are printed to the console).  |


## Examples

*Fine-Tune mBERT for Amharic*

```bash title="Fine-Tune mBERT for Amharic"
python train_mlm.py \
    --language am \
    --dataset hornMT \
    --train_file data/hornMT/am.txt \
    --tokenizer_path output/hornMT/am \
    --model_name bert-base-multilingual-cased \
    --output_dir output/mlm \
    --epochs 3 \
    --batch_size 16 \
    --split_ratio 0.8
```

## Outputs
Fine-tuned models are saved in the following structure:


```lua title="Output Directory Structure"
output/
├── mlm/
│   ├── <dataset>/
│   │   ├── <language>/
│   │   │   ├── <model_name>/
│   │   │   │   ├── config.json
│   │   │   │   ├── pytorch_model.bin
│   │   │   │   ├── tokenizer.json
