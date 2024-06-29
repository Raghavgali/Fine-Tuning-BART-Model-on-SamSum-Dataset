# Fine-Tuned BART Model on SAMSum Dataset

This repository contains a fine-tuned BART model on the SAMSum dataset, aimed at generating abstractive summaries for dialogues. The model has been trained for one epoch using Hugging Face's Transformers library.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model](#model)
- [Training](#training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Abstractive summarization is a challenging task in natural language processing (NLP) that involves generating a concise and coherent summary of a longer text. This project fine-tunes the BART (Bidirectional and Auto-Regressive Transformers) model on the SAMSum dataset to generate summaries for dialogues.

## Dataset

The [SAMSum dataset](https://arxiv.org/abs/1911.12237) consists of about 16,000 chat dialogues created by linguists. Each dialogue is accompanied by a summary.

## Model

We use the `facebook/bart-large-cnn` model as the base model for fine-tuning. BART is a powerful model for sequence-to-sequence tasks and is pre-trained on a large corpus of text.

## Training

The model was fine-tuned for one epoch on the SAMSum dataset using the following training arguments:

- **Batch size**: 2
- **Learning rate**: 2e-5
- **Warmup steps**: 500
- **Weight decay**: 0.01
- **Gradient accumulation steps**: 2
- **Evaluation strategy**: "epoch"
- **Logging steps**: 10
- **Save total limit**: 2
- **fp16**: True (using 16-bit floating point precision for faster training)

## Evaluation

The model's performance was evaluated using the ROUGE metric. The following script was used for evaluation:

```python
import torch
from transformers import pipeline
from tqdm import tqdm
import pandas as pd
from datasets import load_metric

# Load the dataset
test_data = pd.read_csv('path/to/test.csv')  # Replace with your test dataset path

# Set up device
device = 0 if torch.cuda.is_available() else -1
summarizer = pipeline("summarization", model="path/to/finetuned/model", device=device)  # Replace with your model path

# Initialize the ROUGE metric
rouge_metric = load_metric("rouge")

# Evaluate in batches
batch_size = 8
for i in tqdm(range(0, len(test_data), batch_size)):
    batch = test_data.iloc[i:i + batch_size]
    dialogues = batch['dialogue'].tolist()
    summaries = batch['summary'].tolist()
    outputs = summarizer(dialogues)
    generated_summaries = [output['summary_text'] for output in outputs]
    rouge_metric.add_batch(predictions=generated_summaries, references=summaries)

# Compute ROUGE scores
rouge_scores = rouge_metric.compute()
print(rouge_scores)
