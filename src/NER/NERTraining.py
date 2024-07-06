import json
import numpy as np
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer, DataCollatorForTokenClassification
import tensorflow as tf
import torch

datasets = load_dataset('wnut_17', trust_remote_code=True)
label_list = datasets['train'].features['ner_tags'].feature.names

tokenizer = AutoTokenizer.from_pretrained(
    'roberta-base', add_prefix_space=True)
model = AutoModelForTokenClassification.from_pretrained(
    'roberta-base', num_labels=len(label_list))

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Save the label list for later use
label_map = {f'LABEL_{i}': label for i, label in enumerate(label_list)}

# Load pre-trained tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(
    'roberta-base', add_prefix_space=True)
model = AutoModelForTokenClassification.from_pretrained(
    'roberta-base', num_labels=len(label_list))


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples['tokens'], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples['ner_tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs['labels'] = labels
    return tokenized_inputs


tokenized_datasets = datasets.map(tokenize_and_align_labels, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
)

data_collator = DataCollatorForTokenClassification(tokenizer)
metric = load_metric('seqeval', trust_remote_code=True)


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = [[label_list[l] for l in label if l != -100]
                   for label in labels]
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = metric.compute(
        predictions=true_predictions, references=true_labels)
    return {
        'precision': results['overall_precision'],
        'recall': results['overall_recall'],
        'f1': results['overall_f1'],
        'accuracy': results['overall_accuracy'],
    }


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

# Save the model
model.save_pretrained('./NERModel')
tokenizer.save_pretrained('./NERModel')

# Save the label map
with open('./NERModel/label_map.json', 'w') as f:
    json.dump(label_map, f)
