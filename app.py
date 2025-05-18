# Imports
import os
os.environ["USE_TF"] = "0"
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import f1_score, accuracy_score
import torch

# Load GoEmotions dataset from HuggingFace Parquet (already downloaded)
df = pd.read_parquet("hf://datasets/google-research-datasets/go_emotions/simplified/train-00000-of-00001.parquet")

# Define emotion labels (28 classes)
emotion_labels = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral"
]

# Convert list of label indices to multi-hot vectors
def multi_hot(labels, num_classes=28):
    result = []
    for label_list in labels:
        vec = [0] * num_classes
        for idx in label_list:
            vec[idx] = 1
        result.append(vec)
    return result

# Split and convert labels
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"], df["labels"], test_size=0.2, random_state=42
)

train_labels = multi_hot(train_labels.tolist())
val_labels = multi_hot(val_labels.tolist())

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenize texts
train_encodings = tokenizer(
    train_texts.tolist(),
    truncation=True,
    padding="max_length",
    max_length=128
)
val_encodings = tokenizer(
    val_texts.tolist(),
    truncation=True,
    padding="max_length",
    max_length=128
)

# Custom PyTorch dataset
class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

    def __len__(self):
        return len(self.labels)

# Create train/val datasets
train_dataset = EmotionDataset(train_encodings, train_labels)
val_dataset = EmotionDataset(val_encodings, val_labels)

# Load model
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=28,
    problem_type="multi_label_classification"
)

# Metrics function
def compute_metrics(pred):
    logits = pred.predictions
    labels = pred.label_ids
    preds = (logits > 0).astype(int)
    f1 = f1_score(labels, preds, average="micro")
    acc = accuracy_score(labels, preds)
    return {"f1": f1, "accuracy": acc}

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    report_to="none",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=100,
    load_best_model_at_end=True,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

# Train!
trainer.train()

# Evaluate on validation set
results = trainer.evaluate()
print(f"Validation F1: {results['eval_f1']:.3f}")
print(f"Validation Accuracy: {results['eval_accuracy']:.3f}")

# Sample prediction
sample_text = "I'm so excited about this project!"
sample_input = tokenizer(sample_text, return_tensors="pt", truncation=True, padding=True)

# Move the input tensor to the same device as the model
sample_input = {k: v.to(model.device) for k, v in sample_input.items()}

outputs = model(**sample_input)
probs = torch.sigmoid(outputs.logits)
predicted_emotions = [emotion_labels[i] for i, p in enumerate(probs[0]) if p > 0.5]
print(f"Predicted Emotions: {predicted_emotions}")

model.save_pretrained("./saved_model")
tokenizer.save_pretrained("./saved_model")
