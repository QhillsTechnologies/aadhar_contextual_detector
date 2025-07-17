import pandas as pd
import re
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# Custom Dataset for Aadhaar Data (WITHOUT masking during training)
class AadhaarDataset(Dataset):
    def __init__(self, contexts, labels, tokenizer, max_length=128):
        self.contexts = contexts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        context = self.contexts[idx]
        # NO masking during training - let model learn from original text
        encoding = self.tokenizer(
            context,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Load and preprocess data
def load_data(file_path):
    df = pd.read_csv(file_path)
    contexts = df["context"].values
    labels = df["label"].values
    # Encode labels (SENSITIVE=1, NON_SENSITIVE=0)
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    return contexts, labels, label_encoder

# Evaluation function
def evaluate_model(model, val_loader, device):
    model.eval()
    predictions = []
    actual_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)
            
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
    
    accuracy = accuracy_score(actual_labels, predictions)
    return accuracy, predictions, actual_labels

# Training function
def train_model(file_path, model_save_path="distilbert_aadhaar_model", epochs=10):
    # Initialize tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    # Load data
    contexts, labels, label_encoder = load_data(file_path)
    train_contexts, val_contexts, train_labels, val_labels = train_test_split(
        contexts, labels, test_size=0.15, random_state=42
    )

    # Create datasets
    train_dataset = AadhaarDataset(train_contexts, train_labels, tokenizer)
    val_dataset = AadhaarDataset(val_contexts, val_labels, tokenizer)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    # Training setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)

    # Training loop with more epochs
    best_accuracy = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Evaluate on validation set
        val_accuracy, _, _ = evaluate_model(model, val_loader, device)
        avg_loss = total_loss / len(train_loader)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Average Loss: {avg_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
        
        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            model.save_pretrained(model_save_path)
            tokenizer.save_pretrained(model_save_path)
            print(f"New best model saved with accuracy: {best_accuracy:.4f}")

    # Save label encoder
    np.save("label_encoder_classes.npy", label_encoder.classes_)
    print(f"Training completed. Best model saved to {model_save_path}")
    print(f"Best validation accuracy: {best_accuracy:.4f}")

# Function to classify and conditionally mask text
def classify_and_mask(text, model, tokenizer, label_encoder, device, max_length=128):
    """
    Classify text and mask Aadhaar numbers only if classified as SENSITIVE
    """
    # Tokenize input
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )
    
    # Move to device
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    
    # Get prediction
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
        _, prediction = torch.max(outputs.logits, dim=1)
    
    # Convert prediction to label
    predicted_label = label_encoder.inverse_transform([prediction.cpu().item()])[0]
    
    # Mask only if SENSITIVE
    if predicted_label == "SENSITIVE":
        masked_text = re.sub(r"\d{4}\s?\d{4}\s?\d{4}", "[AADHAAR_MASKED]", text)
        return masked_text, predicted_label, True
    else:
        return text, predicted_label, False

# Function to load trained model and process new text
def load_model_and_process(model_path, text_to_process):
    """
    Load the trained model and process new text
    """
    # Load model and tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    
    # Load label encoder
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load("label_encoder_classes.npy", allow_pickle=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Process text
    result_text, predicted_label, was_masked = classify_and_mask(
        text_to_process, model, tokenizer, label_encoder, device
    )
    
    return {
        "original_text": text_to_process,
        "processed_text": result_text,
        "classification": predicted_label,
        "was_masked": was_masked
    }

if __name__ == "__main__":
    # Training phase
    dataset_path = "core_context.csv"
    train_model(dataset_path, epochs=7)  # Increased epochs
    
    # Example usage after training
    sample_texts = [
        "I'm applying for a new bank account and they've requested my Aadhaar details for KYC verification. My Aadhaar number is 1234 5678 9012 and I understand this will be used for identity verification purposes. Please ensure this information is stored securely and only used for the intended banking services. I'll also need to provide the physical copy during my branch visit.",
        "I'm creating a training manual for our customer service team about Indian identity documents. The Aadhaar system uses a 12-digit unique identification number that follows the format XXXX XXXX XXXX. This is important for our staff to understand when handling customer verification requests. The actual numbers are confidential and should never be shared in training materials."

    ]
    
    print("\n" + "="*50)
    print("TESTING CONDITIONAL MASKING")
    print("="*50)
    
    for text in sample_texts:
        result = load_model_and_process("distilbert_aadhaar_model", text)
        print(f"\nOriginal: {result['original_text']}")
        print(f"Classification: {result['classification']}")
        print(f"Processed: {result['processed_text']}")
        print(f"Was Masked: {result['was_masked']}")
        print("-" * 40)