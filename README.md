Using the Aadhaar Classification Script

1. Inference (Classifying and Masking Text)
To classify and mask Aadhaar numbers in new text using a pre-trained model:

Ensure the pre-trained model (distilbert_aadhaar_model) and label_encoder_classes.npy are in the project directory.
Keep the training section commented out:# Training phase
#dataset_path = "core_context.csv"
#train_model(dataset_path, epochs=7)


Modify the sample_texts list in the script to include your text:sample_texts = [
    "Your text with Aadhaar number 1234 5678 9012.",
    "Aadhaar format is XXXX XXXX XXXX."
]


Run the script:python app.py


The script will output the original text, classification, processed text, and whether masking was applied.

2. Training the Model
To train a new model (only if needed):

Ensure your dataset (core_context.csv) is in the project directory with context and label columns (labels: SENSITIVE or NON_SENSITIVE).
Uncomment the training section in the script:# Training phase
dataset_path = "core_context.csv"
train_model(dataset_path, epochs=7)


Run the script:python app.py


