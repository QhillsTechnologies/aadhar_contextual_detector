Using the Aadhaar Classification Script



# 1. Training the Model
To train a new model (only if needed):

Ensure your dataset (core_context.csv) is in the project directory with context and label columns (labels: SENSITIVE or NON_SENSITIVE).

Run the script:python app.py

# 2. After Training (Make sure to comment this section)
To classify and mask Aadhaar numbers in new text using a pre-trained model:

#dataset_path = "core_context.csv"

#train_model(dataset_path, epochs=7)

Run the script: python app.py


The script will output the original text, classification, processed text, and whether masking was applied.
