# 1. Install necessary libraries
!pip install -q transformers datasets

import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from datasets import load_dataset

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

# 2. Load the pretrained dataset (Davidson Dataset)
# Note: The standard huggingface dataset uses 'class' for the label column
dataset = load_dataset("hate_speech_offensive")

# Split the dataset if it only comes with a 'train' split by default
train_test_split = dataset['train'].train_test_split(test_size=0.2, seed=42)
train_val_split = train_test_split['train'].train_test_split(test_size=0.1, seed=42)

datasets = {
    'train': train_val_split['train'],
    'validation': train_val_split['test'],
    'test': train_test_split['test']
}

# 3. Apply tokenization to training and validation data
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize_function(examples):
    # The text column in the Davidson dataset is usually 'tweet'
    return tokenizer(examples["tweet"], padding="max_length", truncation=True, max_length=128)

tokenized_datasets = {
    split: ds.map(tokenize_function, batched=True) for split, ds in datasets.items()
}

# Convert Hugging Face datasets to TensorFlow datasets
batch_size = 32

def to_tf_dataset(hf_dataset):
    return hf_dataset.to_tf_dataset(
        columns=["attention_mask", "input_ids"],
        label_cols=["class"], # Target labels
        shuffle=True,
        batch_size=batch_size,
    )

tf_train_dataset = to_tf_dataset(tokenized_datasets['train'])
tf_validation_dataset = to_tf_dataset(tokenized_datasets['validation'])

# 4. Compile the model
# Initialize DistilBERT for sequence classification with 3 labels
model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=3)

# Optimizer: Adam (Using a smaller learning rate is critical for fine-tuning transformers)
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)

# Loss function: Sparse Categorical Crossentropy
# CRITICAL: Transformers output logits, so from_logits MUST be True
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# Metrics: Accuracy
metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy')]

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# 5. Train the model
print("Starting training...")
history = model.fit(
    tf_train_dataset,
    validation_data=tf_validation_dataset,
    epochs=3 # Keep epochs low for fine-tuning to prevent overfitting
)

# Output final validation loss and accuracy to match your write-up
val_loss, val_accuracy = model.evaluate(tf_validation_dataset)
print(f"\nFinal Output:")
print(f"Validation Loss: {val_loss:.2f}")
print(f"Validation Accuracy: {val_accuracy:.2f}")