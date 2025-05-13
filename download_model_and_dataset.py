import os
from unsloth import FastVisionModel, FastLanguageModel
from datasets import load_dataset

# Set the model names and dataset
language_model_name = "unsloth/Llama-3.2-3B-Instruct"
vision_model_name = "unsloth/Llama-3.2-11B-Vision-Instruct"
dataset_name = "unsloth/Radiology_mini"

# Create a directory to store the downloaded files
download_dir = "downloads"
os.makedirs(download_dir, exist_ok=True)

# Download the language model
print("Downloading language model...")
language_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=language_model_name,
    max_seq_length=2048,  # Customize if needed
    dtype=None,  # Auto-detect data type
    load_in_4bit=True  # Use 4-bit quantization
)

# Save the model and tokenizer locally
language_model.save_pretrained(os.path.join(download_dir, "language_model"))
tokenizer.save_pretrained(os.path.join(download_dir, "tokenizer"))

# Download the vision model
print("Downloading vision model...")
vision_model = FastVisionModel.from_pretrained(
    vision_model_name,
    load_in_4bit=True,
    use_gradient_checkpointing="unsloth",
    token="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"
)

# Save the vision model locally
vision_model.save_pretrained(os.path.join(download_dir, "vision_model"))

# Download the dataset
print("Downloading dataset...")
dataset = load_dataset(dataset_name, split="train")

# Save the dataset locally
dataset.save_to_disk(os.path.join(download_dir, "dataset"))

print("Download completed and saved locally.")
