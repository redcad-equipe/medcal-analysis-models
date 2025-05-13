import torch
from unsloth import FastVisionModel, FastLanguageModel
from transformers import TextStreamer
from datasets import load_from_disk
from unsloth.trainer import UnslothVisionDataCollator
from unsloth import is_bf16_supported

# Load the downloaded models and tokenizer
download_dir = "downloads"
language_model_path = os.path.join(download_dir, "language_model")
tokenizer_path = os.path.join(download_dir, "tokenizer")
vision_model_path = os.path.join(download_dir, "vision_model")

print("Loading language model...")
language_model = FastLanguageModel.from_pretrained(language_model_path)
tokenizer = FastLanguageModel.from_pretrained(tokenizer_path)

print("Loading vision model...")
vision_model = FastVisionModel.from_pretrained(vision_model_path)

# Perform PEFT (Parameter Efficient Fine-Tuning) on the vision model
vision_model = FastVisionModel.get_peft_model(
    vision_model,
    finetune_vision_layers=True,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16,
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    random_state=3407,
    use_rslora=False,
    loftq_config=None,
)

# Load the dataset locally
print("Loading dataset...")
dataset = load_from_disk(os.path.join(download_dir, "dataset"))

# Convert dataset samples for training
def convert_to_conversation(sample):
    instruction = "You are an expert radiographer. Describe accurately what you see in this image."
    conversation = [
        {"role": "user", "content": [{"type": "text", "text": instruction}, {"type": "image", "image": sample["image"]}]},
        {"role": "assistant", "content": [{"type": "text", "text": sample["caption"]}]},
    ]
    return {"messages": conversation}

converted_dataset = [convert_to_conversation(sample) for sample in dataset]

# Before training, run inference to ensure the model works correctly
FastVisionModel.for_inference(vision_model)
image = dataset[0]["image"]
instruction = "You are an expert radiographer. Describe accurately what you see in this image."

messages = [
    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instruction}]}
]

input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
inputs = tokenizer(
    image,
    input_text,
    add_special_tokens=False,
    return_tensors="pt",
).to("cuda")

print("\nBefore training:\n")

# Inference with the model
text_streamer = TextStreamer(tokenizer, skip_prompt=True)
_ = vision_model.generate(**inputs, streamer=text_streamer, max_new_tokens=128, use_cache=True, temperature=1.5, min_p=0.1)

print("Inference completed.")
