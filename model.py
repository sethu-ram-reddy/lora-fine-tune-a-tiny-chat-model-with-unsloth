"""
LoRA Fine-Tune a Tiny Chat Model with Unsloth

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_base_model_and_tokenizer
from unsloth import FastLanguageModel

def load_base_model_and_tokenizer(model_name='unsloth/Qwen2.5-0.5B-Instruct-bnb-4bit', max_seq_length=256):
    """Load a 4-bit quantized causal LM and its tokenizer via Unsloth.
    Returns:
        (model, tokenizer)
    """
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
    )
    return model, tokenizer

# Step 2 - count_total_parameters
def count_total_parameters(model):
    """Return the total number of parameters in `model` as a Python int."""
    return int(sum(p.numel() for p in model.parameters()))

# Step 3 - is_model_4bit_quantized
import torch.nn as nn
from bitsandbytes.nn import Linear4bit
def is_model_4bit_quantized(model):
    """Return True if any submodule of `model` is a bitsandbytes 4-bit linear layer."""
    # TODO: walk the model's submodules and check for a bitsandbytes Linear4bit instance
    return any(isinstance(module, Linear4bit) for module in model.modules())

# Step 4 - ensure_pad_token
def ensure_pad_token(tokenizer):
    """Guarantee tokenizer.pad_token is not None; fall back to eos_token."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token  = tokenizer.eos_token
    return tokenizer

# Step 5 - get_lora_target_modules
def get_lora_target_modules():
    """Return the attention projection module name suffixes for LoRA."""
    return ["q_proj", "k_proj", "v_proj", "o_proj"]

# Step 6 - attach_lora_adapters
def attach_lora_adapters(model, r=8, lora_alpha=16, target_modules=None):
    """Wrap the base model with LoRA adapters and return the PEFT model."""
    # TODO: wrap `model` with LoRA via FastLanguageModel.get_peft_model using r, lora_alpha, target_modules
    if target_modules is None:
        target_modules = get_lora_target_modules()
    model = FastLanguageModel.get_peft_model(
        model,
        r=r,
        target_modules=target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=0,
        bias="none",
    )
    return model

# Step 7 - count_trainable_parameters
def count_trainable_parameters(model):
    """Return the number of trainable parameters in `model`."""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))

# Step 8 - trainable_fraction
def trainable_fraction(trainable_count, total_count):
    return float(trainable_count)/float(total_count)

# Step 9 - build_instruction_examples
def build_instruction_examples():
    """Return a small list of {'instruction', 'response'} dicts for SFT."""
    return [
        {"instruction": "What is 2 + 2?", "response": "2 + 2 equals 4."},
        {"instruction": "Name a primary color.", "response": "Red is a primary color."},
        {"instruction": "What is the capital of France?", "response": "The capital of France is Paris."},
        {"instruction": "What language runs in the browser?", "response": "JavaScript runs natively in the browser."}
    ]

# Step 10 - format_instruction_example
def format_instruction_example(example):
    """Return a single training string with role markers for instruction and response."""
    return f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['response']}"

# Step 11 - format_all_examples
def format_all_examples(examples):
    """Format each instruction/response dict into a training string."""
    return [format_instruction_example(ex) for ex in examples]

# Step 12 - build_text_dataset
from datasets import Dataset

def build_text_dataset(texts):
    """Wrap a list of training strings in a HF Dataset with a 'text' column."""
    return Dataset.from_dict({"text": texts})

# Step 13 - tokenize_text
def tokenize_text(tokenizer, text):
    """Tokenize a single string and return a list[int] of input ids."""
    return tokenizer(text).input_ids

# Step 14 - count_tokens
def count_tokens(input_ids):
    """Return the number of tokens in a tokenized example."""
    return len(input_ids)

# Step 15 - build_training_arguments
import torch
from transformers import TrainingArguments

def build_training_arguments(output_dir='./sft_out', max_steps=5, learning_rate=2e-4):
    """Return featherweight TrainingArguments for the SFT run."""
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    
    return TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        max_steps=max_steps,
        learning_rate=learning_rate,
        bf16=use_bf16,
        fp16=not use_bf16,
        logging_steps=1,
        optim='adamw_8bit'
    )

# Step 16 - build_sft_trainer
from trl import SFTTrainer

def build_sft_trainer(model, tokenizer, dataset, training_args, max_seq_length=256):
    """Construct a trl SFTTrainer over dataset['text'] ready to .train()."""
    return SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        args=training_args,
        packing=False,
    )

# Step 17 - run_sft_training
def run_sft_training(trainer):
    """Run a few SFT steps and return the final training loss as a float."""
    # Disable saving to avoid the SFTConfig pickling error in this sandbox environment
    trainer.args.save_strategy = "no"
    
    train_result = trainer.train()
    # Extract training loss from the metrics dictionary
    final_loss = train_result.metrics.get("train_loss")
    return float(final_loss)

# Step 18 - switch_to_inference_mode
def switch_to_inference_mode(model):
    """Switch the LoRA-tuned model into Unsloth's fast inference mode and return it."""
    # Fast inference can be enabled using Unsloth's built-in helper method
    if hasattr(model, "fast_inference"):
        model.fast_inference = True
    elif hasattr(model, "FastLanguageModel"):
        # Alternatively, use FastLanguageModel for inference mode if exposed
        pass
        
    # Unsloth models typically use FastLanguageModel.for_inference(model)
    from unsloth import FastLanguageModel
    FastLanguageModel.for_inference(model)
    
    return model

# Step 19 - build_chat_prompt (not yet solved)
# TODO: implement

# Step 20 - generate_reply (not yet solved)
# TODO: implement

