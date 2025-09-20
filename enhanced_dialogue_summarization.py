# Enhanced Dialogue Summarization with SAMSum Dataset and Optimizations
# Target: Improve ROUGE scores from 0.44 to 0.60-0.70
# Research Code for Advanced NLP Research

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset, concatenate_datasets
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType
import evaluate
import time
import gc
import warnings
warnings.filterwarnings('ignore')
import os
from typing import Dict, List, Tuple, Optional
import json

# Set environment variable for memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check GPU
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("⚠ WARNING: No GPU detected!")

# ==================== DATASET LOADING AND COMBINATION ====================

def load_and_combine_datasets():
    """Load DialogSum and SAMSum datasets and combine them"""
    print("📊 Loading datasets...")

    # Load DialogSum
    try:
        dialogsum = load_dataset("knkarthick/dialogsum")
        print(f"DialogSum - Train: {len(dialogsum['train']):,}, Val: {len(dialogsum['validation']):,}, Test: {len(dialogsum['test']):,}")
    except Exception as e:
        print(f"⚠️ DialogSum loading failed: {e}")
        dialogsum = None

    # Load SAMSum
    try:
        samsum = load_dataset("samsum")
        print(f"SAMSum - Train: {len(samsum['train']):,}, Val: {len(samsum['validation']):,}, Test: {len(samsum['test']):,}")
    except Exception as e:
        print(f"⚠️ SAMSum loading failed: {e}")
        samsum = None

    # Use available datasets
    if dialogsum and samsum:
        # Combine datasets
        combined_train = concatenate_datasets([dialogsum['train'], samsum['train']])
        combined_val = concatenate_datasets([dialogsum['validation'], samsum['validation']])
        combined_test = concatenate_datasets([dialogsum['test'], samsum['test']])
        print(f"📈 Combined - Train: {len(combined_train):,}, Val: {len(combined_val):,}, Test: {len(combined_test):,}")
    elif samsum:
        print("📈 Using SAMSum only")
        combined_train = samsum['train']
        combined_val = samsum['validation']
        combined_test = samsum['test']
    elif dialogsum:
        print("📈 Using DialogSum only")
        combined_train = dialogsum['train']
        combined_val = dialogsum['validation']
        combined_test = dialogsum['test']
    else:
        print("❌ No datasets available, creating synthetic data")
        return create_synthetic_dataset()

    return {
        'train': combined_train,
        'validation': combined_val,
        'test': combined_test
    }

def create_synthetic_dataset():
    """Create synthetic dialogue summarization dataset for testing"""
    print("🔧 Creating synthetic dataset...")
    
    from datasets import Dataset
    
    # Sample dialogues and summaries
    dialogues = [
        "Alice: Hi Bob, how are you doing today? Bob: I'm doing great, thanks! Just finished my presentation. Alice: That's wonderful! How did it go? Bob: It went really well, the team loved the new features. Alice: I'm so happy for you!",
        "John: Hey Sarah, did you see the email about the meeting? Sarah: Yes, I saw it. It's scheduled for 3 PM tomorrow. John: Perfect, I'll be there. Sarah: Great, see you then!",
        "Mike: Lisa, can you help me with this project? Lisa: Of course! What do you need? Mike: I need help with the data analysis part. Lisa: I'd be happy to help. When do you need it by? Mike: By Friday if possible. Lisa: No problem, I'll get started on it today.",
        "Emma: Tom, I heard you got a promotion! Tom: Yes, I'm really excited about it. Emma: Congratulations! You deserve it. Tom: Thank you so much! Emma: When do you start your new role? Tom: Next Monday. Emma: That's fantastic!",
        "David: Rachel, are you free for lunch today? Rachel: I'm sorry, I have a meeting at noon. David: No worries, how about tomorrow? Rachel: Tomorrow works perfectly! David: Great, I'll see you at 12:30. Rachel: Sounds good!"
    ]
    
    summaries = [
        "Alice and Bob discussed Bob's successful presentation and the team's positive reaction to new features.",
        "John and Sarah confirmed their attendance at tomorrow's 3 PM meeting.",
        "Mike asked Lisa for help with data analysis for a project due Friday, and Lisa agreed to help.",
        "Emma congratulated Tom on his promotion and learned he starts his new role next Monday.",
        "David and Rachel planned to have lunch tomorrow at 12:30 since Rachel has a meeting today."
    ]
    
    # Create larger dataset by repeating and varying
    expanded_dialogues = []
    expanded_summaries = []
    
    for i in range(1000):  # Create 1000 samples
        base_idx = i % len(dialogues)
        dialogue = dialogues[base_idx]
        summary = summaries[base_idx]
        
        # Add some variation
        if i > len(dialogues):
            dialogue = dialogue.replace("Alice", f"Person{i%10}")
            dialogue = dialogue.replace("Bob", f"Friend{i%10}")
        
        expanded_dialogues.append(dialogue)
        expanded_summaries.append(summary)
    
    # Split into train/val/test
    train_size = int(0.8 * len(expanded_dialogues))
    val_size = int(0.1 * len(expanded_dialogues))
    
    train_data = Dataset.from_dict({
        'dialogue': expanded_dialogues[:train_size],
        'summary': expanded_summaries[:train_size]
    })
    
    val_data = Dataset.from_dict({
        'dialogue': expanded_dialogues[train_size:train_size+val_size],
        'summary': expanded_summaries[train_size:train_size+val_size]
    })
    
    test_data = Dataset.from_dict({
        'dialogue': expanded_dialogues[train_size+val_size:],
        'summary': expanded_summaries[train_size+val_size:]
    })
    
    print(f"📈 Synthetic Dataset - Train: {len(train_data):,}, Val: {len(val_data):,}, Test: {len(test_data):,}")
    
    return {
        'train': train_data,
        'validation': val_data,
        'test': test_data
    }

# ==================== ENHANCED DATA PREPROCESSING ====================

def create_enhanced_prompts(dialogues, summaries=None, is_training=True):
    """Create enhanced prompts with better instruction formatting"""
    prompts = []

    for i, dialogue in enumerate(dialogues):
        # Handle potential None values in dialogue
        cleaned_dialogue = dialogue.strip() if dialogue is not None else ""

        # Enhanced prompt with more specific instructions
        if is_training and summaries:
            prompt = f"""Summarize the following conversation in 1-2 concise sentences that capture the main points and outcome.

Conversation:
{cleaned_dialogue}

Summary: {summaries[i]}"""
        else:
            prompt = f"""Summarize the following conversation in 1-2 concise sentences that capture the main points and outcome.

Conversation:
{cleaned_dialogue}

Summary:"""

        prompts.append(prompt)

    return prompts

def advanced_tokenize_data(dataset, tokenizer, max_input_len=768, max_target_len=150):
    """Enhanced tokenization with better length handling"""

    def tokenize_function(examples):
        # Create enhanced prompts
        prompts = create_enhanced_prompts(examples["dialogue"], examples["summary"], is_training=True)

        # Tokenize inputs with attention to length
        model_inputs = tokenizer(
            prompts,
            max_length=max_input_len,
            truncation=True,
            padding=False,
            return_attention_mask=True
        )

        # Tokenize targets
        labels = tokenizer(
            examples["summary"],
            max_length=max_target_len,
            truncation=True,
            padding=False
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset"
    )

# ==================== ENHANCED TRAINING CONFIGURATION ====================

# Improved training configuration for better performance
training_config = {
    'batch_size': 2,  # Reduced for larger input sequences
    'gradient_accumulation_steps': 8,  # Effective batch size = 16
    'learning_rate': 1e-4,  # Slightly lower for stability
    'num_epochs': 4,  # More epochs for better convergence
    'max_input_length': 520,  # Longer context
    'max_target_length': 150,  # Longer summaries
    'warmup_ratio': 0.1,  # Learning rate warmup
    'weight_decay': 0.01,  # Regularization
    'lora_r': 16,  # Keep same as requested
    'lora_alpha': 32,  # Keep same as requested
    'lora_dropout': 0.1,  # Keep same as requested
    'save_strategy': 'epoch',
    'evaluation_strategy': 'epoch',
    'logging_steps': 50,
    'save_total_limit': 2,
    'load_best_model_at_end': True,
    'metric_for_best_model': 'eval_rouge1',
    'greater_is_better': True,
}

def setup_enhanced_lora_training(model):
    """Setup LoRA with enhanced target modules for FLAN-T5"""

    # Enhanced target modules for better coverage
    target_modules = [
        "q", "v", "k", "o",  # Attention layers
        "wi_0", "wi_1", "wo",  # Feed-forward layers
        "shared", "lm_head"  # Additional important layers
    ]

    lora_config = LoraConfig(
        r=training_config['lora_r'],
        lora_alpha=training_config['lora_alpha'],
        target_modules=target_modules,
        lora_dropout=training_config['lora_dropout'],
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
    )

    peft_model = get_peft_model(model, lora_config)

    # Print parameter info
    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in peft_model.parameters())
    print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    return peft_model

# ==================== ADVANCED TRAINING WITH CUSTOM METRICS ====================

def compute_metrics(eval_pred):
    """Compute ROUGE metrics during training"""
    predictions, labels = eval_pred

    # Decode predictions and labels
    tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base')
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # Replace -100 in labels as we can't decode them
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Compute ROUGE
    rouge = evaluate.load('rouge')
    result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True,
        use_aggregator=True
    )

    return {
        'eval_rouge1': result['rouge1'],
        'eval_rouge2': result['rouge2'],
        'eval_rougeL': result['rougeL']
    }

def train_enhanced_model(model, tokenizer, train_dataset, val_dataset):
    """Enhanced training with better configuration"""

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        max_length=training_config['max_input_length']
    )

    # Enhanced training arguments
    training_args = TrainingArguments(
        output_dir='./enhanced_flan_t5_results',
        per_device_train_batch_size=training_config['batch_size'],
        per_device_eval_batch_size=training_config['batch_size'],
        gradient_accumulation_steps=training_config['gradient_accumulation_steps'],
        learning_rate=training_config['learning_rate'],
        num_train_epochs=training_config['num_epochs'],
        warmup_ratio=training_config['warmup_ratio'],
        weight_decay=training_config['weight_decay'],
        eval_strategy=training_config['evaluation_strategy'],
        save_strategy=training_config['save_strategy'],
        logging_steps=training_config['logging_steps'],
        save_total_limit=training_config['save_total_limit'],
        load_best_model_at_end=True,
        metric_for_best_model='eval_rouge1',
        greater_is_better=True,
        report_to=None,
        remove_unused_columns=False,
        dataloader_pin_memory=False,
        fp16=True,  # Mixed precision training
        gradient_checkpointing=True,  # Memory efficiency
        dataloader_num_workers=2,
    )

    # Create trainer with metrics
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("🚀 Starting enhanced training...")
    start_time = time.time()

    # Train with error handling
    try:
        trainer.train()
        training_time = time.time() - start_time
        print(f"✅ Training completed in {training_time/60:.1f} minutes")

        # Get final metrics
        final_metrics = trainer.evaluate()
        print("📊 Final Validation Metrics:")
        for key, value in final_metrics.items():
            if 'rouge' in key:
                print(f"{key}: {value:.4f}")

        # Save trainer state (includes checkpoints)
        trainer.save_state()
        print(f"💾 Trainer state saved to {training_args.output_dir}/trainer_state.json")

        return trainer, training_time, final_metrics

    except Exception as e:
        print(f"❌ Training failed: {e}")
        return None, 0, {}

# ==================== ENHANCED INFERENCE ====================

def generate_enhanced_summaries(model, tokenizer, dialogues, batch_size=4):
    """Generate summaries with enhanced decoding parameters"""
    summaries = []
    model.eval()

    for i in range(0, len(dialogues), batch_size):
        batch_dialogues = dialogues[i:i+batch_size]
        prompts = create_enhanced_prompts(batch_dialogues, is_training=False)

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=training_config['max_input_length']
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=training_config['max_target_length'],
                num_beams=6,  # More beams for better quality
                length_penalty=0.8,  # Slightly prefer longer summaries
                repetition_penalty=1.2,  # Reduce repetition
                early_stopping=True,
                do_sample=False,
                no_repeat_ngram_size=3,  # Prevent 3-gram repetitions
            )

        batch_summaries = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # Clean up summaries (remove the input prompt)
        cleaned_summaries = []
        for summary in batch_summaries:
            # Extract only the generated part after "Summary:"
            if "Summary:" in summary:
                summary = summary.split("Summary:")[-1].strip()
            cleaned_summaries.append(summary)

        summaries.extend(cleaned_summaries)

        if i % (batch_size * 10) == 0:
            torch.cuda.empty_cache()

    return summaries

# ==================== RESULTS ANALYSIS ====================

def analyze_results(predictions, references, dialogues):
    """Analyze and visualize results"""
    print("📊 Analyzing results...")
    
    # Calculate additional metrics
    rouge = evaluate.load('rouge')
    rouge_scores = rouge.compute(
        predictions=predictions,
        references=references,
        use_aggregator=True,
        use_stemmer=True
    )
    
    # Calculate BLEU scores
    bleu = evaluate.load('bleu')
    bleu_scores = bleu.compute(
        predictions=predictions,
        references=references
    )
    
    # Calculate BERTScore
    try:
        bertscore = evaluate.load('bertscore')
        bert_scores = bertscore.compute(
            predictions=predictions,
            references=references,
            lang='en'
        )
        bert_f1 = np.mean(bert_scores['f1'])
    except:
        bert_f1 = 0.0
    
    # Print comprehensive results
    print("\n🎯 COMPREHENSIVE EVALUATION RESULTS:")
    print("=" * 60)
    print(f"ROUGE-1: {rouge_scores['rouge1']:.4f}")
    print(f"ROUGE-2: {rouge_scores['rouge2']:.4f}")
    print(f"ROUGE-L: {rouge_scores['rougeL']:.4f}")
    print(f"BLEU: {bleu_scores['bleu']:.4f}")
    print(f"BERTScore F1: {bert_f1:.4f}")
    print("=" * 60)
    
    # Show sample results
    print("\n📝 SAMPLE RESULTS:")
    print("-" * 80)
    for i in range(min(5, len(dialogues))):
        print(f"\nExample {i+1}:")
        print(f"Dialogue: {dialogues[i][:200]}...")
        print(f"Reference: {references[i]}")
        print(f"Generated: {predictions[i]}")
        print("-" * 40)
    
    return {
        'rouge': rouge_scores,
        'bleu': bleu_scores,
        'bert_f1': bert_f1
    }

# ==================== MAIN EXECUTION ====================

def main():
    """Main execution function"""
    print("🎯 Enhanced Dialogue Summarization Research")
    print("Target: Improve ROUGE scores from 0.44 to 0.60-0.70")
    print("=" * 70)

    # Load datasets
    combined_dataset = load_and_combine_datasets()

    # Load model and tokenizer
    print("\n🤖 Loading FLAN-T5-Base model...")
    model_name = 'google/flan-t5-base'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Use FP16 for memory efficiency
        device_map="auto"
    )

    print(f"✅ {model_name} loaded successfully")

    # Tokenize datasets
    print("\n🔧 Tokenizing datasets...")
    train_tokenized = advanced_tokenize_data(
        combined_dataset['train'],
        tokenizer,
        training_config['max_input_length'],
        training_config['max_target_length']
    )
    val_tokenized = advanced_tokenize_data(
        combined_dataset['validation'],
        tokenizer,
        training_config['max_input_length'],
        training_config['max_target_length']
    )

    # Setup LoRA
    print("\n⚡ Setting up enhanced LoRA...")
    peft_model = setup_enhanced_lora_training(model)

    # Train model
    print("\n🚀 Starting training phase...")
    trainer, training_time, metrics = train_enhanced_model(
        peft_model, tokenizer, train_tokenized, val_tokenized
    )

    if trainer is None:
        print("❌ Training failed, exiting...")
        return

    # Evaluate on test set
    print("\n🧪 Evaluating on test set...")
    test_sample = combined_dataset['test'].select(range(min(1000, len(combined_dataset['test']))))
    test_dialogues = test_sample['dialogue']
    test_references = test_sample['summary']

    test_generated = generate_enhanced_summaries(
        peft_model, tokenizer, test_dialogues
    )

    # Analyze results
    results = analyze_results(test_generated, test_references, test_dialogues)

    # Save model and results
    output_dir = "./enhanced_flan_t5_final"
    peft_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save results to JSON
    results_to_save = {
        'rouge_scores': results['rouge'],
        'bleu_score': results['bleu']['bleu'],
        'bert_f1': results['bert_f1'],
        'training_time': training_time,
        'model_name': model_name,
        'training_config': training_config
    }
    
    with open(f"{output_dir}/results.json", 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    print(f"\n💾 Model and results saved to {output_dir}")
    
    # Final summary
    print("\n🏆 RESEARCH SUMMARY:")
    print("=" * 50)
    print(f"Model: {model_name} with Enhanced LoRA")
    print(f"ROUGE-L Score: {results['rouge']['rougeL']:.4f}")
    print(f"Target Achieved: {'✅ YES' if results['rouge']['rougeL'] >= 0.60 else '❌ NO'}")
    print(f"Training Time: {training_time/60:.1f} minutes")
    print("=" * 50)

    return results

# Run the enhanced training
if __name__ == "__main__":
    final_results = main()