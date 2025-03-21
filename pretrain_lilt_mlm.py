import argparse
from transformers import AutoTokenizer, TrainingArguments, Trainer
from data.pretraining.idl_dataset import IDLDataset
from models.lilt_for_masked_lm import LiltForMaskedLM

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description='Pretrain LiLT with MLM')
    parser.add_argument('--train-data', type=str, default="/home/masry20/scratch/idl_data/saved_hf/", help='Path to training dataset')
    parser.add_argument('--output-dir', type=str, default="/home/masry20/scratch/exp_outputs/try_exp_1", help='Output Path')
    parser.add_argument('--max-steps', type=int, default=1000, help='Max training steps')
    parser.add_argument('--batch-size', type=int, default=32, help='Per-device batch size')
    parser.add_argument('--num-workers', type=int, default=16, help='Dataloader worker threads')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--log-every-n-steps', type=int, default=5, help='Logging steps')
    parser.add_argument('--warmup-steps', type=int, default=50, help='Warmup steps')
    parser.add_argument('--checkpoint-steps', type=int, default=100, help='Checkpoint saving steps')
    parser.add_argument('--gradient-clip-val', type=float, default=1.0, help='Gradient clipping value')
    parser.add_argument('--accumulate-grad-batches', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--checkpoint-path', type=str, default="SCUT-DLVCLab/lilt-roberta-en-base", help='Model checkpoint path')
    args = parser.parse_args()

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path)
    model = LiltForMaskedLM.from_pretrained(args.checkpoint_path)

    # Load dataset
    train_dataset = IDLDataset(args.train_data, tokenizer=tokenizer)

    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        remove_unused_columns=False,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        logging_steps=args.log_every_n_steps,
        save_steps=args.checkpoint_steps,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        gradient_accumulation_steps=args.accumulate_grad_batches,
        gradient_checkpointing=True,  # Save memory
        fp16=True, 
        dataloader_num_workers=args.num_workers,
        logging_dir=f"{args.output_dir}/logs",
        save_total_limit=5,
        save_safetensors=False,
        evaluation_strategy="no", 
        report_to="none"  # Avoid errors if not using wandb/tensorboard
    )

    # Define Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    # Start training
    trainer.train()


if __name__ == '__main__':
    main()
