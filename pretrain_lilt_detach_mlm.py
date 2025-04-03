import argparse
import os
from transformers import AutoTokenizer, TrainingArguments, Trainer
from data.pretraining.idl_dataset import IDLDataset, IDLDatasetNpy
from models.lilt_detach_for_masked_lm import LiltDetachForMaskedLM

def main():
    # Argument parser
    parser = argparse.ArgumentParser(description='Pretrain LiLT with MLM')
    parser.add_argument('--train-data', type=str, default="/home/masry20/projects/def-enamul/masry20/lilt_tmp_data", help='Path to training dataset')
    parser.add_argument('--output-dir', type=str, default="/home/masry20/projects/def-enamul/masry20/lilt_tmp_checkpoints/try_1_mlm_detach", help='Output Path')
    parser.add_argument('--num-train-epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Per-device batch size')
    parser.add_argument('--num-workers', type=int, default=16, help='Dataloader worker threads')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--log-every-n-steps', type=int, default=5, help='Logging steps')
    parser.add_argument('--warmup-ratio', type=int, default=0.1, help='Warmup Ratio')
    #parser.add_argument('--checkpoint-steps', type=int, default=1000, help='Checkpoint saving steps')
    parser.add_argument('--accumulate-grad-batches', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--checkpoint-path', type=str, default="ahmed-masry/lilt-roberta-en-base-init", help='Model checkpoint path')
    args = parser.parse_args()

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path, token=os.environ['HF_TOKEN'])
    model = LiltDetachForMaskedLM.from_pretrained(args.checkpoint_path, token=os.environ['HF_TOKEN'])

    # Load dataset
    train_dataset = IDLDatasetNpy(args.train_data, tokenizer=tokenizer)

    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        remove_unused_columns=False,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        logging_steps=args.log_every_n_steps,
        save_strategy="epoch",
        num_train_epochs=args.num_train_epochs,
        #save_steps=args.checkpoint_steps,
        #max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="linear",
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
