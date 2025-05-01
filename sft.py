import os

import hydra
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from peft import (
    LoraConfig,  # Import LoraConfig
    PeftModel,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
    TrainingArguments,
    # BitsAndBytesConfig # Keep commented unless adding quantization (QLoRA)
)
from trl import SFTTrainer

import wandb

# Check for TOKENIZERS_PARALLELISM environment variable
os.environ["TOKENIZERS_PARALLELISM"] = os.environ.get("TOKENIZERS_PARALLELISM", "false")


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


# --- HYDRA MAIN FUNCTION ---
@hydra.main(version_base=None, config_path="config", config_name="sft_config")
def main(cfg: DictConfig) -> None:
    wandb.init(
        project=cfg.wandb.wandb_project,
        name=cfg.wandb.run_name,
        config=OmegaConf.to_container(cfg),
        save_code=True,
    )

    output_dir = wandb.run.dir
    cfg.training.output_dir = output_dir
    
    # output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    
    class SaveAdapterCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            model = kwargs["model"]
            # only act on PeftModel (i.e. base+LoRA adapter)
            if isinstance(model, PeftModel):
                ckpt_dir    = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
                adapter_dir = os.path.join(ckpt_dir, "adapter_only")
                # this will create adapter_dir/adapter_config.json and pytorch_model.bin
                model.save_pretrained(adapter_dir)
                if cfg.wandb.save_files:
                    wandb.save(os.path.join(adapter_dir, "*"), base_path=args.output_dir)
            return control

    """Main training function driven by Hydra configuration."""
    print("-------------------- Configuration --------------------")
    print(OmegaConf.to_yaml(cfg))
    print("-------------------------------------------------------")

    # 2. Load Dataset
    print(f"Loading dataset: {cfg.dataset.name}")
    train_dataset = load_dataset(cfg.dataset.name, split="train")
    if "train_size_limit" in cfg.dataset and cfg.dataset.train_size_limit:
        train_dataset = train_dataset.select(range(cfg.dataset.train_size_limit))

    eval_dataset = load_dataset(cfg.dataset.name, split="test")
    if "eval_size_limit" in cfg.dataset and cfg.dataset.eval_size_limit:
        eval_dataset = eval_dataset.select(range(cfg.dataset.eval_size_limit))

    print("Dataset loaded.")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")

    model = AutoModelForCausalLM.from_pretrained(cfg.model.name, trust_remote_code=True)

    # 3. Load Tokenizer and Model
    print(f"Loading model and tokenizer: {cfg.model.name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set tokenizer pad_token to eos_token")
    tokenizer.padding_side = "right"

    tokenizer.model_max_length = model.config.max_position_embeddings
    effective_max_length = tokenizer.model_max_length
    print(
        f"Tokenizer model_max_length (used by default if not overridden): {effective_max_length}"
    )

    # 5. Setup PEFT/LoRA Config (Conditionally based on config)
    peft_config = None
    if cfg.lora.enabled:
        print("LoRA is enabled. Setting up LoraConfig.")
        # Ensure target_modules is a standard list
        target_modules_list = list(cfg.model.lora_target_modules)
        peft_config = LoraConfig(
            r=cfg.lora.r,
            lora_alpha=cfg.lora.lora_alpha,
            lora_dropout=cfg.lora.lora_dropout,
            target_modules=target_modules_list,
            bias=cfg.lora.bias,
            task_type=cfg.lora.task_type,  # Should be CAUSAL_LM
        )
        print(f"LoraConfig: {peft_config}")
    else:
        print("LoRA is disabled. Performing full fine-tuning.")

    # 6. Training Arguments (Reads from config)
    print("Setting up Training Arguments")
    cfg.training.output_dir = output_dir
    training_arguments = TrainingArguments(**cfg.training)

    # 7. Initialize SFTTrainer (Reads from config where applicable)
    print("Initializing SFTTrainer")
    trainer = SFTTrainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,  # Pass the LoraConfig object (or None)
        processing_class=tokenizer,
        formatting_func=lambda x: x[cfg.dataset.sft_column],
        callbacks=[SaveAdapterCallback],
        # max_seq_length is not passed directly
    )

    print("Model parameters:")
    print(model)
    print_trainable_parameters(model)

    # Run baseline evaluation first
    print("Running baseline evaluation...")
    baseline_metrics = trainer.evaluate()
    trainer.log_metrics("eval_baseline", baseline_metrics)
    trainer.save_metrics("eval_baseline", baseline_metrics)

    # 8. Train
    print(
        f"Starting training ({'LoRA enabled' if cfg.lora.enabled else 'Full fine-tuning'})..."
    )
    if cfg.training.num_train_epochs > 0:
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
    else:
        print("Skipping training as num_train_epochs <= 0")

    # 9. Save final model
    print(
        f"Saving final model ({'LoRA adapter' if cfg.lora.enabled else 'Full weights'})..."
    )
    adapter_directory = os.path.join(
        output_dir, "lora_adapter" if cfg.lora.enabled else "final_model"
    )
    trainer.save_model(adapter_directory)
    if cfg.wandb.save_files:
        wandb.save(os.path.join(adapter_directory, "*"), base_path=output_dir)
    print(
        f"{'PEFT' if cfg.lora.enabled else 'Final model'} checkpoint: model saved to {adapter_directory}"
    )

    if cfg.lora.enabled:
        merged = trainer.model.merge_and_unload()
        merged_dir = os.path.join(output_dir, "lora_merged")
        merged.save_pretrained(merged_dir)
        if cfg.wandb.save_files:
            wandb.save(os.path.join(merged_dir, "*"), base_path=output_dir)
        print(f"Finished saving the merged model to: {merged_dir}")


if __name__ == "__main__":
    main()
