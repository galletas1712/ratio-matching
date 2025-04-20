import os

import hydra
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import DPOConfig, DPOTrainer  # DPO trainer for preference optimization

import wandb
from utils import split_prompt_and_responses

# Ensure tokenizers parallelism flag is set
os.environ["TOKENIZERS_PARALLELISM"] = os.environ.get("TOKENIZERS_PARALLELISM", "false")


# Utility to print trainable parameter stats
def print_trainable_parameters(model):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    all_params = sum(p.numel() for p in model.parameters())
    print(
        f"trainable params: {trainable_params} || all params: {all_params} || trainable%: {100 * trainable_params / all_params:.2f}%"
    )


# Callback to merge and save LoRA adapters after each save
class MergeAndSaveCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if isinstance(model, PeftModel):
            ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            merged_dir = os.path.join(ckpt_dir, "merged")
            merged = model.merge_and_unload()
            merged.save_pretrained(merged_dir)
            print(f"Merged LoRA model saved to {merged_dir}")
        return control


@hydra.main(version_base=None, config_path="config", config_name="config_dpo")
def main(cfg: DictConfig):
    # Initialize Weights & Biases
    wandb.init(
        project=cfg.training.wandb_project,
        name=cfg.training.run_name,
        config=OmegaConf.to_container(cfg),
        save_code=True,
    )
    output_dir = cfg.hydra.run.dir

    # Print configuration
    print(OmegaConf.to_yaml(cfg))

    # Load and preprocess preference dataset
    ds = load_dataset(cfg.dataset.name)
    train_ds = ds[cfg.dataset.train_split]
    eval_ds = ds[cfg.dataset.eval_split]
    if cfg.dataset.train_size_limit:
        train_ds = train_ds.select(range(cfg.dataset.train_size_limit))
    if cfg.dataset.eval_size_limit:
        eval_ds = eval_ds.select(range(cfg.dataset.eval_size_limit))

    def to_pref(ex):
        prompt, chosen, rejected = split_prompt_and_responses(ex)
        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        }

    train_ds = train_ds.map(to_pref, remove_columns=train_ds.column_names)
    eval_ds = eval_ds.map(to_pref, remove_columns=eval_ds.column_names)
    print(f"Train size: {len(train_ds)}, Eval size: {len(eval_ds)}")

    # Load base model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(cfg.model.name, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = model.config.max_position_embeddings

    # Setup LoRA for PEFT if enabled
    peft_config = None
    ref_model = None
    if cfg.lora.enabled:
        target_modules = list(cfg.lora.target_modules)
        peft_config = LoraConfig(
            r=cfg.lora.r,
            lora_alpha=cfg.lora.lora_alpha,
            lora_dropout=cfg.lora.lora_dropout,
            target_modules=target_modules,
            bias=cfg.lora.bias,
            task_type=cfg.lora.task_type,
        )
        # Omit ref_model to let DPOTrainer create internal reference for PEFT
    else:
        # Full fine-tuning: load a reference copy
        ref_model = AutoModelForCausalLM.from_pretrained(
            cfg.model.name, trust_remote_code=True
        )

    print_trainable_parameters(model)

    # Configure DPO with parameters from config
    dpo_kwargs = OmegaConf.to_container(cfg.dpo)
    dpo_kwargs.update({"output_dir": output_dir})
    dpo_config = DPOConfig(**dpo_kwargs)

    # Initialize DPOTrainer
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
        callbacks=[MergeAndSaveCallback] if cfg.lora.enabled else None,
    )

    # Optional baseline evaluation
    if cfg.training.do_initial_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval_baseline", metrics)
        trainer.save_metrics("eval_baseline", metrics)

    # Run training
    trainer.train()

    # Save final models
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    if peft_config is not None:
        merged = trainer.model.merge_and_unload()
        merged.save_pretrained(os.path.join(output_dir, "dpo_lora_merged"))


if __name__ == "__main__":
    main()
