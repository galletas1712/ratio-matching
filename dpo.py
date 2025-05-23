import os

import hydra
import torch
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from peft import (
    LoraConfig,
    PeftModel,
    # prepare_model_for_kbit_training # Uncomment for QLoRA
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    # BitsAndBytesConfig # Uncomment for QLoRA
)
from trl import DPOConfig
from dpo_trainer import DPOTrainer

import wandb

# Import the utility function
from utils import split_prompt_and_responses

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
@hydra.main(version_base=None, config_path="config", config_name="dpo_config") # Changed config name
def main(cfg: DictConfig) -> None:
    wandb.init(
        project=cfg.wandb.wandb_project,
        name=cfg.wandb.run_name,
        config=OmegaConf.to_container(cfg),
        save_code=True,
    )

    output_dir = wandb.run.dir
    cfg.training.output_dir = output_dir

    # Removed SaveAdapterCallback - rely on standard checkpointing and final save

    # Define the preprocessing function provided by the user
    def to_pref(ex):
        prompt, chosen, rejected = split_prompt_and_responses(ex)
        return {
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
        }

    """Main DPO training function driven by Hydra configuration."""
    print("-------------------- Configuration (DPO) --------------------")
    print(OmegaConf.to_yaml(cfg))
    print("-------------------------------------------------------------")

    # 2. Load Dataset (Ensure it has 'prompt', 'chosen', 'rejected' columns)
    print(f"Loading dataset: {cfg.dataset.name}")
    raw_train_dataset = load_dataset(cfg.dataset.name, split="train")
    raw_eval_dataset = load_dataset(cfg.dataset.name, split="test")

    # Apply preprocessing
    print("Preprocessing dataset...")
    train_dataset = raw_train_dataset.map(to_pref)
    eval_dataset = raw_eval_dataset.map(to_pref)

    # Apply size limits after preprocessing
    if "train_size_limit" in cfg.dataset and cfg.dataset.train_size_limit:
        train_dataset = train_dataset.select(range(cfg.dataset.train_size_limit))
    if "eval_size_limit" in cfg.dataset and cfg.dataset.eval_size_limit:
        eval_dataset = eval_dataset.select(range(cfg.dataset.eval_size_limit))

    print("Dataset loaded and preprocessed.")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Eval dataset size: {len(eval_dataset)}")
    print(f"Example train instance: {train_dataset[0]}") # Log an example

    # 3. Load Tokenizer and Model
    print(f"Loading model and tokenizer: {cfg.model.name}")
    # --- QLoRA Configuration (Optional) ---
    # bnb_config = None
    # if cfg.qlora.enabled:
    #     print("QLoRA is enabled. Setting up BitsAndBytesConfig.")
    #     bnb_config = BitsAndBytesConfig(
    #         load_in_4bit=True,
    #         bnb_4bit_quant_type="nf4",
    #         bnb_4bit_compute_dtype=torch.bfloat16 if cfg.training.bf16 else torch.float16,
    #         bnb_4bit_use_double_quant=True,
    #     )
    #     # Set device_map for QLoRA
    #     device_map = {"": 0} # Or adjust for multi-GPU

    model_kwargs = dict(
        trust_remote_code=True,
        # quantization_config=bnb_config, # Uncomment for QLoRA
        # device_map=device_map if cfg.qlora.enabled else None # Uncomment for QLoRA
        torch_dtype=torch.bfloat16 if cfg.training.get("bf16", False) else (torch.float16 if cfg.training.fp16 else None),
    )
    
    # Load base model
    print(f"Loading base model: {cfg.model.name}")
    model = AutoModelForCausalLM.from_pretrained(cfg.model.name, **model_kwargs)
    print("Base model loaded.")

    # --- Load and Merge SFT Adapter (if specified) ---
    if "sft_adapter_path" in cfg.model and cfg.model.sft_adapter_path:
        sft_adapter_path = cfg.model.sft_adapter_path
        print(f"Loading and merging SFT adapter from: {sft_adapter_path}")
        try:
            # Load the SFT adapter onto the base model
            model = PeftModel.from_pretrained(model, sft_adapter_path, is_trainable=False) # Load non-trainable initially
            print("SFT adapter loaded onto base model.")
            
            # Merge the adapter into the base model
            model = model.merge_and_unload()
            print("SFT adapter merged into base model.")
            # Now 'model' is the merged model, ready for DPO (either full fine-tuning or with a new DPO LoRA adapter)
        except Exception as e:
            print(f"Error loading or merging SFT adapter from {sft_adapter_path}: {e}")
            print("Proceeding with the base model only.")
    else:
        print("No SFT adapter path specified. Using the base model directly for DPO.")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        print("Set tokenizer pad_token to eos_token")
    tokenizer.padding_side = "left" # DPO usually uses left padding for generation consistency

    # 5. Setup PEFT/LoRA Config for DPO (Conditionally based on config)
    peft_config = None
    if cfg.lora.enabled:
        print("LoRA is enabled for DPO. Setting up LoraConfig.")
        # Ensure target_modules is a standard list
        target_modules_list = list(cfg.model.lora_target_modules)
        peft_config = LoraConfig(
            r=cfg.lora.r,
            lora_alpha=cfg.lora.lora_alpha,
            lora_dropout=cfg.lora.lora_dropout,
            target_modules=target_modules_list,
            bias=cfg.lora.bias,
            task_type=cfg.lora.task_type,
        )
        print(f"LoraConfig for DPO: {peft_config}")

        # --- Prepare model for QLoRA (if enabled) ---
        # if cfg.qlora.enabled:
        #     print("Preparing model for K-bit training (QLoRA)")
        #     model = prepare_model_for_kbit_training(model)

        # --- If SFT adapter was loaded, make sure new LoRA layers are trainable ---
        # if "sft_adapter_path" in cfg.model and cfg.model.sft_adapter_path:
        #    pass # PeftModel handles adding new adapters correctly

    else:
        print("LoRA is disabled. Performing full DPO fine-tuning.")

    # 6. Training Arguments (Reads from config)
    print("Setting up Training Arguments for DPO")
    print("Max tokens", model.config.max_position_embeddings)
    cfg.training.output_dir = output_dir
    # Ensure remove_unused_columns is False for DPO
    cfg.training.remove_unused_columns = False
    training_arguments = DPOConfig(
        **cfg.training,
        **cfg.dpo,
        max_prompt_length=None,
        max_completion_length=None,
        max_length=model.config.max_position_embeddings,
    )
    
    # 7. Initialize DPOTrainer
    print("Initializing DPOTrainer")
    dpo_trainer = DPOTrainer(
        model,
        ref_model=None, # Setting ref_model=None will create a reference model copy for LoRA DPO
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config, # Pass the LoraConfig for DPO LoRA
        processing_class=tokenizer,
    )

    print("Model parameters before DPO training:")
    # Access the underlying model if using PEFT
    model_to_print = dpo_trainer.model.model if isinstance(dpo_trainer.model, PeftModel) else dpo_trainer.model
    print(model_to_print)
    print_trainable_parameters(dpo_trainer.model) # Print params of the potentially PEFT-wrapped model

    # Run baseline evaluation first (Optional, DPO eval needs careful interpretation)
    # print("Running baseline evaluation (DPO)...")
    # baseline_metrics = dpo_trainer.evaluate()
    # dpo_trainer.log_metrics("eval_baseline", baseline_metrics)
    # dpo_trainer.save_metrics("eval_baseline", baseline_metrics)

    # 8. Train
    print(
        f"Starting DPO training ({'LoRA enabled' if cfg.lora.enabled else 'Full fine-tuning'})..."
    )
    if cfg.training.num_train_epochs > 0:
        train_result = dpo_trainer.train()
        metrics = train_result.metrics
        dpo_trainer.log_metrics("train", metrics)
        dpo_trainer.save_metrics("train", metrics)
        dpo_trainer.save_state() # Save optimizer state etc.
    else:
        print("Skipping DPO training as num_train_epochs <= 0")

    # 9. Save final model/adapter
    print(
        f"Saving final DPO model ({'LoRA adapter' if cfg.lora.enabled else 'Full weights'})..."
    )
    save_dir = os.path.join(
        output_dir, "dpo_lora_adapter" if cfg.lora.enabled else "dpo_final_model"
    )
    dpo_trainer.save_model(save_dir) # Saves adapter if PEFT config was used, otherwise full model
    print(f"Model saved to {save_dir}")
    if cfg.wandb.save_files:
        wandb.save(os.path.join(save_dir, "*"), base_path=output_dir)


    # 10. Merge and save (Optional, only if LoRA was enabled)
    if cfg.lora.enabled:
        print("Merging DPO LoRA adapter...")
        # Ensure model is on CPU before merging, especially if using device_map
        # dpo_trainer.model.cpu() # May need adjustment depending on setup
        try:
            merged_model = dpo_trainer.model.merge_and_unload()
            merged_dir = os.path.join(output_dir, "dpo_lora_merged")
            merged_model.save_pretrained(merged_dir)
            tokenizer.save_pretrained(merged_dir) # Save tokenizer with merged model
            print(f"Finished saving the merged DPO model to: {merged_dir}")
            if cfg.wandb.save_files:
                wandb.save(os.path.join(merged_dir, "*"), base_path=output_dir)
        except Exception as e:
            print(f"Error during merging and saving: {e}")
            print("Skipping merge and save due to error.")

    print("DPO Training finished.")
    wandb.finish()


if __name__ == "__main__":
    main()

