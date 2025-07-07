"""
Run script for SFT.
"""

import math
import os
import random

import hydra
import torch
import wandb
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.optimization import get_linear_schedule_with_warmup

IGNORE_LABEL = -100


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


def split_prefix_last_assistant(conv: str) -> tuple[str, str]:
    """
    Splits a conversation string into:
      - prefix: everything before the last "Assistant:" block
      - last_msg: the content of that last assistant message

    It looks for the last occurrence of "Assistant:", then—if there is a following "Human:"—
    stops the assistant message there.
    """
    marker = "Assistant:"
    # find start of last Assistant:
    idx = conv.rfind(marker)
    if idx == -1:
        raise ValueError("No 'Assistant:' speaker found in the input.")

    # compute where the assistant message text actually starts
    start = idx + len(marker)
    # see if there's a next Human: after that
    next_human = conv.find("Human:", start)

    if next_human != -1:
        # there's another Human message -> assistant text ends just before it
        last_msg = conv[start:next_human].strip()
    else:
        # no following Human: -> take everything to the end
        last_msg = conv[start:].strip()

    # prefix is everything up to (but not including) that last "Assistant:"
    prefix = conv[:idx].rstrip()
    last_msg = "Assistant: " + last_msg

    return prefix, last_msg


class SFTDataset(Dataset):
    def __init__(
        self,
        split: str,
        dataset_name: str,
        max_seq_length: int,
    ):
        self.dataset_name = dataset_name
        self.split = split

        self.max_seq_length = max_seq_length
        # Load the raw dataset metadata, but don't tokenize yet
        print("Loading dataset:", self.dataset_name)
        self.raw_dataset = load_dataset(self.dataset_name, split=self.split)
        print("Raw dataset loaded:", self.raw_dataset)

    # _load_and_tokenize_dataset is no longer needed here as a separate pre-processing step

    def __len__(self):
        return len(self.raw_dataset)  # type: ignore

    def __getitem__(self, idx):
        # assert isinstance(self.raw_dataset, Dataset)
        example = self.raw_dataset[idx]
        # For SFT, we expect a 'chosen' column with the full conversation
        sample = example["chosen"]
        assert isinstance(sample, str), f"Expected string, got {type(sample)}"
        # We train on the final assistant turn
        prefix, suffix = split_prefix_last_assistant(sample)

        return prefix, suffix


def get_sft_dataloader(
    model_name_or_path: str,
    dataset_name: str,
    split: str,
    max_seq_length: int,
    batch_size: int,
):
    assert split in ["train", "test"], "Split must be 'train' or 'test'."
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    print(tokenizer.pad_token_id, tokenizer.eos_token_id)
    assert (
        tokenizer.pad_token == tokenizer.eos_token
    ), f"Tokenizer's pad_token should be the same as eos_token, pad_token: {tokenizer.pad_token}, eos_token: {tokenizer.eos_token}"

    def collate_fn(batch: list[tuple[str, str]]):
        prefixes, suffixes = zip(*batch)
        B = len(batch)
        eos_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id

        # Tokenize **without** any padding so we get raw lists of IDs
        pre_enc = tokenizer(
            list(prefixes),
            padding=False,
            truncation=True,
            max_length=max_seq_length,
            return_tensors=None,
        )
        suf_enc = tokenizer(
            list(suffixes),
            padding=False,
            truncation=True,
            max_length=max_seq_length,
            return_tensors=None,
        )
        pre_ids = pre_enc["input_ids"]  # List[List[int]]
        suf_ids = suf_enc["input_ids"]  # List[List[int]]

        # Figure out how long each final sequence will be (prefix+suffix+EOS)
        lengths = [len(p) + len(s) + 1 for p, s in zip(pre_ids, suf_ids)]
        max_len = max(lengths)

        # Allocate tensors of shape (B, max_len)
        input_ids = torch.full((B, max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((B, max_len), dtype=torch.long)
        labels = torch.full((B, max_len), IGNORE_LABEL, dtype=torch.long)

        # Fill them in example by example
        for i, (p, s, L) in enumerate(zip(pre_ids, suf_ids, lengths)):
            pre_len = len(p)
            suf_len = len(s)
            # — inputs: [ prefix | suffix | EOS ] then pad
            input_ids[i, :pre_len] = torch.tensor(p, dtype=torch.long)
            input_ids[i, pre_len : pre_len + suf_len] = torch.tensor(
                s, dtype=torch.long
            )
            input_ids[i, pre_len + suf_len] = eos_id

            # attention mask through the EOS
            attention_mask[i, :L] = 1

            # labels: ignore-prefix, then [ suffix | EOS ], then ignore-pad
            labels[i, pre_len : pre_len + suf_len] = torch.tensor(s, dtype=torch.long)
            labels[i, pre_len + suf_len] = eos_id
            # everything else in `labels` stays at IGNORE_LABEL

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    dataset = SFTDataset(
        split=split,
        dataset_name=dataset_name,
        max_seq_length=max_seq_length,
    )

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    return dataloader


def eval(model, dataloader, device, eval_step_cap):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for step, data in enumerate(dataloader):
            if step >= eval_step_cap:
                break
            input_ids = torch.tensor(data["input_ids"], dtype=torch.long, device=device)
            labels = torch.tensor(data["labels"], dtype=torch.long, device=device)
            attention_mask = torch.tensor(
                data["attention_mask"], dtype=torch.long, device=device
            )
            outputs = model(
                input_ids=input_ids, labels=labels, attention_mask=attention_mask
            )
            total_loss += outputs.loss.item()
    return total_loss / min(len(dataloader), eval_step_cap)


@hydra.main(version_base=None, config_path="config", config_name="sft_config")
def main(cfg: DictConfig) -> None:
    """Main SFT training function driven by Hydra configuration."""
    # 1. Initialize W&B
    wandb.init(
        project=cfg.wandb.project,
        name=cfg.wandb.run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
        save_code=True,
    )

    # Set output directory based on WandB run
    output_dir = wandb.run.dir
    cfg.training.output_dir = output_dir
    print(f"Output directory: {output_dir}")
    print("-------------------- Configuration (SFT) --------------------")
    print(OmegaConf.to_yaml(cfg))
    print("-------------------------------------------------------------")

    # 2. Seed for reproducibility
    random.seed(cfg.training.seed)
    torch.manual_seed(cfg.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.training.seed)

    # 3. Load model and tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model: {cfg.model.name} on device: {device}")
    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16 if cfg.training.bf16 else torch.float32,
    }
    model = AutoModelForCausalLM.from_pretrained(cfg.model.name, **model_kwargs).to(
        device
    )

    # 4. Configure LoRA if enabled
    if cfg.lora.enabled:
        print("Applying LoRA configuration...")
        lora_config = LoraConfig(
            r=cfg.lora.r,
            lora_alpha=cfg.lora.lora_alpha,
            lora_dropout=cfg.lora.lora_dropout,
            bias=cfg.lora.bias,
            task_type=cfg.lora.task_type,
            target_modules=list(cfg.model.lora_target_modules),
        )
        model = get_peft_model(model, lora_config)
        print_trainable_parameters(model)

    # 5. Load datasets
    print("Loading datasets...")
    train_dataloader = get_sft_dataloader(
        model_name_or_path=cfg.model.name,
        dataset_name=cfg.dataset.name,
        split="train",
        max_seq_length=cfg.model.max_seq_len,
        batch_size=cfg.training.per_device_train_batch_size,
    )
    test_dataloader = get_sft_dataloader(
        model_name_or_path=cfg.model.name,
        dataset_name=cfg.dataset.name,
        split="test",
        max_seq_length=cfg.model.max_seq_len,
        batch_size=cfg.training.per_device_eval_batch_size,
    )

    # 6. Setup optimizer and scheduler
    steps_per_epoch = math.ceil(
        len(train_dataloader) / cfg.training.gradient_accumulation_steps
    )
    total_steps = steps_per_epoch * cfg.training.num_train_epochs
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.training.warmup_steps,
        num_training_steps=total_steps,
    )

    # 7. Initial evaluation
    print("Running initial evaluation...")
    init_loss = eval(model, test_dataloader, device, 200)  # Capped eval steps
    print(f"Initial Eval Loss: {init_loss:.4f}")
    wandb.log({"eval/initial_loss": init_loss})

    # 8. Training loop
    global_step = 0
    for epoch in range(cfg.training.num_train_epochs):
        model.train()
        for step, data in enumerate(train_dataloader):
            input_ids = data["input_ids"].to(device)
            labels = data["labels"].to(device)
            attention_mask = data["attention_mask"].to(device)

            outputs = model(
                input_ids=input_ids, labels=labels, attention_mask=attention_mask
            )
            loss = outputs.loss / cfg.training.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % cfg.training.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.training.max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                # Logging
                if global_step % cfg.training.logging_steps == 0:
                    print(
                        f"Epoch {epoch} | Step {global_step} | Loss {loss.item() * cfg.training.gradient_accumulation_steps:.4f}"
                    )
                    wandb.log(
                        {
                            "train/loss": loss.item()
                            * cfg.training.gradient_accumulation_steps,
                            "train/lr": scheduler.get_last_lr()[0],
                            "train/step": global_step,
                            "epoch": epoch,
                        }
                    )

                # Evaluation
                if global_step % cfg.training.eval_steps == 0:
                    model.eval()
                    val_loss = eval(model, test_dataloader, device, 200)  # Capped
                    print(f"--> Eval Loss @ step {global_step}: {val_loss:.4f}")
                    wandb.log({"eval/loss": val_loss, "train/step": global_step})
                    model.train()

                # Checkpoint saving
                if global_step % cfg.training.save_steps == 0:
                    ckpt_path = os.path.join(output_dir, f"step-{global_step}")
                    os.makedirs(ckpt_path, exist_ok=True)
                    # Save model adapter
                    model.save_pretrained(ckpt_path)

                    print(f"Checkpoint saved to {ckpt_path}")
                    if cfg.wandb.save_files:
                        art = wandb.Artifact(f"model-step-{global_step}", type="model")
                        art.add_dir(ckpt_path)
                        wandb.log_artifact(art)

    # Final save
    final_path = os.path.join(output_dir, "final_checkpoint")
    model.save_pretrained(final_path)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    tokenizer.save_pretrained(final_path)
    print(f"Final model saved to {final_path}")

    if wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
