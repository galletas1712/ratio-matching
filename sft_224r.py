import argparse
import math
import random
import os
from datetime import datetime

import torch
from transformers import (
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
)

import wandb
from sft_224r_dataloader import get_sft_dataloader


def parse_args():
    parser = argparse.ArgumentParser("Supervised fine-tune a causal-LM")
    parser.add_argument("--model_name", default="EleutherAI/pythia-2.8b")
    parser.add_argument("--dataset_name", default="Anthropic/hh-rlhf")
    parser.add_argument("--output_base_dir", default="./outputs")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument(
        "--grad_accum", type=int, default=256,
        help="Gradient accumulation steps (effective batch size = batch_size * grad_accum)"
    )
    parser.add_argument("--max_grad_norm", type=float, default=15.0)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--eval_step_cap", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=20000)
    parser.add_argument("--device", type=str, default="cuda")
    # Resume from checkpoint
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Directory containing a saved checkpoint to resume from"
    )
    # WandB related arguments
    parser.add_argument("--wandb_project", type=str, default="sft-lora")
    parser.add_argument(
        "--wandb_entity", type=str, default="rsanda-stanford-university"
    )
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, nargs="+", default=None)
    parser.add_argument("--wandb_disable", action="store_true")
    return parser.parse_args()


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
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask
            )
            total_loss += outputs.loss.item()
    return total_loss / min(len(dataloader), eval_step_cap)


def main():
    args = parse_args()
    # Reproducibility
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # WandB initialization
    if not args.wandb_disable:
        run_name = (
            args.wandb_run_name or
            f"sft-{os.path.basename(args.model_name)}-{datetime.now():%Y%m%d_%H%M%S}"
        )
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            tags=args.wandb_tags,
            config=vars(args),
        )

    # Model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, trust_remote_code=True
    ).to(args.device)

    # Data
    train_dataloader = get_sft_dataloader(
        args.model_name,
        args.dataset_name,
        "train",
        args.max_seq_len,
        args.batch_size,
    )
    test_dataloader = get_sft_dataloader(
        args.model_name,
        args.dataset_name,
        "test",
        args.max_seq_len,
        args.batch_size,
    )

    # Scheduler & optimizer setup
    steps_per_epoch = math.ceil(len(train_dataloader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)

    # Separate decay vs no-decay params
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if any(nd in name for nd in ["bias", "LayerNorm", "layernorm", "ln_", "norm", "embeddings"]):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.lr
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Resume logic
    start_epoch = 0
    resume_step = -1
    global_step = 0
    if args.resume_from_checkpoint:
        ckpt_dir = args.resume_from_checkpoint
        print(f"[INFO] Resuming from checkpoint {ckpt_dir}")
        # reload model weights
        model = AutoModelForCausalLM.from_pretrained(ckpt_dir, trust_remote_code=True).to(args.device)
        # load training state
        state = torch.load(os.path.join(ckpt_dir, "training_state.pt"), map_location=args.device)
        optimizer.load_state_dict(state["optimizer_state_dict"])
        scheduler.load_state_dict(state["scheduler_state_dict"])
        start_epoch = state["epoch"]
        resume_step = state["step"]
        global_step = state["global_step"]

    # Initial evaluation (only on fresh runs)
    if not args.resume_from_checkpoint:
        init_loss = eval(model, test_dataloader, args.device, args.eval_step_cap)
        print(f"Initial Eval Loss: {init_loss:.4f}")
        if not args.wandb_disable:
            wandb.log({"eval/initial_loss": init_loss})

    # Output directory
    if args.resume_from_checkpoint:
        output_dir = args.resume_from_checkpoint
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = os.path.join(args.output_base_dir, timestamp)
        os.makedirs(output_dir, exist_ok=True)

    # Training loop
    for epoch in range(start_epoch, args.epochs):
        model.train()
        for step, data in enumerate(train_dataloader):
            # skip already-done steps
            if epoch == start_epoch and step <= resume_step:
                continue

            input_ids = torch.tensor(data["input_ids"], dtype=torch.long, device=args.device)
            labels = torch.tensor(data["labels"], dtype=torch.long, device=args.device)
            attention_mask = torch.tensor(
                data["attention_mask"], dtype=torch.long, device=args.device
            )

            outputs = model(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask
            )
            loss = outputs.loss
            loss.backward()

            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                if not args.wandb_disable:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/step": global_step,
                        "epoch": epoch,
                    })

            print(f"Epoch {epoch} | Step {step} | Loss {loss.item():.4f}")

            # evaluation
            if (step + 1) % args.eval_steps == 0:
                val_loss = eval(model, test_dataloader, args.device, args.eval_step_cap)
                print(f"--> Eval Loss: {val_loss:.4f}")
                if not args.wandb_disable:
                    wandb.log({"eval/loss": val_loss, "train/step": global_step})

            # checkpoint save
            if (step + 1) % args.save_steps == 0:
                ckpt_path = os.path.join(output_dir, f"epoch-{epoch}-step-{step}")
                os.makedirs(ckpt_path, exist_ok=True)
                model.save_pretrained(ckpt_path)
                torch.save({
                    "epoch": epoch,
                    "step": step,
                    "global_step": global_step,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                }, os.path.join(ckpt_path, "training_state.pt"))
                if not args.wandb_disable:
                    art = wandb.Artifact(f"model-epo{epoch}-st{step}", type="model")
                    art.add_dir(ckpt_path)
                    wandb.log_artifact(art)

    # finish wandb
    if "wandb" in globals() and wandb.run is not None:
        wandb.finish()

if __name__ == "__main__":
    main()

