import argparse
import torch
from sft_224r_dataloader import get_sft_dataloader, IGNORE_LABEL


def debug_dataloader(model_name, dataset_name, split, max_seq_len, batch_size, num_batches=1):
    """
    Iterate through the SFT dataloader and print each batch and example for debugging.
    """
    dataloader = get_sft_dataloader(
        model_name_or_path=model_name,
        dataset_name=dataset_name,
        split=split,
        max_seq_length=max_seq_len,
        batch_size=batch_size,
    )
    tokenizer = dataloader.dataset.tokenizer

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
        print(f"\n=== Batch {batch_idx} ===")

        # Convert to tensors for easy slicing
        input_ids = torch.tensor(batch["input_ids"])
        attention_mask = torch.tensor(batch["attention_mask"])
        labels = torch.tensor(batch["labels"])

        # Batch dimensions
        bsz, seq_len = input_ids.shape
        print(f"Batch size: {bsz}, Sequence length: {seq_len}")

        for i in range(bsz):
            ids = input_ids[i].tolist()
            mask = attention_mask[i].tolist()
            lbl = labels[i].tolist()

            # Decode full sequence (includes user + assistant prompts)
            decoded = tokenizer.decode(ids, skip_special_tokens=False)

            print(f"\n-- Sample {i} --")
            print(f"Input IDs ({len(ids)}): {ids}")
            print(f"Attention Mask ({len(mask)}): {mask}")
            print(f"Labels ({len(lbl)}): {lbl}")
            print(f"Decoded Text: {decoded}")

            # Optionally, show target portion where labels != IGNORE_LABEL
            target_positions = [j for j, token in enumerate(lbl) if token != IGNORE_LABEL]
            print(f"Target positions (labels != {IGNORE_LABEL}): {target_positions}")
        print("\n" + "="*50)

    print("\nFinished debugging dataloader.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Debug SFT dataloader steps")
    parser.add_argument("--model_name", default="EleutherAI/pythia-2.8b")
    parser.add_argument("--dataset_name", default="Anthropic/hh-rlhf")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_batches", type=int, default=1,
                        help="Number of batches to display")
    args = parser.parse_args()

    debug_dataloader(
        args.model_name,
        args.dataset_name,
        args.split,
        args.max_seq_len,
        args.batch_size,
        args.num_batches,
    )
