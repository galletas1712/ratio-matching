from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

IGNORE_LABEL = -100


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
        model_name_or_path: str,
        split: str,
        dataset_name: str,
        max_seq_length: int,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, padding_side="left"
        )

        self.max_seq_length = max_seq_length
        # Load the raw dataset metadata, but don't tokenize yet
        print("Loading dataset:", self.dataset_name)
        self.raw_dataset = load_dataset(self.dataset_name, split=self.split)
        print("Raw dataset loaded:", self.raw_dataset)

    # _load_and_tokenize_dataset is no longer needed here as a separate pre-processing step

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, idx):
        example = self.raw_dataset[idx]
        sample = example["chosen"]
        prefix, suffix = split_prefix_last_assistant(sample)

        input_ids = self.tokenizer(text=prefix+suffix, max_length=self.max_seq_length, truncation=True)["input_ids"]
        prefix_input_ids = self.tokenizer(text=prefix, max_length=self.max_seq_length, truncation=True)["input_ids"]

        # Our SFT supervision signal should only come from the decoded tokens after (last assistant message)
        # Note that we don't pad this in the dataset but we do pad it in the collate_fn
        labels = input_ids[len(prefix_input_ids) :]

        return input_ids, labels


def get_sft_dataloader(
    model_name_or_path: str,
    dataset_name: str,
    split: str,
    max_seq_length: int,
    batch_size: int,
):
    assert split in ["train", "test"], "Split must be 'train' or 'test'."
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="left")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    print(tokenizer.pad_token_id, tokenizer.eos_token_id)
    assert (
        tokenizer.pad_token == tokenizer.eos_token
        ), f"Tokenizer's pad_token should be the same as eos_token, pad_token: {tokenizer.pad_token}, eos_token: {tokenizer.eos_token}"

    def collate_fn(
        batch: list[tuple[list[int], list[int]]],
    ) -> dict[str, list[list[int]]]:
        max_input_len = max(len(x[0]) for x in batch)
        input_ids_padded = [
            [tokenizer.pad_token_id] * (max_input_len - len(x[0])) + x[0] for x in batch
        ]
        attention_mask = [
            [0] * (max_input_len - len(x[0])) + [1] * len(x[0]) for x in batch
        ]
        labels_padded = [
            [IGNORE_LABEL] * (max_input_len - len(x[1])) + x[1] for x in batch
        ]
        return {
            "input_ids": input_ids_padded,
            "attention_mask": attention_mask,
            "labels": labels_padded,
        }

    dataset = SFTDataset(
        model_name_or_path=model_name_or_path,
        split=split,
        dataset_name=dataset_name,
        max_seq_length=max_seq_length,
    )

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    return dataloader

