import torch

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
        eos_id   = tokenizer.eos_token_id
        pad_id   = tokenizer.pad_token_id

        # 1) Tokenize **without** any padding so we get raw lists of IDs
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
        pre_ids = pre_enc["input_ids"]   # List[List[int]]
        suf_ids = suf_enc["input_ids"]    # List[List[int]]

        # 2) Figure out how long each final sequence will be (prefix+suffix+EOS)
        lengths = [len(p) + len(s) + 1 for p, s in zip(pre_ids, suf_ids)]
        max_len = max(lengths)

        # 3) Allocate tensors of shape (B, max_len)
        input_ids      = torch.full((B, max_len), pad_id,   dtype=torch.long)
        attention_mask = torch.zeros((B, max_len),          dtype=torch.long)
        labels         = torch.full((B, max_len), IGNORE_LABEL, dtype=torch.long)

        # 4) Fill them in example by example
        for i, (p, s, L) in enumerate(zip(pre_ids, suf_ids, lengths)):
            pre_len = len(p)
            suf_len = len(s)
            # — inputs: [ prefix | suffix | EOS ] then pad
            input_ids[i, :pre_len]                = torch.tensor(p, dtype=torch.long)
            input_ids[i, pre_len : pre_len+suf_len] = torch.tensor(s, dtype=torch.long)
            input_ids[i, pre_len+suf_len]         = eos_id

            # — attention mask through the EOS
            attention_mask[i, :L] = 1

            # — labels: ignore-prefix, then [ suffix | EOS ], then ignore-pad
            labels[i, pre_len : pre_len+suf_len] = torch.tensor(s, dtype=torch.long)
            labels[i, pre_len+suf_len]           = eos_id
            # (everything else in `labels` stays at IGNORE_LABEL)

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
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

