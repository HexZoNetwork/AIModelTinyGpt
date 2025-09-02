# dataset.py
import torch
from torch.utils.data import Dataset
from tokenizers.implementations import ByteLevelBPETokenizer
import os
import glob

class TextDataset(Dataset):
    def __init__(self, files=None, tokenizer_path="tokenizer", seq_len=128):
        self.seq_len = seq_len
        self.ids = []

        # cari semua file part kalau files=None
        if files is None:
            files = sorted(glob.glob("data/wiki_part*.txt"))
            if os.path.exists("data/wiki.txt"):
                files = ["data/wiki.txt"] + files

        # load tokenizer
        tok = ByteLevelBPETokenizer(
            os.path.join(tokenizer_path, "vocab.json"),
            os.path.join(tokenizer_path, "merges.txt")
        )

        # load semua data
        for f in files:
            with open(f, "r", encoding="utf-8") as fh:
                text = fh.read()
                self.ids.extend(tok.encode(text).ids)
            print(f"[LOAD] {f} ({len(text)} chars)")

    def __len__(self):
        return max(1, len(self.ids) // self.seq_len - 1)

    def __getitem__(self, idx):
        i = idx * self.seq_len
        x = torch.tensor(self.ids[i:i+self.seq_len], dtype=torch.long)
        y = torch.tensor(self.ids[i+1:i+1+self.seq_len], dtype=torch.long)
        return x, y
