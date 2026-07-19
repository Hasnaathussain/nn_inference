#!/usr/bin/env python3
import re
from pathlib import Path

import numpy as np


class BertTokenizer:
    def __init__(self, vocab_path):
        tokens = Path(vocab_path).read_text(encoding="utf-8").splitlines()
        self.vocab = {token: index for index, token in enumerate(tokens)}
        self.unk_id = self.vocab["[UNK]"]
        self.cls_id = self.vocab["[CLS]"]
        self.sep_id = self.vocab["[SEP]"]
        self.pad_id = self.vocab["[PAD]"]

    def wordpieces(self, token):
        if token in self.vocab:
            return [self.vocab[token]]
        if len(token) > 100:
            return [self.unk_id]
        pieces = []
        start = 0
        while start < len(token):
            end = len(token)
            found = None
            while end > start:
                piece = token[start:end]
                if start != 0:
                    piece = "##" + piece
                if piece in self.vocab:
                    found = self.vocab[piece]
                    break
                end -= 1
            if found is None:
                return [self.unk_id]
            pieces.append(found)
            start = end
        return pieces

    def encode(self, sentence, max_length=16):
        raw_tokens = re.findall(r"\[MASK\]|[A-Za-z0-9]+|[^\w\s]", sentence)
        token_ids = [self.cls_id]
        for raw in raw_tokens:
            token = raw if raw == "[MASK]" else raw.lower()
            token_ids.extend(self.wordpieces(token))
        token_ids.append(self.sep_id)
        token_ids = token_ids[:max_length]
        if token_ids[-1] != self.sep_id:
            token_ids[-1] = self.sep_id
        attention = [1] * len(token_ids)
        padding = max_length - len(token_ids)
        token_ids.extend([self.pad_id] * padding)
        attention.extend([0] * padding)
        return {
            "input_ids": np.array([token_ids], dtype=np.int64),
            "attention_mask": np.array([attention], dtype=np.int64),
            "token_type_ids": np.zeros((1, max_length), dtype=np.int64),
        }