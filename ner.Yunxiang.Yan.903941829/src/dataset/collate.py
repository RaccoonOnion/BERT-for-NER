"""
# Author: Yinghao Li
# Modified: September 30th, 2023
# ---------------------------------------
# Description: collate function for batch processing
"""

import torch
from transformers import DataCollatorForTokenClassification

from .batch import unpack_instances, Batch


class DataCollator(DataCollatorForTokenClassification):
    def __call__(self, instance_list: list[dict]):
        tk_ids, attn_masks, lbs = unpack_instances(instance_list, ["bert_tk_ids", "bert_attn_masks", "bert_lbs"])
        # Update `tk_ids`, `attn_masks`, and `lbs` to match the maximum length of the batch.
        # The updated type of the three variables should be `torch.int64``.
        # Hint: some functions and variables you may want to use: `self.tokenizer.pad()`, `self.label_pad_token_id`.

        # --- TODO: start of your code ---
        # First, let's pad just the `input_ids` and `attention_mask`
        padded_data = self.tokenizer.pad({
            "input_ids": tk_ids,
            "attention_mask": attn_masks
        }, padding="longest", return_tensors="pt")

        tk_ids = padded_data["input_ids"].to(torch.int64)
        attn_masks = padded_data["attention_mask"].to(torch.int64)

        # Now, let's handle the `lbs` separately
        max_length = tk_ids.size(1)  # Get the length of the longest sequence after padding
        padded_lbs = []
        for label_sequence in lbs:
            # Pad or truncate the label sequence to match max_length
            padded_sequence = label_sequence[:max_length] + [self.label_pad_token_id] * (
                        max_length - len(label_sequence))
            padded_lbs.append(padded_sequence)

        lbs = torch.tensor(padded_lbs, dtype=torch.int64)
        # --- TODO: end of your code ---

        return Batch(input_ids=tk_ids, attention_mask=attn_masks, labels=lbs)
