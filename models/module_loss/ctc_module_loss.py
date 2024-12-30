import math
from typing import Dict, Sequence, Union

import torch
import torch.nn as nn

from data.dictionary import Dictionary

class CTCModuleLoss(nn.Module):
    def __init__(
            self,
            dictionary: Union[Dict, Dictionary],
            letter_case: str = 'unchanged',
            flatten: bool = True,
            reduction: str = 'mean',
            zero_infinity: bool = True) -> None:
        
        super().__init__()
        assert isinstance(flatten, bool)
        assert isinstance(reduction, str)
        assert isinstance(zero_infinity, bool)
        assert letter_case in ['unchanged', 'upper', 'lower']

        self.dictionary = dictionary
        self.letter_case = letter_case
        self.flatten = flatten
        self.ctc_loss = nn.CTCLoss(
            blank=self.dictionary.padding_idx,
            reduction=reduction,
            zero_infinity=zero_infinity
        )

    def forward(self, outputs: torch.Tensor, gt_texts: list)-> Dict[str, torch.Tensor]:
        # Log softmax for CTC loss
        log_probs = torch.log_softmax(outputs, dim=2)
        batch_size, seq_len = outputs.size(0), outputs.size(1)
        
        # Permute for CTC loss (T, N, C)
        log_probs_for_loss = log_probs.permute(1, 0, 2).contiguous()
        
        texts_indexes = self.get_targets(gt_texts)
        targets = [
            indexes[:seq_len]
            for indexes in texts_indexes
        ]
        target_lengths = torch.IntTensor([len(t) for t in targets])
        target_lengths = torch.clamp(target_lengths, max=seq_len).long()
        input_lengths = torch.full(
            size=(batch_size, ), fill_value=seq_len, dtype=torch.long)
        if self.flatten:
            targets = torch.cat(targets)
        else:
            padded_targets = torch.full(
                size=(batch_size, seq_len),
                fill_value=self.dictionary.padding_idx,
                dtype=torch.long)
            for idx, valid_len in enumerate(target_lengths):
                padded_targets[idx, :valid_len] = targets[idx][:valid_len]
            targets = padded_targets
        loss_ctc = self.ctc_loss(log_probs_for_loss, targets, input_lengths,
                                 target_lengths)
        return loss_ctc
    
    def get_targets(self, texts:list):
        indexes_list = []
        for text in texts:
            if self.letter_case in ['upper', 'lower']:
                text = getattr(text, self.letter_case)()
            indexes = self.dictionary.str2idx(text)
            indexes = torch.IntTensor(indexes)
            indexes_list.append(indexes)
        return indexes_list