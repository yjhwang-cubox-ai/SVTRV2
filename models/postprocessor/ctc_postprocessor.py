import torch
import math
from typing import Dict, List, Tuple

class CTCPostProcessor:
    """CTC 디코딩을 위한 후처리 클래스"""
    
    def __init__(self, dictionary):
        """
        Args:
            dictionary: 문자 사전
            padding_idx: 패딩 인덱스
            ignore_indexes: 무시할 인덱스들의 튜플
        """
        self.dictionary = dictionary
        self.padding_idx = dictionary.padding_idx
        self.ignore_indexes = {idx for idx in [
                                dictionary.padding_idx,
                                dictionary.unknown_idx,
                                dictionary.start_idx,
                                dictionary.end_idx
                            ] if idx is not None}

        self.unknown_idx = dictionary.unknown_idx

    def get_single_prediction(self, 
                            max_values: torch.Tensor,
                            max_indices: torch.Tensor
                            ) -> Tuple[list, list]:
        
        max_indices = max_indices.numpy()
        max_values = max_values.numpy()
        
        # 초기 크기로 리스트 할당 (append 연산 줄이기)
        index, score = [], []

        # CTC 디코딩 (중복 제거)
        prev_idx = self.padding_idx
        for t in range(len(max_indices)):
            cur_idx = max_indices[t]
            # 이전 인덱스와 다르고 무시할 인덱스가 아닌 경우만 저장
            if cur_idx not in self.ignore_indexes and cur_idx != prev_idx:
                index.append(cur_idx)
                score.append(max_values[t])
            prev_idx = cur_idx

        return index, score

    def __call__(self, outputs: torch.Tensor) -> List[Dict]:
        if outputs.is_cuda:
            outputs = outputs.cpu()
        outputs = outputs.detach()
        batch_size = outputs.size(0)
        
        # 배치 단위로 max 연산 수행
        max_values, max_indices = torch.max(outputs, dim=-1)  # (B, T)
        
        results = []
        for i in range(batch_size):
            index, score = self.get_single_prediction(
                max_values[i], max_indices[i])  # max 연산 제거
            
            text = self.dictionary.idx2str(index)
            results.append({
                'text': text,
                'score': score,
                'index': index
            })

        return results