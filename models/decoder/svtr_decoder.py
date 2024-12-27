import torch
import torch.nn as nn
from typing import Optional, Dict, Union
from utils.dictionary import Dictionary

class SVTRDecoder(nn.Module):
    """Decoder module in SVTR (Scene Text Recognition with a Single Visual Model).

    Paper reference: https://arxiv.org/abs/2205.00159

    Args:
        in_channels (int): The number of input channels.
        dictionary (Union[Dict, Dictionary]): Dictionary instance or config.
        max_seq_len (int, optional): Maximum output sequence length. Defaults to 25.
    """

    def __init__(self,
                 in_channels: int,
                 dictionary: Union[Dict, 'Dictionary'],
                 max_seq_len: int = 25):
        super().__init__()
        
        self.dictionary = dictionary
        self.max_seq_len = max_seq_len
        
        # Linear projection to vocabulary size
        self.decoder = nn.Linear(
            in_features=in_channels,
            out_features=self.dictionary.num_classes)
        
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, 
                # feat: Optional[torch.Tensor] = None,
                out_enc: Optional[torch.Tensor] = None,
                training: bool = True) -> torch.Tensor:
        """Forward function.

        Args:
            feat (torch.Tensor, optional): The feature map. Defaults to None.
            out_enc (torch.Tensor, optional): Encoder output of shape (N, C, 1, W).
                Defaults to None.
            training (bool, optional): Whether in training mode. Defaults to True.

        Returns:
            torch.Tensor: If training is True, returns raw logits of shape (N, T, C).
                         If training is False, returns probabilities of shape (N, T, C).
                         where T is sequence length and C is number of classes.

        Raises:
            AssertionError: If the height of out_enc is not 1.
        """
        assert out_enc.size(2) == 1, 'Feature height must be 1'
        
        # Shape: (N, C, W) -> (N, W, C)
        x = out_enc.squeeze(2).permute(0, 2, 1)
        
        # Project to vocabulary size
        logits = self.decoder(x)
        
        # Return logits during training, probabilities during inference
        return logits if training else self.softmax(logits)