import torch
import torch.nn as nn
from PIL import Image
import numpy as np
from typing import Dict
from models.svtr_module import SVTR, SVTRModel
from data.dictionary import Dictionary
from models.postprocessor.ctc_postprocessor import CTCPostProcessor


class SVTRInferencer:
    def __init__(self, checkpoint_path: str, dict_path: str, device: str):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(checkpoint_path=checkpoint_path)
        self.dictionary = Dictionary(dict_file=dict_path)
        self.postprocessor = CTCPostProcessor(dictionary=self.dictionary)
        self.softmax = nn.Softmax(dim=-1)
    
    def _load_model(self, checkpoint_path: str) -> SVTRModel:
        model = SVTR.load_from_checkpoint(checkpoint_path=checkpoint_path)
        model.eval()
        model.to(self.device)
        return model.model

    def preprocess(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path).convert('RGB')
        image = image.resize((256, 64))
        image = np.array(image)
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        image = image.unsqueeze(0)
        image = (image - 127.5) / 127.5
        return image.to(self.device)
    
    def predict(self, image_path: str) -> Dict:
        image_tensor = self.preprocess(image_path)

        with torch.no_grad():
            output = self.model(image_tensor)
            output = self.softmax(output)
            result = self.postprocessor(outputs=output)
        
        return result[0]

        