from data.augmentation import TextRecogAugmentations
import json
import os
import cv2
from torch.utils.data import Dataset, DataLoader

class LaoDataset(Dataset):
    def __init__(self, annotation_files: list, is_train: bool = True):
        super().__init__()
        self.annotation_files = annotation_files
        self.transform = TextRecogAugmentations(is_train=is_train)
        self.data_list = []
        
        # annotation 파일 읽기 - json
        for json_file in annotation_files:
            with open(json_file, 'r', encoding='utf-8') as f:
                data_info = json.load(f)['data_list']
            for data in data_info:
                data['img_path'] = os.path.join(os.path.dirname(json_file), data['img_path'])
            self.data_list.extend(data_info)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img_path = self.data_list[idx]['img_path']
        text = self.data_list[idx]['instances'][0]['text']
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image)
            
        return {
            'img': image,
            'label': text
        }