import albumentations as A
from albumentations.pytorch import ToTensorV2

class TextRecogAugmentations:
    def __init__(self, is_train=True):
        if is_train:
            self.transform = A.Compose([
                # TextRecogGeneralAug
                A.OneOf([
                    A.MotionBlur(blur_limit=3, p=0.5),
                    A.MedianBlur(blur_limit=3, p=0.5),
                    A.GaussianBlur(blur_limit=(3,5), p=0.5),
                ], p=1.0),

                # CropHeight
                A.RandomResizedCrop(
                    height=64,
                    width=256,
                    scale=(0.8, 1.0),
                    ratio=(2.0, 4.0),
                    p=0.1
                ),

                # GaussianBlur
                A.GaussianBlur(
                    blur_limit=(5, 5),
                    sigma_limit=(1, 1),
                    p=1.0
                ),
                
                # ColorJitter
                A.ColorJitter(
                    brightness=0.5,
                    contrast=0.5,
                    saturation=0.5,
                    hue=0.1,
                    p=1.0
                ),
                
                # ImageContentJitter와 유사한 효과
                A.OneOf([
                    A.RandomBrightnessContrast(p=1),
                    A.RandomGamma(p=1),
                ], p=1.0),
                
                # AdditiveGaussianNoise
                A.GaussNoise(
                    var_limit=(0.1**0.5, 0.1**0.5),
                    p=1.0
                ),
                
                # ReversePixels와 유사한 효과
                A.InvertImg(p=1.0),
                
                # Final resize
                A.Resize(height=64, width=256),
                
                # Convert to tensor
                ToTensorV2()
            ])
        else:
            self.transform = A.Compose([
                A.Resize(height=64, width=256),
                ToTensorV2()
            ])

    def __call__(self, image):
        transformed = self.transform(image=image)
        return transformed['image']