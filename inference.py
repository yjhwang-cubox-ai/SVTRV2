import torch
from utils.inferencer import SVTRInferencer

def main():
    inferencer = SVTRInferencer(checkpoint_path='svtr-Lao-ID-test/yxn51xzu/checkpoints/best-checkpoint.ckpt',
                                dict_path='dicts/lao_dict.txt',
                                device=('cuda' if torch.cuda.is_available() else 'cpu'))
    
    image = 'laotest/01-23079071_Real_00000000.jpg'

    result = inferencer.predict(image)

    print(result)

if __name__ == '__main__':    
    main()