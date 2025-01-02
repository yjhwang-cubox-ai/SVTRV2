from typing import List
import torch
import unicodedata
import json
import os
import ast
from tqdm import tqdm
from utils.metric import OneMinusNEDMetric, WordMetric, CharMetric
from utils.inferencer import SVTRInferencer

def read_json(json_files: List[str]):
    annotation_info = []
    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as file:
            img_dir_path = os.path.dirname(json_file)
            anno_list = json.load(file)
            annotation_info.extend(
                [{"img_path": os.path.join(img_dir_path, anno['img_path']), "text": anno['instances'][0]['text']} for anno in anno_list['data_list']]
            )
        
    return annotation_info

def lao_text_postprocess(pred_text):
    text = unicodedata.normalize("NFKC", pred_text)
    text = text.replace('ເເ','ແ')
    text = text.replace('ໍໍ', 'ໍ')  # 연속된 ໍ를 하나로
    text = text.replace('້ໍ', 'ໍ້')  # ້ໍ를 ໍ້로 변경
    text = text.replace('ຫນ','ໜ')
    text = text.replace('ຫມ','ໝ')
    
    return text

def main():
    inferencer = SVTRInferencer(checkpoint_path='svtr-Lao-ID-test/yxn51xzu/checkpoints/best-checkpoint.ckpt',
                                dict_path='dicts/lao_dict.txt',
                                device=('cuda' if torch.cuda.is_available() else 'cpu'))
    
    annotation_info = read_json(['/data/LaoTestset/Lao_1202_TEST/annotation_new.json'])
    image_paths = [info['img_path'] for info in annotation_info]

    infer_results = []
    for anno_info in tqdm(annotation_info):
        pred_text = lao_text_postprocess(inferencer.predict(image_path=anno_info['img_path'])['text'])
        gt_text = lao_text_postprocess(anno_info['text'])
        infer_results.append({
                "img": anno_info['img_path'], 
                "gt_text": gt_text, 
                "pred_text": pred_text, 
                "match": gt_text == pred_text
            })
    
    with open("log.txt", "w", encoding="utf-8") as f:
         for result in infer_results:
            f.write(f"{result}\n")
    
    # with open("log.txt", "r", encoding="utf-8") as f:
    #     results = [ast.literal_eval(result) for result in f.readlines()] 

    one_minus_ned = OneMinusNEDMetric()
    one_mainus_ned_results = one_minus_ned.compute_metrics(infer_results)
    
    word_metric = WordMetric()
    word_metric_results = word_metric.compute_metrics(infer_results)
    
    char_metric = CharMetric()
    char_metric_results = char_metric.compute_metrics(infer_results)
    
    print(one_mainus_ned_results)
    print(word_metric_results)
    print(char_metric_results)



if __name__ == "__main__":
    main()
