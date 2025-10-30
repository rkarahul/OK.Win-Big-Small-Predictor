# src/01_ocr_extract.py
import os
import cv2
from paddleocr import PaddleOCR
import pandas as pd
import re

ocr = PaddleOCR(use_angle_cls=True, lang='en', use_gpu=True)

def get_ocr_result(image_path):
    result = ocr.ocr(image_path, cls=True, det=True, rec=True)
    lines = []
    if result:
        for line in result:
            for info in line:
                txt = info[1][0].strip().replace('O', '0').replace('.', '')
                lines.append([info[0], txt, info[1][1]])
    return lines

def extract_data_from_ocr_output(ocr_result):
    data = []
    cur_period = cur_number = cur_bigsmall = None
    period_pat = r'^\d{17}$'
    num_pat     = r'^\d$'
    bs_pat      = r'^(Big|Small)$'

    for _, txt, _ in ocr_result:
        if re.match(period_pat, txt):
            if cur_period and cur_number is not None and cur_bigsmall:
                data.append({"Period": cur_period,
                             "Big_Small": 1 if cur_bigsmall.lower() == "big" else 0})
            cur_period, cur_number, cur_bigsmall = txt, None, None
        elif re.match(num_pat, txt):
            cur_number = txt
        elif re.match(bs_pat, txt):
            cur_bigsmall = txt

    if cur_period and cur_number is not None and cur_bigsmall:
        data.append({"Period": cur_period,
                     "Big_Small": 1 if cur_bigsmall.lower() == "big" else 0})
    return data

def process_folder(folder):
    all_data = []
    for f in sorted(os.listdir(folder)):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(folder, f)
            ocr_res = get_ocr_result(path)
            all_data.extend(extract_data_from_ocr_output(ocr_res))
    return all_data

if __name__ == "__main__":
    data = process_folder("../images")
    pd.DataFrame(data).to_csv("../data/wingo_batch_results.csv", index=False)
    print(f"Extracted {len(data)} rows → data/wingo_batch_results.csv")