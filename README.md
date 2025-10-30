# OK.Win Big/Small Predictor  

**Predict the next “Big” or “Small” outcome on the OK.Win lottery-style game using OCR + time-series features + ML.**
![batch1 (29)](https://github.com/user-attachments/assets/fd8a06d0-63aa-4b52-8cf4-7a9cc3e8decc)


---

## 📌 Project Overview
1. **OCR Extraction** – PaddleOCR reads game screenshots → structured CSV.  
2. **Feature Engineering** – Parse the 17-digit `Period` into year/month/day/hour/minute/second/serial.  
3. **Modeling** – 7 classic classifiers; best model (usually Random Forest) is saved.  
4. **Inference** – Load `best_model.pkl` + `scaler.pkl` to predict live screenshots.

---

## 🚀 Quick Start

```bash
# 1. Clone
git clone https://github.com/rkarahul/OK.Win-Big-Small-Predictor.git
cd okwin-bigsmall-predictor

# 2. Install
pip install -r requirements.txt

# 3. Run pipeline (images → model)
python src/01_ocr_extract.py   # needs folder ./images/
python src/02_format_data.py
python src/03_train_model.py
