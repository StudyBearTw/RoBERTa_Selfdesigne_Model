# 📰 中文真假新聞辨識模型（RoBERTa 自行設計版）

本專案為一個基於 RoBERTa 架構，並自行實作的中文真假新聞辨識模型。整體系統涵蓋從資料預處理、嵌入層設計、編碼器實作到分類任務的完整流程，適合作為自然語言處理（NLP）實務應用與研究的參考。

> 📘 本專案為大學專題成果之一，由 [StudyBearTw](https://github.com/StudyBearTw) 製作與維護。

---

## 🔍 專案簡介

- **任務類型**：中文新聞二分類（真 / 假）
- **模型架構**：模仿 RoBERTa 設計，並由零開始實作（不依賴 Hugging Face 等現成模型）
- **Tokenizer**：使用清華大學中文 BERT Tokenizer（`bert-base-chinese`）
- **輸入格式**：新聞標題與內文組合為單一輸入
- **訓練資料**：真實世界的中文新聞資料集（詳見 `data/`）

---

## 🧠 模型架構

本模型主要分為以下幾個模組（皆為 `.py` 檔案形式）：

- `embeddings/`: 實作 RoBERTa 嵌入層（token、position、segment embedding）
- `encoder/`: 多層 Transformer 編碼器，每層具備自注意力與前饋神經網路
- `model/`: 結合嵌入層與編碼器的主模型類別
- `train.py`: 模型訓練主程式
- `evaluate.py`: 測試與驗證模組效能
- `utils/`: 包含 tokenizer 處理、資料讀取、訓練流程等工具函式

---

## 🛠️ 安裝與執行方式

### 1. 環境安裝

請先安裝 Python 套件：

```bash
pip install -r requirements.txt
````

> ✅ 建議使用 Python 3.8+ 與 PyTorch 1.10+

### 2. 資料準備

將訓練與測試資料放入 `data/` 資料夾，格式如下：

```
data/
├── train.csv
├── test.csv
```

資料格式參考：

```csv
{
  "title": "新聞標題",
  "label": 1
}
```

### 3. 模型訓練

```bash
python train.py
```

### 4. 模型評估

```bash
python evaluate.py
```

---

## 📈 成效與評估

* 評估指標：Accuracy, F1-score
* 測試結果顯示本模型在驗證集上有良好的分類能力（詳見 `results/`）

---

## 📁 專案架構

```
RoBERTa_Selfdesigne_Model/
├── embeddings/
├── encoder/
├── model/
├── data/
├── utils/
├── train.py
├── evaluate.py
├── requirements.txt
└── README.md
```

---

## 📚 參考資源

* [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)
* [HuggingFace Transformers (作為架構參考)](https://github.com/huggingface/transformers)
* [Google BERT Paper](https://arxiv.org/abs/1810.04805)

---

## 📬 聯絡方式

如有任何問題或建議，歡迎聯絡我：

* GitHub: [StudyBearTw](https://github.com/StudyBearTw)
* Email: `studyspiderpig@gmail.com`

---

## 📜 License

本專案採用 MIT 授權條款，詳見 `LICENSE` 檔案。

```

---
