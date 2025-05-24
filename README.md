# 📰 中文真假新聞辨識模型（RoBERTa 自行設計版）

本專案為一個基於 RoBERTa 架構，並自行實作的中文真假新聞辨識模型。整體系統涵蓋從資料預處理、嵌入層設計、編碼器實作到分類任務的完整流程，適合作為自然語言處理（NLP）實務應用與研究的參考。

> 📘 本專案為大學專題成果之一，由 [StudyBearTw](https://github.com/StudyBearTw) 製作與維護。

---

## 🔍 專案簡介

- **任務類型**：中文新聞二分類（真 / 假）
- **模型架構**：模仿 RoBERTa 設計，並由零開始實作（不依賴 Hugging Face 等現成模型）
- **Tokenizer**：使用哈爾濱工業大學的hfl/chinese-bert-wwm-ext Tokenizer
- **輸入格式**：新聞標題與內文組合為單一輸入
- **訓練資料**：真實世界的中文新聞資料集（詳見 `data/`）

---

## 🧠 模型架構

本模型主要分為以下幾個模組（皆為 `.py` 檔案形式）：

- `embeddings/`: 實作 RoBERTa 嵌入層（token、position、segment embedding）
- `encoder/`: 多層 Transformer 編碼器，每層具備自注意力與前饋神經網路
- `model/`: 結合嵌入層與編碼器的主模型類別
- `pretrain.py`: 模型預訓練主程式
- `fine_tune.py`: 模型微調訓練主程式
- `test_model.py`: 測試與驗證模組效能

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
#### pre-train
```bash
python train_mlm.py
```
#### fine-tune
```bash
python run_fine_tune.py
```

### 4. 模型評估

```bash
python test_model.py
```

---

## 📈 成效與評估

* 評估指標：Accuracy（準確率）, Precision（精確率）, Recall（召回率）, F1 score（F1 分數）, Confusion matrix（混淆矩陣）
* 測試結果顯示本模型在驗證集上有良好的分類能力（詳見 `results/`）

---

## 📁 專案架構

```

ROBERTA\_SELFDESIGNE\_MODEL/
├── Dataset/
│   └── README.md
│
├── Fine\_Tune\_model/
│   ├── Fine-Tune\_result.md
│   └── README.md
│
├── output/
│   ├── Pre-Train\_result.md
│   └── README.md
│
├── RoBERTa\_Custom/
│   ├── **pycache**/
│   ├── **init**.py
│   ├── attention.py
│   ├── embeddings.py
│   ├── encoder.py
│   ├── fine\_tune.py
│   ├── mlm\_loss.py
│   ├── model.py
│   ├── pretrain.py
│   └── README.md
│
├── .gitattributes
├── checkDoc.py
├── combine\_dataset.py
├── Dataset\_Check.py
├── dataset\_file\_to\_csv.py
├── LICENSE
├── README.md
├── requirements.txt
├── run\_fine\_tune.py
├── test\_model.py
├── test.py
└── train\_mlm.py
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

## 📄 授權條款

請遵循相關模型的授權條款：
- Chinese BERT WWM: Apache 2.0 License

---

## 📜 License

本專案採用 MIT 授權條款，詳見 `LICENSE` 檔案。

