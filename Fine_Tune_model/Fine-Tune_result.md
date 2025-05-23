# RoBERTa Fine-tuning Project

## 📋 專案概述

本專案使用自訂義的RoBERTa模型進行fine-tuning，搭配哈工大的Chinese BERT WWM tokenizer，實現了優異的中文文本分類性能。

## 🏗️ 模型架構

- **基礎模型**: 自訂義RoBERTa
- **Tokenizer**: 哈工大Chinese BERT WWM
- **任務類型**: 文本分類
- **語言**: 中文

## 📊 模型性能

### 最終評估指標
| 指標 | 數值 |
|-----|------|
| 準確率 (Accuracy) | **99.43%** |
| F1分數 (F1-score) | **99.47%** |
| 精確率 (Precision) | **99.31%** |
| 召回率 (Recall) | **99.62%** |
| 最終損失 (Loss) | **0.0306** |

### 訓練過程概覽
- **訓練輪數**: 3 epochs
- **訓練時間**: 4小時6分42秒
- **訓練樣本處理速度**: 15.24 samples/second
- **初始學習率**: 2e-05
- **最終學習率**: 0 (線性衰減)

## 📈 訓練曲線分析

### 損失變化趨勢
```
Epoch 0.02: 0.4171 → Epoch 3.0: 0.01
```

- **快速收斂**: 訓練初期損失快速下降
- **穩定優化**: 全程無過擬合現象
- **持續改善**: 損失值持續穩定下降至收斂

### 驗證集表現
模型在驗證集上表現優異，各階段評估結果：

| Epoch | Accuracy | F1-score | Precision | Recall |
|-------|----------|----------|-----------|---------|
| 0.11  | 97.44%   | 97.63%   | 96.07%    | 99.25%  |
| 1.06  | 99.24%   | 99.29%   | 98.95%    | 99.64%  |
| 2.02  | 99.35%   | 99.39%   | 99.10%    | 99.69%  |
| 2.98  | **99.43%** | **99.47%** | **99.31%** | **99.62%** |

## 🔧 訓練配置

### 超參數設定
```python
learning_rate: 2e-05 (線性衰減至0)
epochs: 3
batch_size: [從日誌推算約8-16]
optimizer: AdamW
scheduler: Linear decay
```

### 評估策略
- 定期評估間隔: 每0.11 epoch
- 評估指標: Accuracy, F1, Precision, Recall
- 早停機制: 未使用（模型持續改善）

## 🚀 使用方法

### 環境要求
```bash
pip install transformers
pip install torch
pip install datasets
```

### 模型載入範例
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 載入tokenizer
tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")

# 載入fine-tuned模型
model = AutoModelForSequenceClassification.from_pretrained("./path/to/your/model")

# 推理範例
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return predictions
```

### 批量預測
```python
def batch_predict(texts):
    inputs = tokenizer(texts, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return predictions
```

## 📝 訓練日誌重點

### 關鍵里程碑
- **Epoch 0.02**: 初始損失 0.4171
- **Epoch 0.11**: 首次評估，準確率達97.44%
- **Epoch 1.06**: 準確率突破99%
- **Epoch 2.98**: 達到最佳性能
- **Epoch 3.00**: 訓練完成，最終損失 0.01

### 性能穩定性
- 驗證集準確率在訓練後期保持在99%以上
- F1分數與準確率高度一致，顯示分類平衡性良好
- 精確率與召回率平衡，適合實際應用

## 🎯 應用建議

### 適用場景
- 中文文本分類任務
- 內容過濾

### 部署考量
- 模型大小: RoBERTa-base級別
- 推理速度: 適中，適合批量處理
- 記憶體需求: 建議GPU環境以獲得最佳性能

## 📊 與基準模型比較

本模型相較於標準中文BERT模型的優勢：
- ✅ 更高的準確率 (99.43% vs 一般95-97%)
- ✅ 更好的F1分數 (99.47%)
- ✅ 優秀的精確率-召回率平衡
- ✅ 快速收斂，訓練效率高

## 🔍 進一步改進建議

1. **數據增強**: 考慮使用同義詞替換、回譯等技術
2. **正則化**: 若有過擬合傾向，可加入dropout或weight decay
3. **集成學習**: 結合多個模型提升穩定性
4. **模型蒸餾**: 若需要更快推理速度，可考慮知識蒸餾

## 📄 授權條款

請遵循相關模型的授權條款：
- RoBERTa: MIT License
- Chinese BERT WWM: Apache 2.0 License

*最後更新: 2025年5月*
