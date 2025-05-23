# DataSet
## Pre-Train dataset使用Kaggle Dataset
### 1.THUC News
https://www.kaggle.com/datasets/cycloneboy/thucnews
### 2.Yet Another Chinese News Dataset
https://www.kaggle.com/datasets/ceshine/yet-another-chinese-news-dataset
### 3.Chinese_news Dataset
https://www.kaggle.com/datasets/noxmoon/chinese-official-daily-news-since-2016/data
## Fine-Tune dataset使用
## 假新聞標題資料集說明 (Fake News Headline Dataset README)

### 資料集概述

本資料集專為自然語言處理中的「假新聞識別」應用場景而設計。資料集內容涵蓋由系統化方法產生的假新聞標題，以及來自可信來源的真新聞標題，並混合部分未驗證樣本以模擬真實世界情境，提供研究人員一個具挑戰性且貼近實際應用的訓練環境。

### 資料來源與組成

#### 假新聞 (Fake)

**主要來源：**
- 由假新聞生成器 (`generator.py`) 自動生成
- 透過模板替換技術與語意檢查 (GPT-2 perplexity) 產出語氣逼真的標題
- 標注為未經驗證狀態

**輔助樣本：**
- 少量來自 Kaggle 來源之假新聞，大多經過驗證
- 用於增加語料真實性與模糊邊界挑戰性
- **來源：** [WSDM - Fake News Classification](https://www.kaggle.com/datasets/wsdmcup/wsdm-fake-news-classification?select=test.csv)

#### 真新聞 (Real)

包含部分未知來源但經人為檢查過之內容，用以模擬社群平台中真實資訊流動狀況。

**來源：** [Yet Another Chinese News Dataset](https://www.kaggle.com/datasets/ceshine/yet-another-chinese-news-dataset?resource=download)

### 假新聞生成邏輯 (Fake News Generation)

#### 生成方法
採用模板與關鍵詞庫組合生成技術，涵蓋以下主題領域：
- 政治
- 社會
- 科技
- 經濟
- 娛樂

#### 品質過濾
使用 GPT-2 模型計算困惑度 (perplexity)，並排除語意不通順之標題，確保生成內容的自然性與可信度。

#### 標籤分類
每筆標題皆標記對應主題類別，便於後續分析與應用。

#### 生成範例
1. 知名企業家宣布稅收改革，激起民眾抗議 (政治)
2. 台北驚爆金融詐騙，國安局緊急調查 (社會)

## 資料特徵與統計

| 類型 | 筆數 (估計) | 備註 |
|------|-------------|------|
| 假新聞 | ~119,273 | 由生成器與部分未驗證樣本組成 |
| 真新聞 | ~142,663 | 來源可靠，部分為未知來源 |

*註：實際筆數請依資料夾內統計為準*

### 資料集分割

全部資料經隨機分割為三個子資料集：
每個子資料集包含:
- 真新聞 (True news) 
- 假新聞 (False news)

### 使用建議

本資料集適用於：
- 假新聞檢測模型訓練
- 中文新聞文本分類研究
- 自然語言處理模型評估
- 資訊可信度分析研究

### 注意事項

- 資料集中的假新聞內容僅供學術研究使用
- 使用者應注意資料的時效性與來源限制
- 建議在實際應用前進行額外的資料驗證與清理

### 更新與維護

資料集版本資訊與更新記錄請參考相關文件或聯繫資料集維護者。


