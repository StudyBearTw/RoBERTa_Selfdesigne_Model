import os
import torch
import pandas as pd
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 計算評估指標
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    return {
        "accuracy": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }

def main():
    # 定義數據集路徑
    fake_news_path = r"C:\Users\user\Desktop\RoBERTa_Model_Selfdesign\DataSet\FineTune_Data1\fake_news3.csv"
    true_news_path = r"C:\Users\user\Desktop\RoBERTa_Model_Selfdesign\DataSet\FineTune_Data1\true_news3.csv"

    # 檢查數據集文件是否存在
    if not os.path.exists(fake_news_path) or not os.path.exists(true_news_path):
        raise FileNotFoundError("請確認數據集文件是否存在於指定路徑中。")

    # 嘗試以適當的編碼讀取數據集
    try:
        df_fake = pd.read_csv(fake_news_path, encoding="utf-8")
        df_true = pd.read_csv(true_news_path, encoding="utf-8")
    except UnicodeDecodeError:
        print("無法以 utf-8 編碼讀取文件，嘗試使用 'cp950' 或其他中文編碼格式...")
        try:
            df_fake = pd.read_csv(fake_news_path, encoding="cp950")  # 常用於繁體中文
            df_true = pd.read_csv(true_news_path, encoding="cp950")
        except UnicodeDecodeError:
            print("嘗試使用 'gb18030' 編碼...")
            df_fake = pd.read_csv(fake_news_path, encoding="gb18030")  # 常用於簡體中文
            df_true = pd.read_csv(true_news_path, encoding="gb18030")

    # 標記資料
    df_fake["label"] = 0  # 假新聞
    df_true["label"] = 1  # 真新聞

    # 合併資料集
    df = pd.concat([df_fake, df_true], ignore_index=True)
    
    # 檢查並找出標題欄位
    title_column = None
    potential_title_columns = ["title", "標題", "news_title", "headline"]
    for col in potential_title_columns:
        if col in df.columns:
            title_column = col
            print(f"找到標題欄位: {title_column}")
            break
    
    # 如果找不到標題欄位，則嘗試使用內容欄位或其他可能的文本欄位
    if title_column is None:
        potential_text_columns = ["content", "text", "article", "內容", "news"]
        for col in potential_text_columns:
            if col in df.columns:
                title_column = col
                print(f"未找到標題欄位，使用內容欄位代替: {title_column}")
                break
        else:
            # 如果找不到，使用第一個非label的欄位
            title_column = [col for col in df.columns if col != "label"][0]
            print(f"未找到明確的標題或內容欄位，使用: {title_column}")

    # 輸出欄位列表供參考
    print(f"數據欄位: {df.columns.tolist()}")
    print(f"使用 '{title_column}' 作為分類依據")
    
    # 檢查並處理缺失值
    missing_before = df[title_column].isna().sum()
    df = df.dropna(subset=[title_column])
    missing_after = len(df)
    print(f"移除 {missing_before} 筆缺失值後，剩餘 {missing_after} 筆資料")

    # 資料基本統計
    print(f"真新聞筆數: {df[df['label'] == 1].shape[0]}")
    print(f"假新聞筆數: {df[df['label'] == 0].shape[0]}")
    
    # 顯示標題長度統計
    df['title_length'] = df[title_column].apply(len)
    print(f"標題長度統計: 最短 {df['title_length'].min()}, 最長 {df['title_length'].max()}, 平均 {df['title_length'].mean():.2f}")
    
    # 如果標題過長，可能是完整文章而非標題，提供警告
    if df['title_length'].max() > 200:
        print("警告: 有些資料可能不是標題，而是完整文章內容，請檢查資料集")
    
    # 根據標題長度統計選擇合適的max_length
    # 中文標題通常不會超過50個字符
    max_seq_length = min(128, int(df['title_length'].quantile(0.95)))
    print(f"設定序列最大長度為: {max_seq_length}")

    # 將資料劃分為訓練集和測試集
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    print(f"訓練集大小: {len(train_df)}, 測試集大小: {len(test_df)}")

    # 將 Pandas DataFrame 轉換為 Hugging Face Dataset
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    # 載入自訂預訓練RoBERTa模型和指定的tokenizer
    pretrained_model_path = r"C:\Users\user\Desktop\RoBERTa_Model_Selfdesign\output\final_model"  # 您的自訂模型路徑
    tokenizer_path = "hfl/chinese-roberta-wwm-ext"  # 您指定使用的tokenizer
    
    print(f"嘗試載入自訂預訓練模型: {pretrained_model_path}")
    print(f"使用指定的tokenizer: {tokenizer_path}")
    
    # 載入指定的tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        print(f"成功載入tokenizer: {tokenizer_path}")
    except Exception as e:
        print(f"無法載入指定的tokenizer，錯誤: {e}")
        print("嘗試載入備用tokenizer...")
        
        # 如果指定tokenizer載入失敗，嘗試使用備用tokenizer
        try:
            # 嘗試直接從您的模型目錄載入tokenizer相關文件
            if os.path.exists(os.path.join(pretrained_model_path, "vocab.txt")):
                print("在您的模型目錄中找到vocab.txt，嘗試直接載入...")
                tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)
                print("成功從模型目錄載入tokenizer")
            else:
                # 否則使用標準中文BERT tokenizer
                tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-bert-wwm-ext")
                print("使用標準中文BERT tokenizer作為備用")
        except Exception as e2:
            print(f"備用tokenizer載入也失敗，錯誤: {e2}")
            print("使用基本的BERT tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")

    # 資料預處理
    def preprocess_function(examples):
        return tokenizer(
            examples[title_column], 
            truncation=True, 
            padding="max_length", 
            max_length=max_seq_length
        )

    # 應用 tokenizer
    print("正在處理訓練集...")
    train_tokenized = train_dataset.map(preprocess_function, batched=True)
    print("正在處理測試集...")
    test_tokenized = test_dataset.map(preprocess_function, batched=True)

    # 載入自訂預訓練模型
    print(f"正在載入自訂預訓練模型: {pretrained_model_path}")
    
    try:
        # 檢查模型文件格式
        if os.path.exists(os.path.join(pretrained_model_path, "model.pt")):
            print("檢測到model.pt文件，使用PyTorch直接載入模式")
            # 使用PyTorch直接載入模型
            import torch
            
            # 載入模型參數
            model_state_dict = torch.load(os.path.join(pretrained_model_path, "model.pt"))
            
            # 先從基礎模型創建一個實例
            base_model = AutoModelForSequenceClassification.from_pretrained(
                tokenizer_path,  # 使用與tokenizer相同的基礎模型
                num_labels=2,    # 二分類任務
            )
            
            # 嘗試載入參數
            # 如果模型架構不完全匹配，可能需要調整state_dict的鍵名
            try:
                base_model.load_state_dict(model_state_dict)
                print("成功直接載入模型權重")
                model = base_model
            except Exception as e_load:
                print(f"直接載入權重失敗: {e_load}，嘗試部分載入...")
                # 嘗試過濾掉不匹配的權重
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in model_state_dict.items():
                    # 移除可能的'module.'前綴(常見於DDP訓練)
                    name = k.replace("module.", "") if k.startswith("module.") else k
                    # 只保留與base_model中存在的相同key
                    if name in base_model.state_dict():
                        if base_model.state_dict()[name].shape == v.shape:
                            new_state_dict[name] = v
                
                # 載入可匹配的權重
                base_model.load_state_dict(new_state_dict, strict=False)
                print("已部分載入兼容的模型權重")
                model = base_model
        else:
            # 使用transformers標準方式載入
            print("使用transformers標準API載入模型")
            model = AutoModelForSequenceClassification.from_pretrained(
                pretrained_model_path,
                num_labels=2,  # 二分類任務
            )
            print("成功載入自訂預訓練模型")
    except Exception as e:
        print(f"無法以標準方式載入自訂預訓練模型，錯誤: {e}")
        print("錯誤詳情:", str(e))
        
        # 嘗試其他方法載入模型
        try:
            from transformers import RobertaForSequenceClassification
            print("嘗試使用RobertaForSequenceClassification特定類別載入...")
            model = RobertaForSequenceClassification.from_pretrained(
                pretrained_model_path,
                num_labels=2
            )
            print("成功使用RobertaForSequenceClassification載入自訂模型")
        except Exception as e2:
            print(f"特定類別載入也失敗，錯誤: {e2}")
            print("最終嘗試：使用與tokenizer相同的預訓練模型...")
            
            # 如果無法載入自訂模型，使用與tokenizer相同的模型
            model = AutoModelForSequenceClassification.from_pretrained(
                tokenizer_path,
                num_labels=2
            )
            print(f"已使用與tokenizer相同的模型: {tokenizer_path}")
    
    # 顯示模型基本資訊
    print(f"模型參數數量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print(f"模型架構: {model.__class__.__name__}")

    # 檢查是否有GPU可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")
    model.to(device)

    # 設定訓練參數
    batch_size = 16  # 根據您的GPU記憶體調整
    output_dir = "./results_chinese_news_classification"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=5,  # 增加epoch數
        weight_decay=0.01,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",  # 使用F1分數來選擇最佳模型
        logging_dir="./logs",
        logging_steps=100,
        report_to="tensorboard",
        # 增加early stopping
        early_stopping_patience=3,
        early_stopping_threshold=0.01,
    )

    # 使用 Trainer 進行訓練
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=test_tokenized,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # 開始訓練
    print("開始訓練...")
    trainer.train()

    # 評估模型
    print("評估模型...")
    eval_results = trainer.evaluate()
    print(f"評估結果: {eval_results}")

    # 保存微調後的模型
    save_dir = r"C:\Users\user\Desktop\RoBERTa_Model_Selfdesign\Fine_Tune_model"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"模型已保存至 {save_dir}")

    # 測試預測函數
    def predict_fake_news(text):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=max_seq_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)

        probabilities = torch.softmax(outputs.logits, dim=1).detach().cpu().numpy()[0]
        prediction = np.argmax(probabilities)
        confidence = probabilities[prediction]

        result = "真新聞" if prediction == 1 else "假新聞"
        return result, confidence

    # 從測試集中選擇一些樣本進行預測
    print("\n測試模型預測能力...")
    sample_texts = test_df[title_column].sample(5).tolist()
    sample_labels = test_df.loc[test_df[title_column].isin(sample_texts), "label"].tolist()
    
    for text, true_label in zip(sample_texts, sample_labels):
        result, confidence = predict_fake_news(text)
        print(f"新聞: {text}")
        print(f"真實標籤: {'真新聞' if true_label == 1 else '假新聞'}")
        print(f"預測: {result}, 信心度: {confidence:.4f}")
        print("-" * 50)

    # 計算整體測試集表現
    print("\n計算整體測試集表現...")
    predictions = []
    for text in test_df[title_column].tolist():
        result, _ = predict_fake_news(text)
        pred = 1 if result == "真新聞" else 0
        predictions.append(pred)
    
    accuracy = accuracy_score(test_df["label"].tolist(), predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_df["label"].tolist(), predictions, average="binary"
    )
    
    print(f"測試集準確率: {accuracy:.4f}")
    print(f"測試集F1分數: {f1:.4f}")
    print(f"測試集精確率: {precision:.4f}")
    print(f"測試集召回率: {recall:.4f}")

if __name__ == "__main__":
    main()