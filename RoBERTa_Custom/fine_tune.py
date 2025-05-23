import os
import sys
import torch
import pandas as pd
from transformers import (
    BertTokenizer,
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# 使用相對導入
from .model import RobertaForSequenceClassification

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def main():
    # 檢查 GPU 可用性
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用設備: {device}")

    # 載入預訓練模型的 tokenizer
    tokenizer = BertTokenizer.from_pretrained('hfl/chinese-bert-wwm-ext')
    
    # 讀取數據集
    fake_news_path = r"C:\Users\user\Desktop\RoBERTa_Model_Selfdesign\DataSet\FineTune_Data1\fake_news3.csv"
    true_news_path = r"C:\Users\user\Desktop\RoBERTa_Model_Selfdesign\DataSet\FineTune_Data1\true_news3.csv"

    if not os.path.exists(fake_news_path) or not os.path.exists(true_news_path):
        raise FileNotFoundError("數據集文件不存在")

    # 讀取並處理數據
    df_fake = pd.read_csv(fake_news_path, encoding='utf-16', sep='\t')
    df_true = pd.read_csv(true_news_path, encoding='utf-16', sep='\t')

    print(f"假新聞數據行數: {len(df_fake)}, 列: {df_fake.columns.tolist()}")
    print(f"真新聞數據行數: {len(df_true)}, 列: {df_true.columns.tolist()}")
    
    if len(df_fake) == 0 or len(df_true) == 0:
        raise ValueError("數據集為空!")

    if "news_title" not in df_fake.columns or "news_title" not in df_true.columns:
        print(f"假新聞列名: {df_fake.columns.tolist()}")
        print(f"真新聞列名: {df_true.columns.tolist()}")
        raise ValueError("未找到 'news_title' 列!")

    df_fake['label'] = 0  # 假新聞
    df_true['label'] = 1  # 真新聞

    # 合併數據集
    df = pd.concat([df_fake, df_true], ignore_index=True)
    print(f"成功讀取數據集，共 {len(df)} 條記錄")
    print(f"合併後數據集的列: {df.columns.tolist()}")
    print(f"news_title 列的空值數量: {df['news_title'].isnull().sum()}")

    # 數據預處理
    text_column = "news_title"
    df = df.dropna(subset=[text_column])
    print(f"刪除空值後的記錄數: {len(df)}")
    if len(df) == 0:
        raise ValueError("刪除空值後數據集為空!")

    # 分割訓練集和測試集
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    print(f"訓練集大小: {len(train_df)}, 測試集大小: {len(test_df)}")

    # 創建數據集
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    print(f"訓練數據集大小: {len(train_dataset)}")
    print(f"測試數據集大小: {len(test_dataset)}")
    if len(train_dataset) > 0:
        print("訓練數據集第一個樣本:", train_dataset[0])
    else:
        raise ValueError("訓練數據集為空!")

    # 預處理函數
    def preprocess_function(examples):
        encodings = tokenizer(
            examples[text_column],
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors=None
        )
        encodings['labels'] = examples['label']
        return encodings

    print("處理訓練集...")
    train_tokenized = train_dataset.map(
        preprocess_function,
        batched=True,
        batch_size=32,
        remove_columns=train_dataset.column_names,
        desc="處理訓練集"
    )

    print("處理測試集...")
    test_tokenized = test_dataset.map(
        preprocess_function,
        batched=True,
        batch_size=32,
        remove_columns=test_dataset.column_names,
        desc="處理測試集"
    )

    train_tokenized.set_format(
        type='torch',
        columns=['input_ids', 'attention_mask', 'token_type_ids', 'labels']
    )
    test_tokenized.set_format(
        type='torch',
        columns=['input_ids', 'attention_mask', 'token_type_ids', 'labels']
    )

    # 初始化模型
    model = RobertaForSequenceClassification(
        vocab_size=tokenizer.vocab_size,
        max_position_embeddings=512,
        num_labels=2,
        hidden_size=768,
        num_heads=12,
        intermediate_size=3072,
        num_hidden_layers=12,
        type_vocab_size=1,
        dropout=0.1
    ).to(device)

    # 載入預訓練權重（如有需要）
    pretrained_path = r"C:\Users\user\Desktop\RoBERTa_Model_Selfdesign\output\final_model\model.pt"
    if os.path.exists(pretrained_path):
        print("載入預訓練權重...")
        pretrained_weights = torch.load(pretrained_path, map_location=device)
        shared_weights = {k: v for k, v in pretrained_weights.items() if k in model.state_dict()}
        model.load_state_dict(shared_weights, strict=False)
        print("預訓練權重載入完成")
    else:
        print("未找到預訓練權重，使用隨機初始化")

    # 初始化數據整理器
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 訓練參數
    training_args = TrainingArguments(
        output_dir=r"C:\Users\user\Desktop\RoBERTa_Model_Selfdesign\Fine_Tune_model\results",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        weight_decay=0.01,
        logging_dir=r"C:\Users\user\Desktop\RoBERTa_Model_Selfdesign\Fine_Tune_model\logs",
        logging_steps=100,
        learning_rate=2e-5,
        save_total_limit=2,
        evaluation_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        fp16=False,
        gradient_accumulation_steps=1,
        warmup_steps=500,
        dataloader_num_workers=0,
        remove_unused_columns=False
    )

    # 初始化 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=test_tokenized,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    # 檢查數據加載器中的批次數據
    print("檢查數據加載器...")
    try:
        train_dataloader = trainer.get_train_dataloader()
        print(f"訓練數據加載器批次數: {len(train_dataloader)}")
        sample_batch = next(iter(train_dataloader))
        print("樣本批次鍵:", sample_batch.keys())
        if 'input_ids' in sample_batch:
            print("輸入ID形狀:", sample_batch['input_ids'].shape)
        else:
            print("警告: 批次中沒有 'input_ids' 字段!")
        if 'labels' in sample_batch:
            print("標籤形狀:", sample_batch['labels'].shape)
            print("標籤示例:", sample_batch['labels'][:5])
        else:
            print("警告: 批次中沒有 'labels' 字段!")
        # 將樣本批次傳入模型進行測試
        with torch.no_grad():
            sample_batch = {k: v.to(device) for k, v in sample_batch.items()}
            outputs = model(**sample_batch)
        if isinstance(outputs, dict) and "loss" in outputs:
            print("損失值:", outputs["loss"].item())
        elif isinstance(outputs, tuple) and hasattr(outputs[0], "item"):
            print("損失值:", outputs[0].item())
        else:
            print("警告: 模型未返回損失!")
    except Exception as e:
        print(f"測試過程中發生錯誤: {e}")
        import traceback
        traceback.print_exc()

    # 開始訓練
    print("開始訓練...")
    trainer.train()

    # 評估模型
    print("評估模型...")
    eval_results = trainer.evaluate()
    print(f"評估結果: {eval_results}")

    # 保存模型
    save_dir = r"C:\Users\user\Desktop\RoBERTa_Model_Selfdesign\Fine_Tune_model"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, "model.pt")
    torch.save(model.state_dict(), model_path)
    tokenizer.save_pretrained(save_dir)
    print(f"模型已保存至: {model_path}")

if __name__ == "__main__":
    main()