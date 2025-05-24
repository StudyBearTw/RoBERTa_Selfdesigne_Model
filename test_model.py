import torch
import pandas as pd
from transformers import BertTokenizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from RoBERTa_Custom.model import RobertaForSequenceClassification

# 1. 載入模型與 tokenizer
model_dir = r"C:\Users\user\Desktop\RoBERTa_Model_Selfdesign\Fine_Tune_model"
model_path = model_dir + r"\model.pt"
tokenizer = BertTokenizer.from_pretrained(model_dir)

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
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

# 2. 讀取 test 資料
test_path = r"C:\Users\user\Desktop\RoBERTa_Model_Selfdesign\DataSet\Data2\test.tsv"
df = pd.read_csv(test_path, encoding="utf-8", sep="\t")
texts = df["news_title"].tolist()
labels = df["news_authenticity"].tolist()

# 3. 預測
preds = []
for text in texts:
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs["logits"]
        pred = torch.argmax(logits, dim=1).item()
        preds.append(pred)

# 4. 計算指標
acc = accuracy_score(labels, preds)
prec = precision_score(labels, preds)
rec = recall_score(labels, preds)
f1 = f1_score(labels, preds)
cm = confusion_matrix(labels, preds)

print(f"Accuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall: {rec:.4f}")
print(f"F1 score: {f1:.4f}")
print("Confusion Matrix:")
print(cm)