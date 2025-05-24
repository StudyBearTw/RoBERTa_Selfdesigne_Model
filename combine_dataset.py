import pandas as pd
from sklearn.model_selection import train_test_split
import os

# 資料夾路徑
data_dir = r"C:\Users\user\Desktop\RoBERTa_Model_Selfdesign\DataSet\Data2"

# 讀取資料
true_path = os.path.join(data_dir, "true_news4.csv")
fake_path = os.path.join(data_dir, "fake_news4.csv")

true_df = pd.read_csv(true_path, sep="\t", encoding="utf-16")
fake_df = pd.read_csv(fake_path, sep="\t", encoding="utf-16")

# 合併並打亂
df = pd.concat([true_df, fake_df], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 切分
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# 儲存到 Data2 目錄下
train_df.to_csv(os.path.join(data_dir, "train.tsv"), sep="\t", index=False, encoding="utf-8")
test_df.to_csv(os.path.join(data_dir, "test.tsv"), sep="\t", index=False, encoding="utf-8")

print("✅ train.tsv 與 test.tsv 已成功儲存在 Data2 資料夾中！")
