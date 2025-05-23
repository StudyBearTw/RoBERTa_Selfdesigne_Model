import os
import pandas as pd

def prepare_thucnews_dataset(thucnews_dir, output_csv):
    """
    將 THUCNews 數據集整合為單列 CSV 文件
    :param thucnews_dir: THUCNews 數據集的根目錄
    :param output_csv: 輸出的 CSV 文件路徑
    """
    all_texts = []
    for category in os.listdir(thucnews_dir):
        category_dir = os.path.join(thucnews_dir, category)
        if os.path.isdir(category_dir):
            for file_name in os.listdir(category_dir):
                file_path = os.path.join(category_dir, file_name)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        all_texts.append(f.read().strip())
                except Exception as e:
                    print(f"無法讀取文件 {file_path}: {e}")
    
    # 保存為 CSV 文件
    df = pd.DataFrame({'content': all_texts})
    df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"數據集已保存到 {output_csv}")

# 使用示例
if __name__ == "__main__":
    thucnews_dir = r"C:\Users\user\Desktop\RoBERTa_Model_Selfdesign\DataSet\THUC_NEWS"
    output_csv = r"C:/Users/user/Desktop/RoBERTa_Model_Selfdesign/DataSet/thucnews.csv"
    prepare_thucnews_dataset(thucnews_dir, output_csv)