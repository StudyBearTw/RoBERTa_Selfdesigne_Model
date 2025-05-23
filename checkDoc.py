import chardet

fake_news_path = r"C:\Users\user\Desktop\RoBERTa_Model_Selfdesign\DataSet\FineTune_Data1\fake_news3.csv"
true_news_path = r"C:\Users\user\Desktop\RoBERTa_Model_Selfdesign\DataSet\FineTune_Data1\true_news3.csv"

with open(fake_news_path, "rb") as f:
    result = chardet.detect(f.read(10000))
    print(f"Fake news file encoding: {result}")

with open(true_news_path, "rb") as f:
    result = chardet.detect(f.read(10000))
    print(f"True news file encoding: {result}")