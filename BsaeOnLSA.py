import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import jieba
import pandas as pd
import matplotlib.pyplot as plt

def DataPreprocess(data):
    # 过滤掉缺失值
    data = data[pd.notna(data['文本'])]
    data.reset_index(drop=True, inplace=True)  # 重新建立索引
    # 打印过滤后的数据大小
    print(f"Data size after removing missing values: {data.shape}")

    # 分词
    texts = data['文本'].astype('string').tolist()
    # 标签
    label_temp = {'neutral': 0, 'happy': 1, 'angry': 2, 'sad': 3, 'fear': 4, 'surprise': 5}
    labels = data['情绪标签'].map(label_temp)
    return texts, labels.to_numpy()

def mytokenizer(text):
    return list(jieba.cut(text))

data = pd.read_csv('./DataSet/usual_train.csv')
cuts, labels = DataPreprocess(data)

# 使用TF-IDF向量化文本
vectorizer = TfidfVectorizer(tokenizer=mytokenizer)
X_tfidf = vectorizer.fit_transform(cuts)
y = labels

# 使用LSA降维
lsa = TruncatedSVD(n_components=100, random_state=42)
X_lsa = lsa.fit_transform(X_tfidf)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_lsa, y, test_size=0.2, random_state=42)

# 使用支持向量机分类器进行分类
svm = SVC(kernel='linear', random_state=42)
svm.fit(X_train, y_train)

# 在测试集上评估模型
accuracy = svm.score(X_test, y_test)
print("LSA + SVM 模型的准确率：", accuracy)

# 预测测试集
y_pred = svm.predict(X_test)

# 生成分类报告
report = classification_report(y_test, y_pred, target_names=['neutral', 'happy', 'angry', 'sad', 'fear', 'surprise'])
print(report)

# 生成混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)

# 可视化混淆矩阵
plt.figure(figsize=(10, 7))
plt.matshow(conf_matrix, cmap=plt.cm.Blues)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()
