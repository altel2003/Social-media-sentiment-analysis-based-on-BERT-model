import pandas as pd
import jieba
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup  # 分别是bert的分词器，分词模型，优化器,
import matplotlib.pyplot as plt
import numpy as np


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 数据预处理
def DataPreprocess(data):
    # 过滤掉缺失值
    data = data[pd.notna(data['文本'])]
    data.reset_index(drop=True, inplace=True)  # 重新建立索引
    # 打印过滤后的数据大小
    print(f"Data size after removing missing values: {data.shape}")

    # # 分词
    # texts = data['文本'].apply(lambda x: ' '.join(jieba.cut(x)))  # jieba分词
    texts = data['文本']
    # 标签
    label_temp = {'neutral': 0, 'happy': 1, 'angry': 2, 'sad': 3, 'fear': 4, 'surprise': 5}
    labels = data['情绪标签'].map(label_temp)

    return texts, labels


# 文本编码（使用bert分词器）
def TextEncoder(texts, tokenizer, max_len=128):
    inputs = tokenizer(
        texts.tolist(),  # 将输入文本列表转换为Python列表
        padding=True,  # 自动填充到最大长度
        truncation=True,  # 超过最大长度时进行截断
        max_length=max_len,  # 设置最大长度
        return_tensors='pt'  # 返回PyTorch tensors
    )
    return inputs

# 定义文本预测函数
def predict(model, input_string: str):
    model.eval()
    predictions_dict = {0:'natural', 1:'happy', 2:'angry', 3:'sad', 4:'fear', 5:'surprise'}
    final_output = []
    # text = input_string.apply(lambda x: ' '.join(jieba.cut(x)))
    encoding = tokenizer(input_string.tolist(), padding=True, truncation=True, max_length=128, return_tensors='pt')
    input = {key: value.to(device) for key, value in encoding.items()}
    with torch.no_grad():
        inputs = {key: value.to(device) for key, value in input.items() if key != 'labels'}
        outputs = model(**inputs)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        for pred in predictions.tolist():
            final_output.append(predictions_dict[int(pred)])
        return final_output
# 定义数据集类
class WeiboCommentsDataset(Dataset):
    def __init__(self, encoding, labels):
        self.input_ids = [value.clone().detach().long() for value in encoding['input_ids']]
        self.attention_masks = [value.clone().detach().long() for value in encoding['attention_mask']]
        self.token_type_ids = [value.clone().detach().long() for value in encoding['token_type_ids']]
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __getitem__(self, idx):
        # 这里字典的key是根据模型的输入来定义的，如使用直接继承模型的方法，不能改动key
        item = {'input_ids': self.input_ids[idx], 'attention_mask': self.attention_masks[idx],
                'token_type_ids': self.token_type_ids[idx], 'labels': self.labels[idx]}
        # items = {key: val[idx] for key, val in self.encoding.items()}
        # labels = self.labels[idx]
        return item

    def __len__(self):
        return len(self.input_ids)


# 定义模型训练器类
class ModelTrainer:
    def __init__(self, model, train_dataloader, val_dataloader, test_dataloader, optimizer,scheduler, num_epochs):
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.all_loss = []
        self.all_test_accuracy = []
        self.all_val_accuracy = []

    # 定义每步训练函数
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for batch in self.train_dataloader:
            # print(len(batch[0]['input_ids']))
            self.optimizer.zero_grad()
            # inputs, labels = batch
            inputs = {key: value.to(self.device) for key, value in batch.items() if key != 'labels'}
            # labels = labels.to(self.device)
            labels = batch['labels'].to(self.device)
            outputs = self.model(**inputs, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
        self.all_loss.append(total_loss / len(self.train_dataloader))
        return total_loss / len(self.train_dataloader)

    # 定义评估函数
    def evaluate(self, dataloader, is_test=False):
        self.model.eval()
        total_accuracy = 0
        with torch.no_grad():
            for batch in dataloader:
                inputs = {key: value.to(self.device) for key, value in batch.items() if key != 'labels'}
                labels = batch['labels'].to(self.device)
                outputs = self.model(**inputs, labels=labels)
                logits = outputs.logits  # 作用是将[batch_size, num_labels]的tensor转换为[batch_size, num_labels]的numpy数组
                predictions = torch.argmax(logits, dim=-1)
                total_accuracy += (predictions == labels).sum().item()
            accuracy = total_accuracy / len(dataloader.dataset)
            if is_test:
                self.all_test_accuracy.append(accuracy)
            else:
                self.all_val_accuracy.append(accuracy)
        return accuracy

    # 定义训练函数
    def train(self, val_skip_steps):
        best_val_accuary = 0
        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch()
            if epoch % val_skip_steps == 0:
                val_accuracy = self.evaluate(self.val_dataloader, is_test=False)
                test_accuracy = self.evaluate(self.test_dataloader, is_test=True)
                print(
                    f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Accuracy: {val_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}')
                # 保存最优模型与最后一次训练的模型
                if val_accuracy > best_val_accuary:
                    torch.save(self.model.state_dict(), 'best_model.pth')
                    best_val_accuary = val_accuracy
                torch.save(self.model.state_dict(), 'last_model.pth')

    # 定义可视化模型评价函数
    def model_evaluate(self, is_visualize=False):
        self.model.eval()
        all_predictions = []
        all_labels = []
        with torch.no_grad():
            for batch in self.test_dataloader:
                # inputs, labels = batch
                # inputs = {key: value.to(self.device) for key, value in inputs.items()}
                # labels = labels.to(self.device)
                inputs = {key: value.to(self.device) for key, value in batch.items() if key != 'labels'}
                labels = batch['labels'].to(self.device)
                outputs = self.model(**inputs, labels=labels)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            print(all_predictions)
            print(all_labels)
        print(f'Test Accuracy: {(np.array(all_predictions) == np.array(all_labels)).sum() / len(all_labels):.4f}')
        if is_visualize:
            # 绘制混淆矩阵
            cm = confusion_matrix(all_labels, all_predictions)
            plt.matshow(cm, cmap=plt.cm.Blues)
            plt.colorbar()
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.show()
            # 绘制loss曲线
            plt.plot(self.all_loss)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.show()
            # 绘制准确率曲线
            plt.plot(self.all_test_accuracy, label='Test')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.show()
            # 绘制分类报告
            report = classification_report(all_labels, all_predictions,
                                           target_names=['neutral', 'happy', 'angry', 'sad', 'fear', 'surprise'])
            print(report)




if __name__ == '__main__':
    # 加载数据
    data_train = pd.read_csv('./DataSet/usual_train.csv')
    data_test = pd.read_csv('./DataSet/usual_test_labeled.csv')
    data_val = pd.read_csv('./DataSet/usual_eval_labeled.csv')

    # 数据预处理
    input_train, label_train = DataPreprocess(data_train)
    input_test, label_test = DataPreprocess(data_test)
    input_val, label_val = DataPreprocess(data_val)

    # 定义分词器和模型
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=6).to(device)

    # 编码训练集和测试集
    X_train = TextEncoder(input_train, tokenizer)
    y_train = torch.tensor(label_train)
    X_Test = TextEncoder(input_test, tokenizer)
    y_test = torch.tensor(label_test)
    X_val = TextEncoder(input_val, tokenizer)
    y_val = torch.tensor(label_val)

    # 定义数据集
    train_dataset = WeiboCommentsDataset(X_train, y_train)
    test_dataset = WeiboCommentsDataset(X_Test, y_test)
    val_dataset = WeiboCommentsDataset(X_val, y_val)

    # 定义数据加载器
    train_dataloader = DataLoader(train_dataset, batch_size=48,shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=48)
    test_dataloader = DataLoader(test_dataset, batch_size=48)

    epochs = 30
    # 定义优化器
    optimizer = AdamW(model.parameters(), lr=1e-6, weight_decay=0.001)
    # 定义学习率衰减器
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=5, num_training_steps=total_steps)  # num_warmup_steps： 预热

    # 定义训练器
    trainer = ModelTrainer(model, train_dataloader, val_dataloader, test_dataloader, optimizer,scheduler,
                           num_epochs=epochs)

    # 训练模型并保存最优模型与最后一次训练的模型
    trainer.train(val_skip_steps=1)

    # 测试模型
    trainer.model_evaluate(is_visualize=True)
