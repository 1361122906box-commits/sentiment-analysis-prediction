import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import jieba
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt

# 自定义分词函数
def tokenize(text, max_length):
    tokens = jieba.lcut(text)
    # 截断或填充到固定长度
    if len(tokens) > max_length:
        tokens = tokens[:max_length]
    else:
        tokens += [''] * (max_length - len(tokens))
    return tokens

# 加载数据
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# 构建词汇表
def build_vocab(texts):
    vocab = {}
    index = 0
    for text in texts:
        tokens = jieba.lcut(text)
        for token in tokens:
            if token not in vocab:
                vocab[token] = index
                index += 1
    return vocab

# 数据预处理
class WeiboDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length, vocab):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.vocab = vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        tokens = self.tokenizer(text, self.max_length)
        token_ids = [self.vocab[token] if token in self.vocab else 0 for token in tokens]  # 使用词典将分词结果转换为索引
        return {
            'input_ids': torch.tensor(token_ids),
            'label': torch.tensor(label)
        }

# 加载数据
name_dict = {
        "经济政策": "2025经济政策解读",
        "中美贸易": "2025中美贸易谈判",
        "A股成交": "A股半日成交1.25万亿缩量711亿",
        "福建舰入列": "福建舰入列",
        "网络谣言": "公安机关查处网络谣言",
        "泡泡玛特": "泡泡玛特直播事故",
        "日本女子杀害母亲": "日本71岁女子杀害102岁母亲",
    }
name = name_dict['泡泡玛特']
json_file = fr"C:\Users\issuser\Documents\trae_projects\情感分析预测\清洗后的数据源\{name}.json"
data = load_data(json_file)

# 检查数据格式
if len(data) > 0:
    # 检查前5条数据的格式
    for item in data[:5]:
        print(f"内容: {item.get('content', 'N/A')}, 时间: {item.get('time', 'N/A')}")
else:
    print("数据为空，无法检查格式。")

# 提取文本和标签
texts = [item['content'] for item in data]
labels = [0 if '负面' in item['content'] else 1 if '正面' in item['content'] else 2 for item in data]  # 假设情感标签在内容中

# 构建词汇表
vocab = build_vocab(texts)

# 将标签编码为数值
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# 划分数据集
texts_train, texts_test, labels_train, labels_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# 创建数据集
max_length = 128  # 设置最大长度
train_dataset = WeiboDataset(texts_train, labels_train, tokenize, max_length, vocab)
test_dataset = WeiboDataset(texts_test, labels_test, tokenize, max_length, vocab)

# 创建数据加载器
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义 LSTM 模型
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, num_classes):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids):
        embedded = self.embedding(input_ids)
        lstm_out, _ = self.lstm(embedded)
        return self.fc(lstm_out[:, -1, :])  # 使用最后一个时间步的输出进行分类

# 参数设置
vocab_size = len(vocab)  # 使用实际词汇表大小
embedding_dim = 128
hidden_dim = 256
num_layers = 2
num_classes = 3  # 假设情感分类为负面、中性、正面

# 初始化模型
model = LSTMModel(vocab_size, embedding_dim, hidden_dim, num_layers, num_classes)

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练循环
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        input_ids = batch['input_ids']
        labels = batch['label']

        optimizer.zero_grad()
        outputs = model(input_ids)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

# 评估模型
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids']
        labels = batch['label']

        outputs = model(input_ids)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy: {correct / total}')

# 提取情感标签和时间戳
timestamps = []
sentiments = []
for item in data:
    content = item.get('content', '')
    time_str = item.get('time', '')
    if content:  # 确保内容不为空
        # 情感分类逻辑
        if '负面' in content:
            sentiments.append(0)
        elif '正面' in content:
            sentiments.append(1)
        else:
            sentiments.append(2)
    timestamps.append(time_str)

# 检查情感标签和时间戳
print(f"时间戳数量：{len(timestamps)}")
print(f"情感标签数量：{len(sentiments)}")
print(f"前5个时间戳：{timestamps[:5]}")
print(f"前5个情感标签：{sentiments[:5]}")

# 将时间戳转换为小时
from datetime import datetime

def parse_time(time_str):
    try:
        return datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
    except:
        return None

valid_timestamps = []
valid_sentiments = []
for time_str, sentiment in zip(timestamps, sentiments):
    dt = parse_time(time_str)
    if dt is not None:
        valid_timestamps.append(dt.hour)
        valid_sentiments.append(sentiment)

# 检查转换结果
print(f"有效时间戳数量：{len(valid_timestamps)}")
print(f"有效情感标签数量：{len(valid_sentiments)}")
print(f"前5个有效时间戳：{valid_timestamps[:5]}")
print(f"前5个有效情感标签：{valid_sentiments[:5]}")

# 计算情感趋势
from collections import defaultdict

sentiment_trend = defaultdict(lambda: {'负面': 0, '中性': 0, '正面': 0})
for hour, sentiment in zip(valid_timestamps, valid_sentiments):
    if sentiment == 0:
        sentiment_trend[hour]['负面'] += 1
    elif sentiment == 1:
        sentiment_trend[hour]['正面'] += 1
    else:
        sentiment_trend[hour]['中性'] += 1

# 转换为适合绘图的格式
hours = list(sentiment_trend.keys())
positive_counts = [trend['正面'] for trend in sentiment_trend.values()]
neutral_counts = [trend['中性'] for trend in sentiment_trend.values()]
negative_counts = [trend['负面'] for trend in sentiment_trend.values()]

# 检查趋势数据
print(f"小时列表：{hours}")
print(f"正面情感计数：{positive_counts}")
print(f"中性情感计数：{neutral_counts}")
print(f"负面情感计数：{negative_counts}")

# 绘制情感趋势图
from matplotlib.font_manager import FontProperties

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定中文字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 确保数据按小时排序
sorted_data = sorted(zip(hours, positive_counts, neutral_counts, negative_counts), key=lambda x: x[0])
hours, positive_counts, neutral_counts, negative_counts = zip(*sorted_data)

plt.figure(figsize=(12, 6))
plt.plot(hours, positive_counts, label='正面情感', marker='o', linestyle='-')
plt.plot(hours, neutral_counts, label='中性情感', marker='o', linestyle='-')
plt.plot(hours, negative_counts, label='负面情感', marker='o', linestyle='-')

plt.xlabel('时间（小时）')
plt.ylabel('评论数量')
plt.title('情感趋势预测')
plt.legend()
plt.grid(True)

# 自动调整x轴刻度，避免刻度过于密集
plt.xticks(hours, rotation=45)
plt.tight_layout()

# 保存图像到文件
plt.savefig('情感趋势预测.png')  # 保存为PNG格式
plt.savefig('情感趋势预测.pdf')  # 保存为PDF格式

# 显示图像
plt.show()

# 关闭绘图窗口
plt.close()