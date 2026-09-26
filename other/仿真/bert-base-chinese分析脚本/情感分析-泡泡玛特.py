import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import datetime
from collections import defaultdict

# 设置本地模型路径
local_model_path = r"D:\Code代码\BJTU\models\bert-base-chinese"


def read_json_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


# 示例用法
file_path = '泡泡玛特直播事故.json'
data = read_json_file(file_path)
print(f"数据条数: {len(data)}")


def extract_comments_and_timestamps(data):
    comments = []
    timestamps = []
    for item in data:
        comments.append(item.get('content', ''))
        timestamps.append(item.get('time', ''))
    return comments, timestamps


# 示例用法
comments, timestamps = extract_comments_and_timestamps(data)
print(f"前5条评论: {comments[:5]}")
print(f"前5个时间戳: {timestamps[:5]}")

# 使用本地模型
print("正在加载本地模型...")
try:
    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(local_model_path, num_labels=3)
    print("✅ 本地模型加载成功！")

    # 检查模型文件
    import os

    files = os.listdir(local_model_path)
    print(f"模型文件夹中的文件: {files}")

except Exception as e:
    print(f"❌ 模型加载失败: {e}")
    # 如果失败，使用基于规则的方法
    tokenizer = None
    model = None

# 情感分析部分
if tokenizer is not None and model is not None:
    print("使用BERT模型进行情感分析...")


    class CommentDataset(Dataset):
        def __init__(self, comments, tokenizer, max_length=128):
            self.comments = comments
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.comments)

        def __getitem__(self, idx):
            comment = str(self.comments[idx])
            inputs = self.tokenizer.encode_plus(
                comment,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            return {
                'input_ids': inputs['input_ids'].flatten(),
                'attention_mask': inputs['attention_mask'].flatten()
            }


    # 创建数据集
    dataset = CommentDataset(comments, tokenizer)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    model.to(device)

    # 模型推理
    model.eval()
    predictions = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predicted_labels = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(predicted_labels)
            if (i + 1) % 10 == 0:
                print(f"已处理 {i + 1} 个批次")

    sentiments = ['负面' if label == 0 else '中性' if label == 1 else '正面' for label in predictions]

else:
    # 基于规则的情感分析
    print("使用基于规则的情感分析...")
    sentiments = []

    # 定义情感词典
    positive_words = ['好', '棒', '赞', '喜欢', '不错', '支持', '厉害', '优秀', '完美', '满意',
                      '开心', '高兴', '惊喜', '超值', '推荐', '物超所值', '点赞', '棒棒', '太好了',
                      '漂亮', '精美', '可爱', '值得', '划算', '便宜', '实惠', '好用', '方便']

    negative_words = ['差', '烂', '垃圾', '讨厌', '失望', '不好', '恶心', '坑', '骗', '垃圾',
                      '糟糕', '差劲', '无语', '生气', '愤怒', '后悔', '浪费', '骗钱', '上当',
                      '贵', '贵死', '不值', '质量差', '服务差', '慢', '久', '难', '复杂']

    for comment in comments:
        comment_str = str(comment)
        positive_count = sum(1 for word in positive_words if word in comment_str)
        negative_count = sum(1 for word in negative_words if word in comment_str)

        if positive_count > negative_count:
            sentiments.append('正面')
        elif negative_count > positive_count:
            sentiments.append('负面')
        else:
            sentiments.append('中性')

print(f"情感分析完成！")
print(f"正面: {sentiments.count('正面')} 条")
print(f"中性: {sentiments.count('中性')} 条")
print(f"负面: {sentiments.count('负面')} 条")

# 显示一些分析结果
print("\n前10条评论的情感分析结果:")
for i, (comment, sentiment) in enumerate(zip(comments[:10], sentiments[:10])):
    print(f"{i + 1}. {comment} -> {sentiment}")


# 时间序列分析
def parse_time(time_str):
    try:
        return datetime.datetime.strptime(time_str, '%H:%M')
    except:
        return None


# 处理时间戳和情感
valid_data = []
for time_str, sentiment, comment in zip(timestamps, sentiments, comments):
    dt = parse_time(time_str)
    if dt is not None:
        valid_data.append((dt, sentiment, comment))

# 按时间排序
valid_data.sort(key=lambda x: x[0])

if valid_data:
    # 计算情感趋势
    sentiment_trend = defaultdict(lambda: {'正面': 0, '中性': 0, '负面': 0})
    for dt, sentiment, _ in valid_data:
        hour = dt.hour
        sentiment_trend[hour][sentiment] += 1

    # 准备绘图数据
    hours = sorted(sentiment_trend.keys())
    positive_counts = [sentiment_trend[h]['正面'] for h in hours]
    neutral_counts = [sentiment_trend[h]['中性'] for h in hours]
    negative_counts = [sentiment_trend[h]['负面'] for h in hours]

    # 绘制情感趋势图
    plt.figure(figsize=(12, 6))
    plt.plot(hours, positive_counts, label='正面情感', marker='o', linewidth=2)
    plt.plot(hours, neutral_counts, label='中性情感', marker='s', linewidth=2)
    plt.plot(hours, negative_counts, label='负面情感', marker='^', linewidth=2)

    plt.xlabel('时间（小时）')
    plt.ylabel('评论数量')
    plt.title('泡泡玛特直播评论情感趋势分析')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(hours)

    # 添加数值标签
    for i, (h, pos, neu, neg) in enumerate(zip(hours, positive_counts, neutral_counts, negative_counts)):
        plt.text(h, pos, str(pos), ha='center', va='bottom')
        plt.text(h, neu, str(neu), ha='center', va='bottom')
        plt.text(h, neg, str(neg), ha='center', va='bottom')

    plt.tight_layout()
    plt.show()

    print(f"\n情感趋势分析完成！时间范围: {hours[0]}:00 - {hours[-1]}:00")
else:
    print("没有有效的时间数据用于趋势分析")

# 保存分析结果到文件
results = []
for comment, timestamp, sentiment in zip(comments, timestamps, sentiments):
    results.append({
        'comment': comment,
        'timestamp': timestamp,
        'sentiment': sentiment
    })

# 保存为JSON文件
with open('情感分析结果.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("分析结果已保存到 '情感分析结果.json'")