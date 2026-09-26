# main_analysis.py  主分析脚本
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import datetime
from collections import defaultdict
from visualization_data_exporter import generate_visualization_data
from data_loader import DataLoader as CustomDataLoader
from model_trainer import ModelTrainer

# 使用统一的字体工具
from font_utils import setup_chinese_font

# 设置中文字体
font_success = setup_chinese_font()
if not font_success:
    print("⚠️ 字体设置警告: 使用默认字体，中文可能显示异常")

# 项目设置
name_dict = {
    "经济政策": "2025经济政策解读",
    "中美贸易": "2025中美贸易谈判",
    "A股成交": "A股半日成交1.25万亿缩量711亿",
    "福建舰入列": "福建舰入列",
    "网络谣言": "公安机关查处网络谣言",
    "泡泡玛特": "泡泡玛特直播事故",
    "日本女子杀害母亲": "日本71岁女子杀害102岁母亲",
}

def parse_time(time_str):
    """更健壮的时间解析函数"""
    if not time_str or pd.isna(time_str):
        return None

    time_str = str(time_str).strip()

    formats = [
        '%Y-%m-%d %H:%M:%S',
        '%Y/%m/%d %H:%M:%S',
        '%Y-%m-%d %H:%M',
        '%Y/%m/%d %H:%M',
        '%Y年%m月%d日 %H:%M',
        '%Y.%m.%d %H:%M:%S'
    ]

    for fmt in formats:
        try:
            return datetime.datetime.strptime(time_str, fmt)
        except ValueError:
            continue

    print(f"无法解析的时间格式: {time_str}")
    return None

class CommentDataset(Dataset):
    """BERT数据集类"""
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

def analyze_sentiment_bert(comments, tokenizer, model, device):
    """使用BERT进行情感分析"""
    dataset = CommentDataset(comments, tokenizer)
    dataloader = TorchDataLoader(dataset, batch_size=8, shuffle=False)

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
    return sentiments

def analyze_sentiment_rule_based(comments):
    """基于规则的情感分析（备用）"""
    sentiments = []
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

    return sentiments

def plot_sentiment_trend(name, valid_data):
    """绘制情感趋势图"""
    sentiment_trend = defaultdict(lambda: {'正面': 0, '中性': 0, '负面': 0})
    for dt, sentiment, _ in valid_data:
        hour = dt.hour
        sentiment_trend[hour][sentiment] += 1

    hours = sorted(sentiment_trend.keys())
    positive_counts = [sentiment_trend[h]['正面'] for h in hours]
    neutral_counts = [sentiment_trend[h]['中性'] for h in hours]
    negative_counts = [sentiment_trend[h]['负面'] for h in hours]

    plt.figure(figsize=(12, 6))
    plt.plot(hours, positive_counts, label='正面情感', marker='o', linewidth=2)
    plt.plot(hours, neutral_counts, label='中性情感', marker='s', linewidth=2)
    plt.plot(hours, negative_counts, label='负面情感', marker='^', linewidth=2)

    plt.xlabel('时间（小时）')
    plt.ylabel('评论数量')
    plt.title(f'{name}评论情感趋势分析')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(hours)

    for i, (h, pos, neu, neg) in enumerate(zip(hours, positive_counts, neutral_counts, negative_counts)):
        plt.text(h, pos, str(pos), ha='center', va='bottom')
        plt.text(h, neu, str(neu), ha='center', va='bottom')
        plt.text(h, neg, str(neg), ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(f"./output/{name}_情感趋势图.png", dpi=300, bbox_inches='tight')
    plt.close()

def main_analysis(event_name, path):
    """主分析函数"""
    print(f"开始分析事件: {event_name}")

    data_loader = CustomDataLoader()
    comments, timestamps = data_loader.get_all_data_for_analysis(event_name)

    if len(comments) == 0:
        print(f"事件 '{event_name}' 没有找到数据")
        return None

    print("正在加载模型...")
    model_trainer = ModelTrainer()
    tokenizer, model = model_trainer.load_trained_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    print("✅ 使用BERT模型进行情感分析...")
    sentiments = analyze_sentiment_bert(comments, tokenizer, model, device)

    positive_count = sentiments.count('正面')
    negative_count = sentiments.count('负面')
    neutral_count = sentiments.count('中性')

    print(f"情感分析完成！")
    print(f"正面: {positive_count} 条")
    print(f"负面: {negative_count} 条")
    print(f"中性: {neutral_count} 条")

    valid_data = []
    for time_str, sentiment, comment in zip(timestamps, sentiments, comments):
        dt = parse_time(time_str)
        if dt is not None:
            valid_data.append((dt, sentiment, comment))

    if valid_data:
        plot_sentiment_trend(event_name, valid_data)

    viz_data = generate_visualization_data(event_name, comments, sentiments, timestamps)

    output_dir = "./output"
    import os
    os.makedirs(output_dir, exist_ok=True)

    viz_filename = f"{output_dir}/{event_name}_可视化数据.json"
    with open(viz_filename, 'w', encoding='utf-8') as f:
        json.dump(viz_data, f, ensure_ascii=False, indent=2)

    results = []
    for comment, timestamp, sentiment in zip(comments, timestamps, sentiments):
        results.append({
            'comment': comment,
            'timestamp': timestamp,
            'sentiment': sentiment
        })

    with open(f'{output_dir}/{event_name}_情感分析结果.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"分析完成！结果已保存到 {output_dir}/ 目录")
    return viz_data

if __name__ == "__main__":
    name = "泡泡玛特直播事故"
    viz_data = main_analysis(name, "./清洗后的数据源/泡泡玛特直播事故.json")