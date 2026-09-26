# visualization_data_exporter.py
import json
import pandas as pd
from collections import Counter
import jieba
import re
from datetime import datetime
import os
import numpy as np
from datetime import timedelta


# 在文件顶部添加导入
try:
    from lstm_predictor import get_predictor
    LSTM_AVAILABLE = True
except ImportError:
    print("⚠️ LSTM预测模块不可用，将使用简单预测")
    LSTM_AVAILABLE = False


def initialize_jieba():
    """初始化jieba分词"""
    try:
        # 添加一些网络热词到词典
        jieba.add_word('泡泡玛特')
        jieba.add_word('福建舰')
        jieba.add_word('A股')
        jieba.add_word('中美贸易')
        jieba.add_word('网络谣言')
        jieba.add_word('直播事故')
    except:
        pass


def clean_text(text):
    """清洗文本"""
    if not isinstance(text, str):
        return ""
    # 移除URL、特殊字符等
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^\w\u4e00-\u9fff]', ' ', text)
    return text.strip()


def parse_time(time_str):
    """统一的时间解析函数（与main_analysis.py保持一致）"""
    if not time_str or pd.isna(time_str):
        return None

    time_str = str(time_str).strip()

    # 尝试多种时间格式
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
            return datetime.strptime(time_str, fmt)
        except ValueError:
            continue

    print(f"无法解析的时间格式: {time_str}")
    return None


def generate_visualization_data(name, comments, sentiments, timestamps, predictions=None):
    """生成可视化所需的所有数据"""

    initialize_jieba()

    # 1. 基础统计信息
    total_comments = len(comments)
    sentiment_counts = Counter(sentiments)
    positive_pct = sentiment_counts['正面'] / total_comments * 100 if total_comments > 0 else 0
    negative_pct = sentiment_counts['负面'] / total_comments * 100 if total_comments > 0 else 0
    neutral_pct = sentiment_counts['中性'] / total_comments * 100 if total_comments > 0 else 0

    # 2. 情感分布数据（饼图用）
    sentiment_distribution = {
        'categories': ['正面', '负面', '中性'],
        'values': [
            sentiment_counts['正面'],
            sentiment_counts['负面'],
            sentiment_counts['中性']
        ],
        'percentages': [
            round(positive_pct, 1),
            round(negative_pct, 1),
            round(neutral_pct, 1)
        ],
        'colors': ['#52c41a', '#ff4d4f', '#faad14']  # 绿、红、黄
    }

    # 3. 实时评论数据（滚动显示用）
    live_comments = []
    for i, (comment, sentiment, timestamp) in enumerate(zip(comments[:100], sentiments[:100], timestamps[:100])):
        # 清理评论内容
        clean_comment = clean_text(comment)
        if len(clean_comment) > 0:
            live_comments.append({
                'id': i + 1,
                'content': clean_comment[:60] + '...' if len(clean_comment) > 60 else clean_comment,
                'sentiment': sentiment,
                'time': timestamp.split(' ')[1] if ' ' in timestamp else timestamp,  # 只显示时间部分
                'avatar_color': '#ff4d4f' if sentiment == '负面' else '#52c41a' if sentiment == '正面' else '#faad14'
            })

    # 4. 关键词提取（词云用）
    all_text = ' '.join([clean_text(comment) for comment in comments])
    words = jieba.cut(all_text)

    # 过滤停用词和短词
    stop_words = {
        '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个', '上',
        '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己',
        '这个', '但', '什么', '现在', '知道', '就是', '可以', '才', '年', '哦', '吗', '嗯',
        '啊', '呢', '吧', '这', '那', '他', '她', '它', '我们', '他们', '这样', '这种',
        '这些', '那些', '因为', '所以', '如果', '虽然', '然后', '已经', '还是', '应该'
    }

    word_freq = Counter()
    for word in words:
        word = word.strip()
        if (len(word) > 1 and
                word not in stop_words and
                not word.isdigit() and
                not re.match(r'^[a-zA-Z0-9]$', word)):
            word_freq[word] += 1

    # 取前50个关键词
    top_keywords = dict(sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:50])

    # 5. 时间序列情感数据（趋势图用）- 保持不变
    hourly_data = {}
    valid_count = 0
    for timestamp, sentiment in zip(timestamps, sentiments):
        dt = parse_time(timestamp)
        if dt is not None:
            valid_count += 1
            hour_key = dt.strftime('%Y-%m-%d %H:00')
            if hour_key not in hourly_data:
                hourly_data[hour_key] = {'positive': 0, 'negative': 0, 'neutral': 0, 'total': 0}

            if sentiment == '正面':
                hourly_data[hour_key]['positive'] += 1
            elif sentiment == '负面':
                hourly_data[hour_key]['negative'] += 1
            else:
                hourly_data[hour_key]['neutral'] += 1
            hourly_data[hour_key]['total'] += 1

    print(f"时间序列数据: 有效时间戳 {valid_count}/{len(timestamps)}")

    # 计算每小时情感得分
    time_series_data = []
    time_series_scores = []  # 单独保存得分序列用于LSTM

    for hour, counts in sorted(hourly_data.items()):
        if counts['total'] > 0:
            sentiment_score = (counts['positive'] - counts['negative']) / counts['total']
            time_series_data.append({
                'time': hour,
                'sentiment_score': round(sentiment_score, 3),
                'positive': counts['positive'],
                'negative': counts['negative'],
                'neutral': counts['neutral'],
                'total': counts['total']
            })
            time_series_scores.append(sentiment_score)

    # 6. LSTM情感预测
    prediction_data = []
    # 动态预测范围：最少6小时，最多24小时，基于历史数据量调整
    future_hours = min(600, max(6, len(time_series_scores) // 2))

    if time_series_data and len(time_series_scores) >= 8:  # 至少有8个数据点
        try:
            last_time_str = time_series_data[-1]['time']
            last_time = datetime.strptime(last_time_str, '%Y-%m-%d %H:00')

            if LSTM_AVAILABLE and len(time_series_scores) >= 13:  # 足够数据时使用LSTM
                predictor = get_predictor()

                # 训练LSTM模型（每个事件单独训练）
                success = predictor.train(time_series_scores, epochs=150, seq_length=12)

                if success:
                    # 使用LSTM进行预测
                    future_predictions = predictor.predict(time_series_scores, future_steps=future_hours)
                    prediction_method = "LSTM"
                else:
                    # LSTM训练失败，使用简单预测
                    future_predictions = simple_trend_predict(time_series_scores, future_hours)
                    prediction_method = "简单预测(LSTM训练失败)"
            else:
                # 数据不足，使用简单预测
                future_predictions = simple_trend_predict(time_series_scores, future_hours)
                prediction_method = "简单预测(数据不足)"

            # 生成预测数据点
            for i, pred_score in enumerate(future_predictions, 1):
                pred_time = last_time + timedelta(hours=i)
                prediction_data.append({
                    'time': pred_time.strftime('%Y-%m-%d %H:00'),
                    'sentiment_score': round(float(pred_score), 3),
                    'type': 'prediction'
                })

            print(f"✅ {prediction_method}生成预测数据: {len(prediction_data)}条")

        except Exception as e:
            print(f"❌ 预测失败: {e}")
            # 失败时使用简单预测
            prediction_data = generate_simple_predictions(time_series_data, future_hours)
    else:
        print("⚠️ 数据不足，跳过预测")
        prediction_data = generate_simple_predictions(time_series_data, future_hours)

    # 7. 组装完整的数据结构 - 保持不变
    visualization_data = {
        'event_info': {
            'name': name,
            'total_comments': total_comments,
            'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'positive_percentage': round(positive_pct, 1),
            'negative_percentage': round(negative_pct, 1),
            'neutral_percentage': round(neutral_pct, 1),
            'sentiment_summary': get_sentiment_summary(positive_pct, negative_pct, neutral_pct)
        },
        'sentiment_distribution': sentiment_distribution,
        'live_comments': live_comments,
        'keywords': top_keywords,
        'time_series': time_series_data,
        'predictions': prediction_data
    }

    print(f"可视化数据生成完成: time_series={len(time_series_data)}, predictions={len(prediction_data)}")
    return visualization_data


def simple_trend_predict(time_series_scores, future_steps=6):
    """简单趋势预测函数"""
    if not time_series_scores:
        return [0] * future_steps

    # 计算近期趋势
    recent_scores = time_series_scores[-4:]  # 最近4个点
    avg_score = sum(recent_scores) / len(recent_scores)

    # 计算趋势方向
    if len(time_series_scores) >= 2:
        trend = time_series_scores[-1] - time_series_scores[-2]
    else:
        trend = 0

    predictions = []
    for i in range(1, future_steps + 1):
        # 基于当前值和趋势进行预测，逐渐减弱
        pred = avg_score + trend * (0.7 ** i)
        # 限制在合理范围内
        pred = max(-1.0, min(1.0, pred))
        predictions.append(pred)

    return predictions


def generate_simple_predictions(time_series_data, future_hours):
    """生成简单预测数据（备选）"""
    prediction_data = []
    if not time_series_data:
        return prediction_data

    try:
        last_time_str = time_series_data[-1]['time']
        last_time = datetime.strptime(last_time_str, '%Y-%m-%d %H:00')
        last_score = time_series_data[-1]['sentiment_score']

        for i in range(1, future_hours + 1):
            pred_time = last_time + timedelta(hours=i)
            # 简单地向中性回归
            pred_score = last_score * (0.9 ** i)
            prediction_data.append({
                'time': pred_time.strftime('%Y-%m-%d %H:00'),
                'sentiment_score': round(pred_score, 3),
                'type': 'prediction'
            })
    except:
        pass

    return prediction_data


def get_sentiment_summary(positive_pct, negative_pct, neutral_pct):
    """生成情感摘要"""
    if negative_pct > 50:
        return "舆情以负面情绪为主，需要重点关注"
    elif positive_pct > 50:
        return "舆情以正面情绪为主，整体积极"
    elif neutral_pct > 50:
        return "舆情以中性情绪为主，相对平稳"
    else:
        return "舆情情绪分布较为均衡"


def export_all_events_data():
    """导出所有事件的可视化数据"""
    from main_analysis import main_analysis

    events_data = {}

    # 事件配置
    events = {
        "春晚机器人": "./清洗后的数据源/2025春晚机器人.json",
        "经济政策解读": "./清洗后的数据源/2025经济政策解读.json",
        "中美贸易谈判": "./清洗后的数据源/2025中美贸易谈判.json",
        "A股半日成交1.25万亿缩量711亿": "./清洗后的数据源/A股半日成交1.25万亿缩量711亿.json",
        "福建舰入列": "./清洗后的数据源/福建舰入列.json",
        "公安机关查处网络谣言": "./清洗后的数据源/公安机关查处网络谣言.json",
        "泡泡玛特直播事故": "./清洗后的数据源/泡泡玛特直播事故.json",
        "日本71岁女子杀害102岁母亲": "./清洗后的数据源/日本71岁女子杀害102岁母亲.json"
    }

    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

    for event_name, file_path in events.items():
        print(f"\n正在处理事件: {event_name}")

        try:
            if os.path.exists(file_path):
                # 运行分析并获取可视化数据
                viz_data = main_analysis(event_name, file_path)
                events_data[event_name] = viz_data

                print(f"✅ {event_name} 数据处理完成")
            else:
                print(f"❌ 文件不存在: {file_path}")

        except Exception as e:
            print(f"❌ 处理事件 {event_name} 时出错: {e}")
            continue

    # 保存所有事件的聚合数据
    all_events_file = f'{output_dir}/ALL_EVENTS_可视化数据.json'
    with open(all_events_file, 'w', encoding='utf-8') as f:
        json.dump(events_data, f, ensure_ascii=False, indent=2)

    print(f"\n🎉 所有事件的可视化数据已生成！")
    print(f"📁 数据文件: {all_events_file}")

    return events_data


if __name__ == "__main__":
    # 生成所有事件的数据
    export_all_events_data()