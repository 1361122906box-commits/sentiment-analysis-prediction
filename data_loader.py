# data_loader.py    数据加载和预处理脚本

# data_loader.py
import json
import pandas as pd
import os
from collections import Counter
import chardet


class DataLoader:
    def __init__(self):
        self.train_path = "训练集/train_dataset.json"
        self.test_path = "训练集/test_dataset.json"
        self.gold_test_path = "训练集/manual_annotation_template.csv"

    def detect_encoding(self, file_path):
        """检测文件编码"""
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding']
                confidence = result['confidence']
                print(f"检测到文件编码: {encoding} (置信度: {confidence:.2f})")
                return encoding
        except Exception as e:
            print(f"编码检测失败: {e}")
            return 'utf-8'

    def load_train_data(self):
        """加载训练集数据"""
        print("加载训练集数据...")
        try:
            with open(self.train_path, 'r', encoding='utf-8') as f:
                train_data = json.load(f)

            train_comments = [item['content'] for item in train_data]
            train_topics = [item['topic'] for item in train_data]

            print(f"训练集大小: {len(train_comments)} 条")
            print("训练集话题分布:", dict(Counter(train_topics)))

            return train_comments, train_topics, train_data
        except Exception as e:
            print(f"加载训练集失败: {e}")
            return [], [], []

    def load_test_data(self):
        """加载测试集数据"""
        print("加载测试集数据...")
        try:
            with open(self.test_path, 'r', encoding='utf-8') as f:
                test_data = json.load(f)

            test_comments = [item['content'] for item in test_data]
            test_topics = [item['topic'] for item in test_data]

            print(f"测试集大小: {len(test_comments)} 条")
            print("测试集话题分布:", dict(Counter(test_topics)))

            return test_comments, test_topics, test_data
        except Exception as e:
            print(f"加载测试集失败: {e}")
            return [], [], []

    def load_gold_test_data(self):
        """加载黄金测试集（已标注）"""
        print("加载黄金测试集...")
        try:
            # 检测文件编码
            encoding = self.detect_encoding(self.gold_test_path)

            # 尝试不同的编码方式读取CSV文件
            encodings_to_try = [encoding, 'gbk', 'gb2312', 'utf-8', 'latin-1']

            for enc in encodings_to_try:
                try:
                    print(f"尝试使用 {enc} 编码读取文件...")
                    gold_df = pd.read_csv(self.gold_test_path, encoding=enc)

                    # 检查必要的列是否存在
                    required_columns = ['content', 'sentiment']
                    missing_columns = [col for col in required_columns if col not in gold_df.columns]

                    if missing_columns:
                        print(f"缺少必要的列: {missing_columns}")
                        print(f"文件中的列: {gold_df.columns.tolist()}")
                        continue

                    # 清理数据
                    gold_df = gold_df.dropna(subset=['content', 'sentiment'])

                    # 标准化情感标签（处理大小写和空格）
                    gold_df['sentiment'] = gold_df['sentiment'].astype(str).str.strip().str.lower()

                    # 过滤有效的情感标签
                    valid_sentiments = ['positive', 'negative', 'neutral']
                    gold_df = gold_df[gold_df['sentiment'].isin(valid_sentiments)]

                    if len(gold_df) == 0:
                        print("警告: 没有找到有效的情感标签数据")
                        print(f"找到的情感标签: {gold_df['sentiment'].unique()}")
                        continue

                    gold_comments = gold_df['content'].tolist()
                    gold_sentiments = gold_df['sentiment'].tolist()

                    # 处理topic列（如果存在）
                    if 'topic' in gold_df.columns:
                        gold_topics = gold_df['topic'].tolist()
                    else:
                        gold_topics = ['unknown'] * len(gold_comments)

                    print(f"黄金测试集大小: {len(gold_comments)} 条")
                    print("黄金测试集情感分布:", dict(Counter(gold_sentiments)))
                    print("黄金测试集话题分布:", dict(Counter(gold_topics)))

                    return gold_comments, gold_sentiments, gold_topics, gold_df

                except UnicodeDecodeError:
                    print(f"编码 {enc} 失败，尝试下一个...")
                    continue
                except Exception as e:
                    print(f"使用编码 {enc} 读取失败: {e}")
                    continue

            # 如果所有编码都失败，尝试手动读取
            print("尝试手动读取文件...")
            return self._manual_read_csv()

        except Exception as e:
            print(f"加载黄金测试集失败: {e}")
            return [], [], [], pd.DataFrame()

    def _manual_read_csv(self):
        """手动读取CSV文件（备用方法）"""
        try:
            with open(self.gold_test_path, 'rb') as f:
                content = f.read()

            # 尝试不同的编码
            for encoding in ['gbk', 'gb2312', 'utf-8', 'latin-1']:
                try:
                    text = content.decode(encoding)
                    lines = text.split('\n')

                    # 解析CSV（简单版本）
                    data = []
                    headers = None

                    for i, line in enumerate(lines):
                        if not line.strip():
                            continue

                        if i == 0:  # 标题行
                            headers = line.strip().split(',')
                            print(f"检测到列: {headers}")
                        else:
                            values = line.strip().split(',')
                            if len(values) >= 2:  # 至少包含内容和情感
                                row_data = {}
                                for j, header in enumerate(headers):
                                    if j < len(values):
                                        row_data[header] = values[j]

                                # 检查是否有内容和情感
                                if 'content' in row_data and 'sentiment' in row_data:
                                    data.append(row_data)

                    if data:
                        gold_df = pd.DataFrame(data)
                        gold_comments = gold_df['content'].tolist()
                        gold_sentiments = gold_df['sentiment'].tolist()
                        gold_topics = gold_df.get('topic', ['unknown'] * len(gold_comments)).tolist()

                        print(f"手动读取成功: {len(gold_comments)} 条数据")
                        return gold_comments, gold_sentiments, gold_topics, gold_df

                except Exception as e:
                    print(f"手动读取编码 {encoding} 失败: {e}")
                    continue

            return [], [], [], pd.DataFrame()

        except Exception as e:
            print(f"手动读取完全失败: {e}")
            return [], [], [], pd.DataFrame()

    def get_all_data_for_analysis(self, event_name):
        """获取指定事件的所有数据用于分析"""
        train_comments, train_topics, train_data = self.load_train_data()
        test_comments, test_topics, test_data = self.load_test_data()

        # 合并训练集和测试集中指定事件的数据
        all_comments = []
        all_timestamps = []

        for data in [train_data, test_data]:
            for item in data:
                if item['topic'] == event_name:
                    all_comments.append(item['content'])
                    all_timestamps.append(item['time'])

        print(f"事件 '{event_name}' 总数据量: {len(all_comments)} 条")

        return all_comments, all_timestamps

    def analyze_data_files(self):
        """分析数据文件状态"""
        print("=" * 50)
        print("数据文件分析")
        print("=" * 50)

        files_to_check = [
            ("训练集", self.train_path),
            ("测试集", self.test_path),
            ("黄金测试集", self.gold_test_path)
        ]

        for name, path in files_to_check:
            if os.path.exists(path):
                file_size = os.path.getsize(path) / 1024  # KB
                print(f"✅ {name}: {path} ({file_size:.1f} KB)")

                if path.endswith('.csv'):
                    encoding = self.detect_encoding(path)
                    print(f"   编码: {encoding}")
            else:
                print(f"❌ {name}: 文件不存在 - {path}")


if __name__ == "__main__":
    loader = DataLoader()

    # 首先分析文件状态
    loader.analyze_data_files()

    print("\n" + "=" * 50)
    print("开始加载数据...")
    print("=" * 50)

    # 测试数据加载
    train_comments, train_topics, train_data = loader.load_train_data()
    test_comments, test_topics, test_data = loader.load_test_data()
    gold_comments, gold_sentiments, gold_topics, gold_df = loader.load_gold_test_data()

    print("\n" + "=" * 50)
    print("数据加载完成")
    print("=" * 50)