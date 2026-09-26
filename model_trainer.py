# model_trainer.py 模型训练脚本
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader as TorchDataLoader  # 重命名
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from data_loader import DataLoader as CustomDataLoader  # 重命名
import os
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 支持中英文标签映射
        self.label_map = {
            'negative': 0, '负面': 0,
            'neutral': 1, '中性': 1,
            'positive': 2, '正面': 2
        }

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # 处理标签（支持中英文）
        if isinstance(label, str):
            # 清理标签
            label = label.strip().lower()
            # 检查标签是否在映射中
            if label in self.label_map:
                label_id = self.label_map[label]
            else:
                # 如果标签不在映射中，默认设为中性
                print(f"警告: 未知标签 '{label}'，设为中性")
                label_id = 1  # 中性
        else:
            # 如果已经是数字，直接使用
            label_id = int(label)

        encoding = self.tokenizer.encode_plus(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label_id, dtype=torch.long)
        }


class ModelTrainer:
    def __init__(self):
        self.data_loader = CustomDataLoader()  # 使用重命名后的类
        self.local_model_path = "./finetuned_models/bert-base-chinese"
        self.finetuned_model_path = "./finetuned_models/bert-sentiment-weibo"

    def prepare_training_data(self):
        """准备训练数据"""
        print("准备训练数据...")

        # 加载训练集
        train_comments, train_topics, train_data = self.data_loader.load_train_data()

        if len(train_comments) == 0:
            print("❌ 训练集为空，无法训练")
            return [], [], [], []

        # 为训练集生成初步标签（使用规则方法）
        from main_analysis import analyze_sentiment_rule_based
        print("为训练集生成情感标签...")
        train_sentiments = analyze_sentiment_rule_based(train_comments)

        # 加载黄金测试集作为验证集
        gold_comments, gold_sentiments, gold_topics, gold_df = self.data_loader.load_gold_test_data()

        if len(gold_comments) == 0:
            print("❌ 黄金测试集为空，无法验证")
            return [], [], [], []

        # 标准化标签格式（确保都是英文）
        train_sentiments = self.standardize_labels(train_sentiments)
        gold_sentiments = self.standardize_labels(gold_sentiments)

        print(f"训练集大小: {len(train_comments)}")
        print(f"验证集大小: {len(gold_comments)}")
        print("训练集情感分布:", dict(pd.Series(train_sentiments).value_counts()))
        print("验证集情感分布:", dict(pd.Series(gold_sentiments).value_counts()))

        return train_comments, train_sentiments, gold_comments, gold_sentiments

    def standardize_labels(self, labels):
        """标准化情感标签为英文"""
        label_mapping = {
            '负面': 'negative',
            '中性': 'neutral',
            '正面': 'positive',
            'negative': 'negative',
            'neutral': 'neutral',
            'positive': 'positive'
        }

        standardized = []
        for label in labels:
            label_str = str(label).strip()
            if label_str in label_mapping:
                standardized.append(label_mapping[label_str])
            else:
                # 未知标签，默认设为neutral
                print(f"警告: 未知标签 '{label_str}'，设为neutral")
                standardized.append('neutral')

        return standardized

    def create_data_loader(self, dataset, batch_size=16, shuffle=True):
        """创建DataLoader，兼容不同PyTorch版本"""
        try:
            # 使用重命名后的TorchDataLoader
            return TorchDataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        except TypeError:
            try:
                # 尝试旧版本参数
                return TorchDataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
            except TypeError:
                # 最简版本
                return TorchDataLoader(dataset, batch_size=batch_size)

    def train_bert_model(self, epochs=3, batch_size=16):
        """训练BERT模型"""
        print("开始训练BERT模型...")

        # 准备数据
        train_comments, train_sentiments, val_comments, val_sentiments = self.prepare_training_data()

        if len(train_comments) == 0 or len(val_comments) == 0:
            print("❌ 数据准备失败，无法训练")
            return None, None

        # 检查标签分布
        print("训练前标签检查:")
        print(f"训练集标签类型: {set(train_sentiments)}")
        print(f"验证集标签类型: {set(val_sentiments)}")

        # 加载模型和分词器
        print("加载BERT模型和分词器...")
        tokenizer = AutoTokenizer.from_pretrained(self.local_model_path)
        model = AutoModelForSequenceClassification.from_pretrained(self.local_model_path, num_labels=3)

        # 创建数据集
        train_dataset = SentimentDataset(train_comments, train_sentiments, tokenizer)
        val_dataset = SentimentDataset(val_comments, val_sentiments, tokenizer)

        # 测试数据集样本
        print("测试数据集样本...")
        try:
            sample = train_dataset[0]
            print(f"数据集样本测试成功: input_ids形状 {sample['input_ids'].shape}")
        except Exception as e:
            print(f"数据集测试失败: {e}")
            return None, None

        # 创建DataLoader
        print("创建DataLoader...")
        train_loader = self.create_data_loader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = self.create_data_loader(val_dataset, batch_size=batch_size, shuffle=False)

        print(f"训练批次数量: {len(train_loader)}")
        print(f"验证批次数量: {len(val_loader)}")

        # 设置训练参数
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")
        model.to(device)

        # 使用torch.optim中的AdamW
        optimizer = AdamW(model.parameters(), lr=2e-5)
        total_steps = len(train_loader) * epochs

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        # 训练循环
        best_accuracy = 0
        for epoch in range(epochs):
            print(f"\n--- Epoch {epoch + 1}/{epochs} ---")
            model.train()
            total_loss = 0
            total_correct = 0
            total_samples = 0

            for batch_idx, batch in enumerate(train_loader):
                try:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)

                    optimizer.zero_grad()
                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    logits = outputs.logits

                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                    total_loss += loss.item()

                    # 计算准确率
                    predictions = torch.argmax(logits, dim=1)
                    correct = (predictions == labels).sum().item()
                    total_correct += correct
                    total_samples += len(labels)

                    if (batch_idx + 1) % 10 == 0:
                        accuracy = correct / len(labels)
                        print(
                            f'Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')

                except Exception as e:
                    print(f"批次 {batch_idx} 处理失败: {e}")
                    continue

            if total_samples > 0:
                epoch_accuracy = total_correct / total_samples
                avg_loss = total_loss / len(train_loader)
                print(f'Epoch {epoch + 1} 完成, 平均Loss: {avg_loss:.4f}, 训练准确率: {epoch_accuracy:.4f}')
            else:
                print(f'Epoch {epoch + 1} 完成, 无有效数据')
                continue

            # 在验证集上评估
            val_accuracy, val_report = self.evaluate_model(model, val_loader, device)
            print(f'验证集准确率: {val_accuracy:.4f}')

            # 保存最佳模型
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                self.save_model(model, tokenizer)
                print(f"新的最佳模型已保存，准确率: {best_accuracy:.4f}")

        print(f"\n训练完成！最佳验证集准确率: {best_accuracy:.4f}")
        return model, tokenizer

    def evaluate_model(self, model, data_loader, device):
        """评估模型性能"""
        model.eval()
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)

        report = classification_report(all_labels, all_predictions,
                                       target_names=['negative', 'neutral', 'positive'],
                                       zero_division=0)

        print(f"准确率: {accuracy:.4f}")
        print(f"精确率: {precision:.4f}")
        print(f"召回率: {recall:.4f}")
        print(f"F1-score: {f1:.4f}")

        return accuracy, report

    def save_model(self, model, tokenizer):
        """保存模型"""
        os.makedirs(os.path.dirname(self.finetuned_model_path), exist_ok=True)
        model.save_pretrained(self.finetuned_model_path)
        tokenizer.save_pretrained(self.finetuned_model_path)
        print(f"模型已保存到: {self.finetuned_model_path}")

    def load_trained_model(self):
        """加载训练好的模型"""
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.finetuned_model_path)
            model = AutoModelForSequenceClassification.from_pretrained(self.finetuned_model_path)
            print("✅ 加载微调后的BERT模型")
            return tokenizer, model
        except Exception as e:
            print(f"❌ 加载微调模型失败: {e}")
            print("✅ 加载原始BERT模型")
            tokenizer = AutoTokenizer.from_pretrained(self.local_model_path)
            model = AutoModelForSequenceClassification.from_pretrained(self.local_model_path, num_labels=3)
            return tokenizer, model


def main():
    """主训练函数"""
    print("=" * 60)
    print("BERT模型训练开始")
    print("=" * 60)

    trainer = ModelTrainer()

    try:
        model, tokenizer = trainer.train_bert_model(epochs=3)
        if model is not None:
            print("\n🎉 模型训练成功完成！")
        else:
            print("\n❌ 模型训练失败")
    except Exception as e:
        print(f"\n❌ 训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()