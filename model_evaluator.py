# model_evaluator.py
import pandas as pd
import torch
from torch.utils.data import DataLoader as TorchDataLoader  # 重命名
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix
from data_loader import DataLoader as CustomDataLoader  # 重命名
from model_trainer import ModelTrainer
import os
import matplotlib.pyplot as plt
import seaborn as sns
from font_utils import setup_chinese_font, get_chinese_font_properties

# 设置中文字体
setup_chinese_font()


class ModelEvaluator:
    def __init__(self):
        self.data_loader = CustomDataLoader()  # 使用重命名后的类
        self.model_trainer = ModelTrainer()

    def evaluate_bert(self, test_comments, test_sentiments):
        """评估BERT模型"""
        # 加载模型
        tokenizer, model = self.model_trainer.load_trained_model()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        # 创建数据集
        from model_trainer import SentimentDataset
        test_dataset = SentimentDataset(test_comments, test_sentiments, tokenizer)
        test_loader = TorchDataLoader(test_dataset, batch_size=16, shuffle=False)  # 使用重命名后的类

        # 预测
        model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(logits, dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())

        # 计算指标
        metrics = self.calculate_metrics(all_labels, all_predictions, 'BERT')

        # 详细报告
        print("\nBERT详细分类报告:")
        print(classification_report(all_labels, all_predictions,
                                    target_names=['negative', 'neutral', 'positive'],
                                    zero_division=0))

        # 将数字预测转换回标签
        id_to_label = {0: 'negative', 1: 'neutral', 2: 'positive'}
        bert_pred_labels = [id_to_label[pred] for pred in all_predictions]

        return metrics, bert_pred_labels


    def evaluate_all_models(self):
        """评估所有模型性能"""
        print("=" * 60)
        print("开始模型评估...")
        print("=" * 60)

        # 加载黄金测试集
        gold_comments, gold_sentiments, gold_topics, gold_df = self.data_loader.load_gold_test_data()

        if len(gold_comments) == 0:
            print("❌ 黄金测试集为空，无法进行评估")
            return None

        print(f"使用黄金测试集进行评估，样本数量: {len(gold_comments)}")

        # 1. 评估SVM模型
        print("\n" + "=" * 50)
        print("评估SVM模型...")
        print("=" * 50)
        svm_metrics, svm_predictions = self.evaluate_svm(gold_comments, gold_sentiments)

        # 2. 评估BERT模型
        print("\n" + "=" * 50)
        print("评估BERT模型...")
        print("=" * 50)
        bert_metrics, bert_predictions = self.evaluate_bert(gold_comments, gold_sentiments)

        # 3. 创建对比表格
        comparison_df = self.create_comparison_table(svm_metrics, bert_metrics)

        # 4. 绘制混淆矩阵
        self.plot_confusion_matrices(gold_sentiments, svm_predictions, bert_predictions)

        # 5. 保存结果
        self.save_results(comparison_df, svm_metrics, bert_metrics, gold_sentiments, svm_predictions, bert_predictions)

        return comparison_df

    def evaluate_svm(self, test_comments, test_sentiments):
        """评估SVM模型"""
        print("训练SVM模型...")

        # 加载训练数据
        train_comments, train_sentiments, _, _ = self.model_trainer.prepare_training_data()

        if len(train_comments) == 0:
            print("❌ 训练数据为空，无法训练SVM")
            return {}, []

        # 将标签转换为数字
        label_to_id = {'negative': 0, 'neutral': 1, 'positive': 2}
        train_labels = [label_to_id[sentiment] for sentiment in train_sentiments]
        test_labels = [label_to_id[sentiment] for sentiment in test_sentiments]

        print(f"SVM训练集大小: {len(train_comments)}")
        print(f"SVM测试集大小: {len(test_comments)}")

        # 特征提取
        vectorizer = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.8)
        X_train = vectorizer.fit_transform(train_comments)
        X_test = vectorizer.transform(test_comments)

        print(f"特征维度: {X_train.shape[1]}")

        # 训练SVM
        svm_model = SVC(kernel='linear', random_state=42, probability=True)
        svm_model.fit(X_train, train_labels)

        # 预测
        svm_predictions = svm_model.predict(X_test)
        svm_probabilities = svm_model.predict_proba(X_test)

        # 计算指标
        metrics = self.calculate_metrics(test_labels, svm_predictions, 'SVM')

        # 详细报告
        print("\nSVM详细分类报告:")
        print(classification_report(test_labels, svm_predictions,
                                    target_names=['negative', 'neutral', 'positive'],
                                    zero_division=0))

        # 将数字预测转换回标签
        id_to_label = {0: 'negative', 1: 'neutral', 2: 'positive'}
        svm_pred_labels = [id_to_label[pred] for pred in svm_predictions]

        return metrics, svm_pred_labels

    def calculate_metrics(self, true_labels, predictions, model_name):
        """计算评估指标"""
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions, average='weighted', zero_division=0)
        recall = recall_score(true_labels, predictions, average='weighted', zero_division=0)
        f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)

        # 计算每个类别的指标
        precision_per_class = precision_score(true_labels, predictions, average=None, zero_division=0)
        recall_per_class = recall_score(true_labels, predictions, average=None, zero_division=0)
        f1_per_class = f1_score(true_labels, predictions, average=None, zero_division=0)

        return {
            '模型': model_name,
            '准确率': f"{accuracy * 100:.1f}%",
            '精确率': f"{precision * 100:.1f}%",
            '召回率': f"{recall * 100:.1f}%",
            'F1-score': f"{f1 * 100:.1f}%",
            '详细指标': {
                'negative': {
                    'precision': f"{precision_per_class[0] * 100:.1f}%",
                    'recall': f"{recall_per_class[0] * 100:.1f}%",
                    'f1': f"{f1_per_class[0] * 100:.1f}%"
                },
                'neutral': {
                    'precision': f"{precision_per_class[1] * 100:.1f}%",
                    'recall': f"{recall_per_class[1] * 100:.1f}%",
                    'f1': f"{f1_per_class[1] * 100:.1f}%"
                },
                'positive': {
                    'precision': f"{precision_per_class[2] * 100:.1f}%",
                    'recall': f"{recall_per_class[2] * 100:.1f}%",
                    'f1': f"{f1_per_class[2] * 100:.1f}%"
                }
            }
        }

    def create_comparison_table(self, svm_metrics, bert_metrics):
        """创建模型对比表格"""
        # 创建主要指标对比
        main_metrics = [
            {
                '模型': svm_metrics['模型'],
                '准确率': svm_metrics['准确率'],
                '精确率': svm_metrics['精确率'],
                '召回率': svm_metrics['召回率'],
                'F1-score': svm_metrics['F1-score']
            },
            {
                '模型': bert_metrics['模型'],
                '准确率': bert_metrics['准确率'],
                '精确率': bert_metrics['精确率'],
                '召回率': bert_metrics['召回率'],
                'F1-score': bert_metrics['F1-score']
            }
        ]

        comparison_df = pd.DataFrame(main_metrics)

        print("\n" + "=" * 60)
        print("模型性能对比")
        print("=" * 60)
        print(comparison_df.to_string(index=False))

        # 打印详细指标
        print("\n" + "=" * 60)
        print("详细类别指标对比")
        print("=" * 60)

        for sentiment in ['negative', 'neutral', 'positive']:
            print(f"\n{sentiment.upper()} 类别:")
            print(f"SVM - 精确率: {svm_metrics['详细指标'][sentiment]['precision']}, "
                  f"召回率: {svm_metrics['详细指标'][sentiment]['recall']}, "
                  f"F1: {svm_metrics['详细指标'][sentiment]['f1']}")
            print(f"BERT - 精确率: {bert_metrics['详细指标'][sentiment]['precision']}, "
                  f"召回率: {bert_metrics['详细指标'][sentiment]['recall']}, "
                  f"F1: {bert_metrics['详细指标'][sentiment]['f1']}")

        return comparison_df

    def plot_confusion_matrices(self, true_labels, svm_predictions, bert_predictions):
        """绘制混淆矩阵"""
        # 将标签转换为数字
        label_to_id = {'negative': 0, 'neutral': 1, 'positive': 2}
        true_labels_numeric = [label_to_id[label] for label in true_labels]
        svm_predictions_numeric = [label_to_id[label] for label in svm_predictions]
        bert_predictions_numeric = [label_to_id[label] for label in bert_predictions]

        # 创建子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # SVM混淆矩阵
        svm_cm = confusion_matrix(true_labels_numeric, svm_predictions_numeric)
        sns.heatmap(svm_cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                    xticklabels=['负面', '中性', '正面'],
                    yticklabels=['负面', '中性', '正面'])
        ax1.set_title('SVM混淆矩阵', fontproperties=get_chinese_font_properties(14))
        ax1.set_xlabel('预测标签', fontproperties=get_chinese_font_properties(12))
        ax1.set_ylabel('真实标签', fontproperties=get_chinese_font_properties(12))

        # BERT混淆矩阵
        bert_cm = confusion_matrix(true_labels_numeric, bert_predictions_numeric)
        sns.heatmap(bert_cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
                    xticklabels=['负面', '中性', '正面'],
                    yticklabels=['负面', '中性', '正面'])
        ax2.set_title('BERT混淆矩阵', fontproperties=get_chinese_font_properties(14))
        ax2.set_xlabel('预测标签', fontproperties=get_chinese_font_properties(12))
        ax2.set_ylabel('真实标签', fontproperties=get_chinese_font_properties(12))

        plt.tight_layout()
        plt.savefig('./output/混淆矩阵对比.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("✅ 混淆矩阵已保存到: ./output/混淆矩阵对比.png")

    def save_results(self, comparison_df, svm_metrics, bert_metrics, true_labels, svm_predictions, bert_predictions):
        """保存评估结果"""
        output_dir = "./output"
        os.makedirs(output_dir, exist_ok=True)

        # 保存对比表格
        comparison_df.to_csv(f'{output_dir}/模型性能对比.csv', index=False, encoding='utf-8-sig')

        # 保存预测结果
        results_df = pd.DataFrame({
            'true_label': true_labels,
            'svm_prediction': svm_predictions,
            'bert_prediction': bert_predictions
        })
        results_df.to_csv(f'{output_dir}/预测结果详情.csv', index=False, encoding='utf-8-sig')

        # 保存详细结果
        detailed_results = {
            'svm_metrics': svm_metrics,
            'bert_metrics': bert_metrics,
            'comparison': comparison_df.to_dict('records'),
            'summary': {
                'best_model': 'BERT' if float(bert_metrics['准确率'][:-1]) > float(svm_metrics['准确率'][:-1]) else 'SVM',
                'best_accuracy': max(float(bert_metrics['准确率'][:-1]), float(svm_metrics['准确率'][:-1])),
                'test_samples': len(true_labels)
            }
        }

        with open(f'{output_dir}/模型评估详细结果.json', 'w', encoding='utf-8') as f:
            import json
            json.dump(detailed_results, f, ensure_ascii=False, indent=2)

        print(f"\n🎉 评估结果已保存到 {output_dir}/ 目录")
        print(f"📊 模型性能对比: {output_dir}/模型性能对比.csv")
        print(f"📋 预测结果详情: {output_dir}/预测结果详情.csv")
        print(f"📄 详细评估结果: {output_dir}/模型评估详细结果.json")
        print(f"🖼️  混淆矩阵: {output_dir}/混淆矩阵对比.png")

        # 打印总结
        best_model = detailed_results['summary']['best_model']
        best_accuracy = detailed_results['summary']['best_accuracy']
        print(f"\n🏆 最佳模型: {best_model} (准确率: {best_accuracy:.1f}%)")


def main():
    """主评估函数"""
    print("=" * 60)
    print("模型评估系统启动")
    print("=" * 60)

    evaluator = ModelEvaluator()

    try:
        results = evaluator.evaluate_all_models()
        if results is not None:
            print("\n🎉 模型评估完成！")
        else:
            print("\n❌ 模型评估失败")
    except Exception as e:
        print(f"\n❌ 评估过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()