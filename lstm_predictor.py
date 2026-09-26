# lstm_predictor.py
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore')


class SentimentLSTM(nn.Module):
    """LSTM情感趋势预测模型"""

    def __init__(self, input_size=1, hidden_size=32, output_size=1, num_layers=2):
        super(SentimentLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)

        # LSTM前向传播
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])  # 取最后一个时间步
        out = self.fc(out)
        return out


class LSTMPredictor:
    """LSTM预测器封装类"""

    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(-1, 1))  # 情感得分范围-1到1
        self.is_trained = False

    def prepare_training_data(self, time_series_data, seq_length=12):
        """准备LSTM训练数据"""
        if len(time_series_data) < seq_length + 1:
            return None, None

        # 转换为numpy数组
        data = np.array(time_series_data).reshape(-1, 1)

        # 数据标准化
        data_normalized = self.scaler.fit_transform(data).flatten()

        # 创建序列样本
        sequences = []
        targets = []

        for i in range(len(data_normalized) - seq_length):
            seq = data_normalized[i:i + seq_length]
            target = data_normalized[i + seq_length]
            sequences.append(seq)
            targets.append(target)

        if len(sequences) == 0:
            return None, None

        return np.array(sequences), np.array(targets)

    def train(self, time_series_data, epochs=200, seq_length=12):
        """训练LSTM模型"""
        print("开始训练LSTM预测模型...")

        # 准备数据
        X, y = self.prepare_training_data(time_series_data, seq_length)
        if X is None:
            print("数据量不足，无法训练LSTM模型")
            return False

        print(f"训练数据形状: X{X.shape}, y{y.shape}")

        # 转换为PyTorch张量
        X_tensor = torch.FloatTensor(X).unsqueeze(-1)  # (batch, seq_len, 1)
        y_tensor = torch.FloatTensor(y).unsqueeze(-1)  # (batch, 1)

        # 初始化模型
        self.model = SentimentLSTM(input_size=1, hidden_size=32, output_size=1, num_layers=2)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)

        # 训练循环
        self.model.train()
        losses = []

        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)  # 梯度裁剪
            optimizer.step()

            losses.append(loss.item())

            if epoch % 50 == 0:
                print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')

        self.is_trained = True
        print(f"✅ LSTM模型训练完成，最终损失: {losses[-1]:.6f}")
        return True

    def predict(self, time_series_data, future_steps=6, seq_length=12):
        """使用LSTM进行预测"""
        if not self.is_trained or self.model is None:
            print("模型未训练，使用简单预测")
            return self._simple_predict(time_series_data, future_steps)

        self.model.eval()

        # 使用最近的数据
        if len(time_series_data) < seq_length:
            print("数据不足，使用简单预测")
            return self._simple_predict(time_series_data, future_steps)

        recent_data = time_series_data[-seq_length:]

        try:
            # 标准化
            recent_normalized = self.scaler.transform(
                np.array(recent_data).reshape(-1, 1)
            ).flatten()

            predictions = []
            current_sequence = recent_normalized.copy()

            with torch.no_grad():
                for i in range(future_steps):
                    seq_tensor = torch.FloatTensor(current_sequence).unsqueeze(0).unsqueeze(-1)
                    pred = self.model(seq_tensor)
                    pred_value = pred.item()
                    predictions.append(pred_value)

                    # 更新序列：移除第一个元素，添加新预测
                    current_sequence = np.append(current_sequence[1:], pred_value)

            # 反标准化
            predictions = self.scaler.inverse_transform(
                np.array(predictions).reshape(-1, 1)
            ).flatten()

            # 确保预测值在合理范围内
            predictions = np.clip(predictions, -1.0, 1.0)

            print(f"✅ LSTM预测完成: {predictions.tolist()}")
            return predictions.tolist()

        except Exception as e:
            print(f"LSTM预测失败: {e}")
            return self._simple_predict(time_series_data, future_steps)

    def _simple_predict(self, time_series_data, future_steps=6):
        """简单的趋势预测（备选方案）"""
        if len(time_series_data) == 0:
            return [0] * future_steps

        # 使用最近3个点的平均值，并逐渐向中性(0)回归
        recent_avg = np.mean(time_series_data[-3:]) if len(time_series_data) >= 3 else time_series_data[-1]

        predictions = []
        for i in range(1, future_steps + 1):
            # 逐渐回归到中性，但保持原有趋势方向
            if recent_avg > 0:
                pred = max(0, recent_avg * (0.85 ** i))
            else:
                pred = min(0, recent_avg * (0.85 ** i))
            predictions.append(pred)

        print(f"使用简单预测: {predictions}")
        return predictions


# 全局预测器实例
_global_predictor = None


def get_predictor():
    """获取全局预测器实例"""
    global _global_predictor
    if _global_predictor is None:
        _global_predictor = LSTMPredictor()
    return _global_predictor