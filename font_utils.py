# font_utils.py
import matplotlib.pyplot as plt
import platform
import os


def setup_chinese_font():
    """设置中文字体"""
    try:
        if platform.system() == 'Windows':
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
        elif platform.system() == 'Darwin':  # macOS
            plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC']
        else:  # Linux
            plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC']

        plt.rcParams['axes.unicode_minus'] = False
        print("✅ 中文字体设置成功")
        return True
    except Exception as e:
        print(f"❌ 中文字体设置失败: {e}")
        return False


def get_chinese_font_properties(size=12):
    """获取中文字体属性"""
    return {'size': size}