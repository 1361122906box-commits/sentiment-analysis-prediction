# batch_processor.py    批量处理脚本
from visualization_data_exporter import export_all_events_data
import json
import os


def check_data_files():
    """检查数据文件是否存在"""
    events = {
        "经济政策解读": "./清洗后的数据源/2025经济政策解读.json",
        "中美贸易谈判": "./清洗后的数据源/2025中美贸易谈判.json",
        "A股半日成交1.25万亿缩量711亿": "./清洗后的数据源/A股半日成交1.25万亿缩量711亿.json",
        "福建舰入列": "./清洗后的数据源/福建舰入列.json",
        "公安机关查处网络谣言": "./清洗后的数据源/公安机关查处网络谣言.json",
        "泡泡玛特直播事故": "./清洗后的数据源/泡泡玛特直播事故.json",
        "日本71岁女子杀害102岁母亲": "./清洗后的数据源/日本71岁女子杀害102岁母亲.json"
    }

    print("检查数据文件...")
    missing_files = []
    for event_name, file_path in events.items():
        if os.path.exists(file_path):
            print(f"✅ {event_name}: {file_path}")
        else:
            print(f"❌ {event_name}: 文件不存在 - {file_path}")
            missing_files.append(file_path)

    return len(missing_files) == 0


def main():
    """主批量处理函数"""
    print("=" * 60)
    print("网络舆情分析系统 - 批量处理")
    print("=" * 60)

    # 检查数据文件
    if not check_data_files():
        print("\n⚠️  请确保所有数据文件都存在后再继续")
        return

    print("\n开始批量处理所有事件...")

    try:
        # 导出所有事件数据
        all_events_data = export_all_events_data()

        print("\n🎉 批量处理完成！")
        print("生成的文件:")
        print("📊 ./output/ALL_EVENTS_可视化数据.json - 所有事件的聚合数据")
        print("📈 ./output/[事件名称]_可视化数据.json - 单个事件数据")
        print("📊 ./output/[事件名称]_情感分析结果.json - 详细分析结果")
        print("📈 ./output/[事件名称]_情感趋势图.png - 情感趋势图表")

    except Exception as e:
        print(f"❌ 批量处理失败: {e}")


if __name__ == "__main__":
    main()