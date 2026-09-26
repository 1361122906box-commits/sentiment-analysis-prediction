import json
from datetime import datetime, timedelta
import re
import os


def parse_time(time_str, content_str=None):
    now = datetime.now()
    today = now.date()
    current_year = today.year

    # 匹配"X分钟前"的格式
    if "分钟前" in time_str:
        minutes = int(re.match(r'(\d+)分钟前', time_str).group(1))
        return (now - timedelta(minutes=minutes)).strftime('%Y-%m-%d %H:%M:%S')
    # 匹配"今天XX:XX"的格式
    elif "今天" in time_str and re.match(r'今天(\d{1,2}:\d{1,2})', time_str):
        time_part = re.match(r'今天((?:[01]\d|2[0-3]):[0-5]\d)', time_str).group(1)
        return datetime.combine(today, datetime.strptime(time_part, '%H:%M').time()).strftime('%Y-%m-%d %H:%M:%S')
    # 匹配"今天"的格式
    elif time_str == "今天":
        return datetime.combine(today, datetime.min.time()).strftime('%Y-%m-%d %H:%M:%S')
    # 匹配"MM月DD日"的格式
    elif re.match(r'(\d{1,2})月(\d{1,2})日', time_str):
        month, day = map(int, re.match(r'(\d{1,2})月(\d{1,2})日', time_str).groups())
        # 从content_str中提取时间
        if content_str and re.search(r'(\d{1,2}:\d{1,2})', content_str):
            time_content = re.search(r'(\d{1,2}:\d{1,2})', content_str).group(1)
            hour, minute = map(int, time_content.split(':'))
            return datetime(current_year, month, day, hour, minute).strftime('%Y-%m-%d %H:%M:%S')
        else:
            return datetime(current_year, month, day).strftime('%Y-%m-%d %H:%M:%S')
    # 匹配"YYYY年MM月DD日"的格式
    elif re.match(r'(\d{4})年(\d{1,2})月(\d{1,2})日', time_str):
        year, month, day = map(int, re.match(r'(\d{4})年(\d{1,2})月(\d{1,2})日', time_str).groups())
        # 从content_str中提取时间
        if content_str and re.search(r'(\d{1,2}:\d{1,2})', content_str):
            time_content = re.search(r'(\d{1,2}:\d{1,2})', content_str).group(1)
            hour, minute = map(int, time_content.split(':'))
            return datetime(year, month, day, hour, minute).strftime('%Y-%m-%d %H:%M:%S')
        else:
            return datetime(year, month, day).strftime('%Y-%m-%d %H:%M:%S')
    else:
        return None  # 返回 None 表示时间格式不正确


def process_json(file_path, save_file):
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"文件 {file_path} 不存在")
        return

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"解析JSON文件时出错：{e}")
        return

    # 初始化一个字典来统计时间格式的出现次数
    time_format_counts = {}

    if isinstance(data, list):
        cleaned_data = []
        for item in data:
            if "time" in item and "content" in item:
                original_time = item["time"]
                content = item["content"]

                # 解析时间
                parsed_time = parse_time(original_time, content)

                if parsed_time:
                    # 统计时间格式
                    if original_time in time_format_counts:
                        time_format_counts[original_time] += 1
                    else:
                        time_format_counts[original_time] = 1

                    # 更新时间字段
                    item["time"] = parsed_time
                    cleaned_data.append(item)
        data = cleaned_data
    elif isinstance(data, dict):
        if "time" in data and "content" in data:
            original_time = data["time"]
            content = data["content"]

            # 解析时间
            parsed_time = parse_time(original_time, content)

            if parsed_time:
                # 统计时间格式
                if original_time in time_format_counts:
                    time_format_counts[original_time] += 1
                else:
                    time_format_counts[original_time] = 1

                # 更新时间字段
                data["time"] = parsed_time
        else:
            print("字典中缺少 'time' 或 'content' 字段")
            return
    else:
        print("JSON数据格式不支持处理")
        return

    # 输出统计结果
    print("原始时间格式及其出现次数：")
    for fmt, count in time_format_counts.items():
        print(f"{fmt}: {count}次")

    # 写回清洗后的数据
    try:
        with open(save_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"已清洗并保存到 {save_file}")
    except Exception as e:
        print(f"写入JSON文件时出错：{e}")


# 示例用法
if __name__ == "__main__":
    import os

    # 获取当前脚本所在目录作为项目根目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.join(current_dir, "数据源")
    cleaned_dir = os.path.join(current_dir, "清洗后的数据源")

    # 确保输出目录存在
    os.makedirs(cleaned_dir, exist_ok=True)

    name_dict = {
        "经济政策": "2025经济政策解读",
        "中美贸易": "2025中美贸易谈判",
        "A股成交": "A股半日成交1.25万亿缩量711亿",
        "福建舰入列": "福建舰入列",
        "网络谣言": "公安机关查处网络谣言",
        "泡泡玛特": "泡泡玛特直播事故",
        "日本女子杀害母亲": "日本71岁女子杀害102岁母亲",
    }

    for i in name_dict.keys():
        name = name_dict[i]
        json_file = os.path.join(source_dir, f"{name}.json")
        save_file = os.path.join(cleaned_dir, f"{name}.json")

        print(f"处理: {name}")
        print(f"源文件: {json_file}")
        print(f"目标文件: {save_file}")

        process_json(json_file, save_file)
        print("-" * 50)