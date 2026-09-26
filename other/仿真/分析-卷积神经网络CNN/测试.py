# 加载数据
name_dict = {
    "经济政策": "2025经济政策解读",
    "中美贸易": "2025中美贸易谈判",
    "A股成交": "A股半日成交1.25万亿缩量711亿",
    "福建舰入列": "福建舰入列",
    "网络谣言": "公安机关查处网络谣言",
    "泡泡玛特": "泡泡玛特直播事故",
    "日本女子杀害母亲": "日本71岁女子杀害102岁母亲",
}
name = name_dict['泡泡玛特']
json_file = fr"C:\Users\issuser\Documents\trae_projects\情感分析预测\数据源\{name}.json"
# 加载数据
import json
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data
data = load_data(json_file)

# 检查数据是否正确加载
print(f"数据类型：{type(data)}")
print(f"数据条数：{len(data)}")
print(f"前5条数据：{data[:5]}")