import json
import random

# 读取原始JSON文件
with open('test.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 随机抽取30条数据
random_subset = random.sample(data, 200)

# 将这30条数据写入一个新的JSON文件
with open('test_small.json', 'w', encoding='utf-8') as file:
    json.dump(random_subset, file, ensure_ascii=False, indent=4)

print("成功随机抽取并保存了30条数据。")
