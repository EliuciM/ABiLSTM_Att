import pandas as pd
import json
import random
import os
from collections import Counter

# 读取 CSV 文件
df = pd.read_csv("data/qunaer_20250226.csv")

# 确保列名一致
df = df[['reviewid', 'reviewtext', 'sentiment']]

# 统计各个标签的数量
label_counts = Counter(df['sentiment'])

# 为每个类别随机抽取 20% 作为验证集
val_indices = []
for label, count in label_counts.items():
    label_indices = df[df['sentiment'] == label].index.tolist()
    val_sample_size = int(count * 0.2)  # 计算该类别 20% 的数据量
    val_indices.extend(random.sample(label_indices, val_sample_size))

# 生成 val.json 和 train.json
val_data = df.loc[val_indices].to_dict(orient='records')
train_data = df.drop(index=val_indices).to_dict(orient='records')

# 确保输出目录存在
output_dir = "data/qunaer_20250226_balance82"
os.makedirs(output_dir, exist_ok=True)

# 保存 JSON 文件
with open(f"{output_dir}/val.json", "w", encoding="utf-8") as f:
    json.dump(val_data, f, ensure_ascii=False, indent=4)

with open(f"{output_dir}/train.json", "w", encoding="utf-8") as f:
    json.dump(train_data, f, ensure_ascii=False, indent=4)

print(f"数据划分完成！val.json: {len(val_data)} 条, train.json: {len(train_data)} 条")
