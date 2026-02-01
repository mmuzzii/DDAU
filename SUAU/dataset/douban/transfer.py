import pandas as pd

data_file = "C:/Users/23386/PycharmProjects/rec/SUAU/dataset/douban/test.txt"
# 使用 pandas 读取文件
df = pd.read_csv(data_file, sep=' ', names=['user_id', 'item_id', 'rating'])

# 2. 数据转换：按 user_id 分组，收集每个用户的交互 item_id 列表
user_item_interactions = df.groupby('user_id')['item_id'].apply(list)

# 3. 写入新的 train.txt 文件
# 转换为所需的格式，每行一个用户及其交互的物品列表
with open("C:/Users/23386/PycharmProjects/rec/SUAU/dataset/douban/test_transformed.txt", "w") as f:
    for user, items in user_item_interactions.items():
        # 将每个用户及其交互物品写入文件，格式：user_id item_id1 item_id2 ...
        line = f"{user} " + " ".join(map(str, items)) + "\n"
        f.write(line)