import pandas as pd

# 加载train set文件（假设每行是"用户ID 物品ID 评分"格式，用空格分隔）
train_set = pd.read_csv("./yelp2018/train.txt", sep=" ", header=None, names=["user_id", "item_id", "rating"])

# 定义函数：保证每个用户至少保留一条交互记录
def sample_with_user_retain(data, frac, random_state=42):
    sampled_data = data.groupby("user_id", group_keys=False).apply(
        lambda group: group.sample(frac=frac, random_state=random_state) if len(group) > 1 else group
    )
    return sampled_data

# 生成 50% 和 70% 的训练集
train_set_50 = sample_with_user_retain(train_set, frac=0.5)
train_set_70 = sample_with_user_retain(train_set, frac=0.7)

# 保存文件，确保按原始顺序保存，且无引号
train_set_50.to_csv("./yelp50/train.txt", sep=" ", header=False, index=False, quoting=3)
train_set_70.to_csv("./yelp70/train.txt", sep=" ", header=False, index=False, quoting=3)

