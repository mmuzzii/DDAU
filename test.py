import numpy as np
from collections import defaultdict

# 读取数据并解析
def parse_txt(file_path):
    item_user_dict = defaultdict(set)  # 存储每个物品的交互用户集合
    with open(file_path, 'r') as f:
        for line in f:
            user_id, item_id, rate = map(int, line.strip().split())
            item_user_dict[item_id].add(user_id)  # 添加交互用户到集合
    return item_user_dict

# 统计每个物品交互的用户数量
def calculate_statistics(item_user_dict):
    interaction_counts = [len(users) for users in item_user_dict.values()]  # 每个物品的交互用户数
    min_count = np.min(interaction_counts)
    max_count = np.max(interaction_counts)
    mean_count = np.mean(interaction_counts)
    return min_count, max_count, mean_count

# 主程序
if __name__ == "__main__":
    file_path = "./dataset/yelp2018/train.txt" # 替换为您的文件路径
    item_user_dict = parse_txt(file_path)
    min_count, max_count, mean_count = calculate_statistics(item_user_dict)

    print(f"最小交互用户数: {min_count}")
    print(f"最大交互用户数: {max_count}")
    print(f"平均交互用户数: {mean_count:.2f}")


