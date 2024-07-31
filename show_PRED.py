#%%
import pickle
with open('dataset_model_dict_with_PRED', 'rb') as file:
    dataset_model_dict_with_PRED = pickle.load(file)
#%%
import torch
from Device import device
import random
import pickle
with open('utilized_POS2COHP_dataset','rb') as file:
    dataset = pickle.load(file)
# dataset = torch.load('utilized_POS2COHP_dataset')
# random.shuffle(dataset)
model = torch.load('models/POS2COHP_Net_FcN_RG_Homo.pth')
error_list = []
for idx,i in enumerate(dataset[:int(len(dataset)*0.9)]):
    print(idx)
    error = i.MN_icohp.to(device)-model(i.to(device))
    error_list.append(error)
    # print(i.MN_icohp)
    # print(model(i.to(device)))
    # print(error)
    # break
#%%
import torch
dataset = torch.load('processed/POS2E_edge_FcN_RG_Homo.pt')
error_list = []
for idx,i in enumerate(dataset[int(len(dataset)*0.9):]):
    print(idx)
    print(i.slab+'_'+i.metal_pair)
    # print(i.edge_index.T)
    # print(i.edge_attr_real.T)
    # print(i.edge_attr_pred.T)
    # 创建布尔掩码，去掉两个张量中值为 -5.0 的元素
    mask = i.edge_attr_real != -5.0
    error = i.edge_attr_real-i.edge_attr_pred
    print(error[mask].shape)
    error_list.append(error[mask])
    # break
#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm, kstest, shapiro
# 将多维tensor展平成一维数组
errors = np.concatenate([tensor.flatten().cpu().detach().numpy() for tensor in error_list])

# 可视化误差的直方图和正态分布拟合曲线
sns.histplot(errors, kde=False, stat="density", bins=90, color='blue', label='Histogram')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, np.mean(errors), np.std(errors))
plt.plot(x, p, 'k', linewidth=2, label='Normal fit')
plt.legend(loc='best')
plt.title('Histogram and Normal Distribution Fit')
plt.xlabel('Error')
plt.ylabel('Density')
plt.show()

# 正态性检验
# Kolmogorov-Smirnov test
ks_statistic, ks_p_value = kstest(errors, 'norm', args=(np.mean(errors), np.std(errors)))
print(f"KS test statistic: {ks_statistic}, p-value: {ks_p_value}")

# Shapiro-Wilk test
shapiro_statistic, shapiro_p_value = shapiro(errors)
print(f"Shapiro-Wilk test statistic: {shapiro_statistic}, p-value: {shapiro_p_value}")

# 结果解释
if ks_p_value > 0.05 and shapiro_p_value > 0.05:
    print("误差符合正态分布")
else:
    print("误差不符合正态分布")
print(np.mean(errors))
print(np.std(errors))
# 计算MAE
mae = np.mean(np.abs(errors))
print(f"Mean Absolute Error (MAE): {mae}")
#%%
import torch
from Device import device
import random
import pickle
with open('utilized_POS2COHP_dataset','rb') as file:
    dataset = pickle.load(file)
dataset = dataset.shuffle()
for i in dataset[:10]:
    print(i.MN_edge_index.shape)
    print(i.MN_edge_index.T)
    print(i.edge_index.T)
    tensor1 = i.MN_edge_index.T
    tensor2 = i.edge_index.T
    # 找到两个张量的交集
    intersection = np.array([item for item in tensor1 if item.tolist() in tensor2.tolist()])

    # 找到第一个张量中不属于第二个张量的元素
    unique_tensor1 = np.array([item for item in tensor1 if item.tolist() not in tensor2.tolist()])

    # 找到第二个张量中不属于第一个张量的元素
    unique_tensor2 = np.array([item for item in tensor2 if item.tolist() not in tensor1.tolist()])

    # 打印结果
    print("交集:\n", intersection)
    print("属于第一个但不属于第二个的元素:\n", unique_tensor1)
    print("属于第二个但不属于第一个的元素:\n", unique_tensor2)
    break
#%%
import torch
dataset = torch.load('processed/POS2COHP_FcN_RG_Homo.pt')
for idx,i in enumerate(dataset):
    # print(idx)
    print(i.slab+'_'+i.metal_pair)
    if idx>=2:
        break
#%%
import os
import re
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

# Directory containing the files
directory = './models/'

# Patterns to match the filenames
patterns = {
    # "CONT2EwithoutCOHP": re.compile(r"^CONT2EwithoutCOHP.*_(\d{8}_\d{6})\.txt$"),
    "POS2E_edge_Net_FcN_RG": re.compile(r"^POS2E_edge_Net_FcN_RG.*_(\d{8}_\d{6})\.txt$"),
    # "POS2EwithoutCOHP": re.compile(r"^POS2EwithoutCOHP.*_(\d{8}_\d{6})\.txt$"),
    # "POS2COHP2E": re.compile(r"^POS2COHP2E.*_(\d{8}_\d{6})\.txt$"),
    "POS2E_edge_Net_train": re.compile(r"^POS2E_edge_Net_train.*_(\d{8}_\d{6})\.txt$"),
}

# Pattern to match the lines containing the metrics
line_pattern = re.compile(r"Va MAE:(\d+\.\d+)\(eV\)")
line_pattern_te = re.compile(r"Te MAE:(\d+\.\d+)\(eV\)")

# Function to find the minimum Test MAE in a file
def find_min_test_mae(filepath):
    print(filepath)
    min_valid_mae = float('inf')
    with open(filepath, 'r') as file:
        for idx,line in enumerate(file):
            if idx<2:
                continue
            match = line_pattern.search(line)
            match_te = line_pattern_te.search(line)
            if match and match_te:
                valid_mae = float(match.group(1))
                test_mae = float(match_te.group(1))
                if valid_mae < min_valid_mae:
                    min_valid_mae = valid_mae
                    cor_test_mae = test_mae
    return cor_test_mae

# Function to process files based on the pattern
def process_files(directory, pattern):
    results = []
    for filename in os.listdir(directory):
        match = pattern.search(filename)
        if match:
            timestamp_str = match.group(1)
            timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            filepath = os.path.join(directory, filename)
            min_test_mae = find_min_test_mae(filepath)
            results.append((timestamp, min_test_mae))
    results.sort()
    return results

# Process files for each pattern
min_mae_results = {}
for key, pattern in patterns.items():
    min_mae_results[key] = process_files(directory, pattern)

# Plot the results
categories = list(min_mae_results.keys())
num_files = max(len(results) for results in min_mae_results.values())

# Initialize the data for the plot
data = {category: [] for category in categories}

# Fill the data, ensuring each category has the same length by padding with None
for category, results in min_mae_results.items():
    for timestamp, min_test_mae in results:
        data[category].append(min_test_mae)
    while len(data[category]) < num_files:
        data[category].append(None)

# Create the bar plot
x = np.arange(num_files)  # the label locations
width = 0.2  # the width of the bars

fig, ax = plt.subplots(figsize=(10, 6))

for i, category in enumerate(categories):
    maes = data[category]
    bar_positions = x + i * width
    bars = ax.bar(bar_positions, [mae for mae in maes if mae is not None], width, label=category)
    # Annotate bars with their values
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel('File Index')
ax.set_ylabel('Min Test MAE')
ax.set_title('Comparison of Minimum Test MAE by File Type and Time Order')
ax.set_xticks(x + width)
ax.set_xticklabels([str(i) for i in range(num_files)])
ax.legend()

fig.tight_layout()

plt.show()
#%%
# I thought if C num equals, then E of two structures may be close so our method is not necessary, but actually they differ.
import pickle
Metals = ["Sc", "Ti", "V" , "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", 
           "Y" , "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
           "Ce", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au",
           "Al", "Ga", "Ge", "In", "Sn", "Sb", "Tl", "Pb", "Bi"]
with open('raw/raw_energy_data_dict_all.pkl', 'rb') as f:
    E_dict = pickle.load(f)
for M1 in Metals:
    for M2 in Metals:
        print('##############################################')
        for k,v in E_dict.items():
            if k.split('_')[2]==M1 and k.split('_')[3]==M2 and len(k.split('_')[1])==4:
                print(k,v)