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