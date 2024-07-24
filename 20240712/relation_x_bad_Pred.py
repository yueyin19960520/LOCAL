#%%
from mendeleev import element
from Device import device
Metals = ["Al", "Sc", "Ti", "V" , "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", 
           "Ga", "Ge","Y" , "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd",
           "Ag", "Cd", "In", "Sn", "Sb", "Ce", "Hf", "Ta", "W",  "Re", 
          "Os", "Ir", "Pt", "Au", "Tl", "Pb", "Bi"]

# 使用字典缓存对象 反复构造对象很慢
element_cache = {}
for symbol in Metals:
    element_cache[symbol] = element(symbol)
#%%
import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 加载数据集
data_path = 'processed/POS2E_FcN_BC_Homo.pt'
data = torch.load(data_path)
data = data[int(len(data)*0.9):]
# data = data.to(device)


# 加载训练好的模型
model_path = 'models/POS2E_Net_FcN_BC_Homo_all.pth'
model = torch.load(model_path)
model = model.to(device)
model.eval()  # 设置模型为评估模式

# 创建一个 DataLoader
batch_size = 512
loader = DataLoader(data, batch_size=batch_size)

# 创建一个列表来保存结果
results = []

# 遍历数据集并获取预测值与真实值
with torch.no_grad():
    for batch in loader:
        batch = batch.to(device)
        # 获取预测值
        outputs = model(batch)
        metal_pair = batch.metal_pair
        slab = batch.slab
        for i in range(len(metal_pair)):
            M1 = metal_pair[i].split('_')[0]
            M2 = metal_pair[i].split('_')[1]
            # features = element_cache[M1].atomic_radius_rahm+element_cache[M2].atomic_radius_rahm
            # features = M1
            # features = element_cache[M1].electronegativity('pauling')+element_cache[M2].electronegativity('pauling')
            true_labels = batch.y[i]
            # features=true_labels.item()
            features = slab[i].split('_')[0]
            pred_value = outputs[i]
            
            
            
            results.append((features, pred_value.item(), true_labels.item()))

# 提取特征和对应的预测误差
errors = [(result[0], abs(result[1] - result[2])) for result in results]
print(errors[:2])

# 作图，假设我们对特征的某一维度进行分析，这里假设为第一个特征维度
x = [result[0] for result in errors]
y = [result[1] for result in errors]

plt.figure(1)
plt.scatter(x, y, s=0.5)
plt.xlabel(f'Feature')
plt.ylabel('Prediction Error')
plt.title('Prediction Error vs Feature')
plt.ylim((0,0.8))
plt.show()

#%%
# 创建一个空的字典用于存放每个类别对应的误差值列表
data = {}

# 遍历类别和误差值，将误差值添加到对应类别的列表中
for category, error in zip(x[:], y[:]):
    if category not in data:
        data[category] = []
    data[category].append(error)

# 绘制箱线图
plt.figure(2)
plt.boxplot(data.values(), labels=data.keys())
plt.xlabel('Category')
plt.ylabel('Error')
plt.title('Box plot of errors by category')
plt.ylim((0,0.8))
plt.show()
print('test MAE:',sum(y)/len(y))