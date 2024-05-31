#%%
import torch
from mendeleev import element
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.preprocessing import StandardScaler
from matminer.featurizers.composition import Miedema
from pymatgen.core.composition import Composition
import os

miedema_featurizer = Miedema()
Metals = ["Al", "Sc", "Ti", "V" , "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", 
           "Ga", "Ge","Y" , "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd",
           "Ag", "Cd", "In", "Sn", "Sb", "Ce", "Hf", "Ta", "W",  "Re", 
          "Os", "Ir", "Pt", "Au", "Tl", "Pb", "Bi"]

# 使用字典缓存对象 反复构造对象很慢
element_cache = {}
for symbol in Metals:
    element_cache[symbol] = element(symbol)

comp_cache = {}
for M1 in Metals:
    for M2 in Metals:
        composition = f'{M1}50{M2}50'
        comp = Composition(composition)
        comp_cache[composition] = comp
# 加载数据
import torch
dataset_path = 'POS2E_FcN_BC_Homo.pt'
datas = torch.load(dataset_path)
# for i in datas:
#     print(i.slab+i.metal_pair)

# Dataframe
features = ['M1_mass','M2_mass','M1_r','M2_r',
            'M1_electronegativity_pauling','M2_electronegativity_pauling',
            'coord_num',]
use_qv = ['QV1','QV2','QV3','QV4','QV5','QV6']


features += ['M1_neighbor_C_num','M2_neighbor_C_num',
             'M1_neighbor_N_num','M2_neighbor_N_num',
             'M1_M2_unrelaxed_distance'
             ]
            # 'M1_CN_distance_min','M1_CN_distance_max','M1_CN_distance_mean','M1_CN_distance_std']
features += ['M1_dipole_polarizability', 'M2_dipole_polarizability',
             'M1_vdw_radius','M2_vdw_radius',
             'M1_heat_of_formation','M2_heat_of_formation',
             'M1_ionenergies_1','M2_ionenergies_1',
            #  'M1_eletronegativity_allen','M2_eletronegativity_allen',
             'M1_eletronegativity_allred_rochow','M2_eletronegativity_allred_rochow',
             'M1_eletronegativity_cottrell_sutton','M2_eletronegativity_cottrell_sutton',
             'M1_eletronegativity_ghosh','M2_eletronegativity_ghosh',
             'M1_eletronegativity_gordy','M2_eletronegativity_gordy',
             'M1_eletronegativity_martynov_batsanov','M2_eletronegativity_martynov_batsanov',
             'M1_eletronegativity_mulliken','M2_eletronegativity_mulliken',
             'M1_eletronegativity_nagle','M2_eletronegativity_nagle',
             'M1_eletronegativity_sanderson','M2_eletronegativity_sanderson',

            #  'M1_metallic_radius','M2_metallic_radius',
            #  'M1_metallic_radius_c12','M2_metallic_radius_c12',
            #  'M1_proton_affinity','M2_proton_affinity',
             'M1_vdw_radius','M2_vdw_radius',
             'M1_vdw_radius_alvarez','M2_vdw_radius_alvarez',
            #  'M1_vdw_radius_batsanov','M2_vdw_radius_batsanov',
            #  'M1_vdw_radius_bondi','M2_vdw_radius_bondi',
            #  'M1_vdw_radius_dreiding','M2_vdw_radius_dreiding',
             'M1_vdw_radius_mm3','M2_vdw_radius_mm3',
            #  'M1_vdw_radius_rt','M2_vdw_radius_rt',
            #  'M1_vdw_radius_truhlar','M2_vdw_radius_truhlar',
             'M1_vdw_radius_uff','M2_vdw_radius_uff',
             'Miedema_deltaH_inter', 'Miedema_deltaH_amor', 'Miedema_deltaH_ss_min',



             ]
print(f'feature num:{len(features)}')

# 取一个子集做实验
qvx_dict = {}
qvx_datas = [data for data in datas if data.slab.split('_')[0] in use_qv]
print(len(qvx_datas))
for idx,data in enumerate(qvx_datas):
    name = f'{data.slab}_{data.metal_pair}'
    C_idx = name.split('_')[1]
    C_idx_set = set([int(j) for j in list(C_idx)])
    # print(name)
    qvx_dict[name] = []
    M1 = data.metal_pair.split('_')[0]
    M2 = data.metal_pair.split('_')[1]
    qvx_dict[name].append(element_cache[M1].mass)
    qvx_dict[name].append(element_cache[M2].mass)
    qvx_dict[name].append(element_cache[M1].atomic_radius_rahm)
    qvx_dict[name].append(element_cache[M2].atomic_radius_rahm)
    qvx_dict[name].append(element_cache[M1].electronegativity('pauling'))
    qvx_dict[name].append(element_cache[M2].electronegativity('pauling'))
    
    qv = name.split('_')[0]
    if qv in ['QV1','QV2','QV3']:
        num_N = 6
        M1_idx_set = {0,1,2}
        M2_idx_set = {3,4,5}
        
        M1_neighbor_C_num = len(M1_idx_set.intersection(C_idx_set))
        M2_neighbor_C_num = len(M2_idx_set.intersection(C_idx_set))
        M1_neighbor_N_num = 3 - M1_neighbor_C_num
        M2_neighbor_N_num = 3 - M2_neighbor_C_num
        if qv=='QV1':
            M1_M2_unrelaxed_distance = 2.467
        elif qv=='QV2':
            M1_M2_unrelaxed_distance = 2.467
        elif qv=='QV3':
            M1_M2_unrelaxed_distance = 2.137
    elif qv in ['QV4', 'QV6']:
        num_N = 7
        M1_idx_set = {0,1,2,3}
        M2_idx_set = {3,4,5,6}
        
        M1_neighbor_C_num = len(M1_idx_set.intersection(C_idx_set))
        M2_neighbor_C_num = len(M2_idx_set.intersection(C_idx_set))
        M1_neighbor_N_num = 4 - M1_neighbor_C_num
        M2_neighbor_N_num = 4 - M2_neighbor_C_num
        if qv == 'QV4':
            M1_M2_unrelaxed_distance = 3.263
        elif qv == 'QV6':
            M1_M2_unrelaxed_distance =3.700
    elif qv == 'QV5':
        num_N = 8
        M1_idx_set = {0,1,2,3}
        M2_idx_set = {4,5,6,7}
        
        M1_neighbor_C_num = len(M1_idx_set.intersection(C_idx_set))
        M2_neighbor_C_num = len(M2_idx_set.intersection(C_idx_set))
        M1_neighbor_N_num = 4 - M1_neighbor_C_num
        M2_neighbor_N_num = 4 - M2_neighbor_C_num
        M1_M2_unrelaxed_distance = 4.273

    qvx_dict[name].append(num_N)
    qvx_dict[name].append(M1_neighbor_C_num)
    qvx_dict[name].append(M2_neighbor_C_num)
    qvx_dict[name].append(M1_neighbor_N_num)
    qvx_dict[name].append(M2_neighbor_N_num)
    qvx_dict[name].append(M1_M2_unrelaxed_distance)
    # print(name)
    # print(M1_neighbor_C_num,M2_neighbor_C_num,M1_neighbor_N_num,M2_neighbor_N_num,M1_M2_unrelaxed_distance)

    qvx_dict[name].append(element_cache[M1].dipole_polarizability)
    qvx_dict[name].append(element_cache[M2].dipole_polarizability)
    qvx_dict[name].append(element_cache[M1].vdw_radius)
    qvx_dict[name].append(element_cache[M2].vdw_radius)
    qvx_dict[name].append(element_cache[M1].heat_of_formation)
    qvx_dict[name].append(element_cache[M2].heat_of_formation)
    qvx_dict[name].append(element_cache[M1].ionenergies[1])
    qvx_dict[name].append(element_cache[M2].ionenergies[1])

    # qvx_dict[name].append(element_cache[M1].electronegativity('allen'))
    # qvx_dict[name].append(element_cache[M2].electronegativity('allen'))
    qvx_dict[name].append(element_cache[M1].electronegativity('allred-rochow'))
    qvx_dict[name].append(element_cache[M2].electronegativity('allred-rochow'))
    qvx_dict[name].append(element_cache[M1].electronegativity('cottrell-sutton'))
    qvx_dict[name].append(element_cache[M2].electronegativity('cottrell-sutton'))
    qvx_dict[name].append(element_cache[M1].en_ghosh)
    qvx_dict[name].append(element_cache[M2].en_ghosh)
    qvx_dict[name].append(element_cache[M1].electronegativity('gordy'))
    qvx_dict[name].append(element_cache[M2].electronegativity('gordy'))
    qvx_dict[name].append(element_cache[M1].electronegativity(scale='martynov-batsanov'))
    qvx_dict[name].append(element_cache[M2].electronegativity(scale='martynov-batsanov'))
    qvx_dict[name].append(element_cache[M1].electronegativity('mulliken'))
    qvx_dict[name].append(element_cache[M2].electronegativity('mulliken'))
    qvx_dict[name].append(element_cache[M1].electronegativity('nagle'))
    qvx_dict[name].append(element_cache[M2].electronegativity('nagle'))
    qvx_dict[name].append(element_cache[M1].electronegativity())
    qvx_dict[name].append(element_cache[M2].electronegativity())

    # qvx_dict[name].append(element_cache[M1].metallic_radius)
    # qvx_dict[name].append(element_cache[M2].metallic_radius)
    # qvx_dict[name].append(element_cache[M1].metallic_radius_c12)
    # qvx_dict[name].append(element_cache[M2].metallic_radius_c12)
    # qvx_dict[name].append(element_cache[M1].proton_affinity)
    # qvx_dict[name].append(element_cache[M2].proton_affinity)
    qvx_dict[name].append(element_cache[M1].vdw_radius)
    qvx_dict[name].append(element_cache[M2].vdw_radius)
    qvx_dict[name].append(element_cache[M1].vdw_radius_alvarez)
    qvx_dict[name].append(element_cache[M2].vdw_radius_alvarez)
    # qvx_dict[name].append(element_cache[M1].vdw_radius_batsanov)
    # qvx_dict[name].append(element_cache[M2].vdw_radius_batsanov)
    # qvx_dict[name].append(element_cache[M1].vdw_radius_bondi)
    # qvx_dict[name].append(element_cache[M2].vdw_radius_bondi)
    # qvx_dict[name].append(element_cache[M1].vdw_radius_dreiding)
    # qvx_dict[name].append(element_cache[M2].vdw_radius_dreiding)
    qvx_dict[name].append(element_cache[M1].vdw_radius_mm3)
    qvx_dict[name].append(element_cache[M2].vdw_radius_mm3)
    # qvx_dict[name].append(element_cache[M1].vdw_radius_rt)
    # qvx_dict[name].append(element_cache[M2].vdw_radius_rt)
    # qvx_dict[name].append(element_cache[M1].vdw_radius_truhlar)
    # qvx_dict[name].append(element_cache[M2].vdw_radius_truhlar)
    qvx_dict[name].append(element_cache[M1].vdw_radius_uff)
    qvx_dict[name].append(element_cache[M2].vdw_radius_uff)

    
    Miedema_features = miedema_featurizer.featurize(comp_cache[f'{M1}50{M2}50'])
    qvx_dict[name].append(Miedema_features[0])
    qvx_dict[name].append(Miedema_features[1])
    qvx_dict[name].append(Miedema_features[2])

    qvx_dict[name].append(data.y)
    qvx_dict[name].append(name)
# print(qv6_dict)
columns = features+['y','name']
df = pd.DataFrame(list(qvx_dict.values()), columns=columns)
# print(df.head())
X = df.iloc[:, :-2]
y = df.iloc[:, -2]
# print(X.head())
# X = X.sample(frac=1, axis=1) # sfs依赖于特征顺序
print(X.head())
print(X.shape)
# print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

##########################################标准化#############################################
# 创建 StandardScaler 实例
scaler = StandardScaler()
# 标准化训练数据
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# 将标准化后的数据转换为 DataFrame
X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
#%%
for i,feature in enumerate(features):
    print(i+1,feature,X_train.iloc[0,i])
#%%
################################ 能量密度分布 #######################################
import matplotlib.pyplot as plt
import numpy as np
def plot(l,xl='X',yl='num',title='density plot'):#画一个列表的密度分布
    plt.figure(figsize=(10,5))
    accuracy = 30
    min_=min(l)
    max_=max(l)
    y = []
    data_range = np.linspace(min_, max_, accuracy)
    for i in data_range:
        n = 0
        for j in l:
            if j>=i and j<i+(max_-min_)/accuracy:
                n+=1
        y.append(n)
        # y.append(n/len(l))
    x = list(data_range)
    if xl=='adsorb_energy':
        plt.xlim((-1.56, 16.35))
        my_x_ticks = np.linspace(-1.56, 16.35, 10)
    if xl=='sum_of_r':
        plt.xlim((380, 580))
        my_x_ticks = np.linspace(380, 580, 10)
    plt.xticks(my_x_ticks)
    plt.xlabel(xl)
    plt.ylabel(yl)
    plt.title(title)
    plt.scatter(x,y)
    plt.show()
plot(list(y),'adsorb_energy','n','distribution of adsorb E')

#%%
################################## 符号回归 #####################################
from gplearn.genetic import SymbolicRegressor
from gplearn.functions import make_function
import time
import pickle
start_time = time.time()  # 记录当前时间作为epoch开始时间
print(columns)
# 定义并训练 SymbolicRegressor

# 定义指数函数
def _exp(x):
    # 指数函数的边界处理
    x_clipped = np.clip(x, -2, 2)  # 将输入值限制在一个较小的范围内，避免数值溢出
    # 安全的指数函数计算
    result = np.exp(x_clipped)
    return result

# 创建指数函数
exp = make_function(function=_exp, name='exp', arity=1)

if not os.path.isfile('./Symbolic_regressor_0.69mae_std'):
    print('no exist sr')
    sr = SymbolicRegressor(
    population_size=60000,
    tournament_size=500, # tournament_size越小，选择压力越大，算法收敛的速度可能更快，但也有可能错过一些隐藏的优秀公式。
    generations=3,
    stopping_criteria=0.8,
    p_crossover=0.7, # 一般较高的值有助于保持多样性，但过高可能影响局部优化。
    p_subtree_mutation=0.1, # 可以适当增加以保持多样性，特别是在种群收敛过快的情况下。
    p_hoist_mutation=0.10, # 高升变异操作的概率。通过将子树提升到更高层次来减少表达式的复杂度。
    p_point_mutation=0.05, # 点变异操作的概率。改变表达式中的单个节点，增加多样性。
    max_samples=0.9, # 每代要使用的样本数
    verbose=1,
    parsimony_coefficient=0.0007, # 过拟合，可以增加该值。
    # random_state=42,
    function_set=['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'sin', 'cos','tan',
                  'abs','neg', 'inv',exp],
    const_range=(-5.,5.),
    n_jobs=8,
    warm_start=True
)
    
    sr.fit(X_train, y_train)

    # 打印发现的数学表达式
    print("Discovered mathematical expression:", sr._program)

    # 用发现的表达式进行预测
    y_test_pred_sr = sr.predict(X_test)

    # 计算并打印均方误差
    mae = np.mean(abs(y_test - y_test_pred_sr) )
    print(f'Mean Abs Error: {mae}')

    end_time = time.time()  # 记录当前时间作为epoch结束时间
    epoch_time = end_time - start_time  # 计算epoch运行时间
    print(f"SFS took {epoch_time} seconds")
    # print(y_test)
    # print(y_test_pred_sr)
    with open('./Symbolic_regressor','wb') as file:
        pickle.dump(sr, file)
else:
    print('found exist sr')
    with open('./Symbolic_regressor_0.69mae_std', 'rb') as file:
        sr = pickle.load(file)
        # 打印发现的数学表达式
        print("Discovered mathematical expression:", sr._program)

        # 用发现的表达式进行预测
        y_test_pred_sr = sr.predict(X_test)

        # 计算并打印均方误差
        mae = np.mean(abs(y_test - y_test_pred_sr) )
        print(f'Mean Abs Error: {mae}')
        


