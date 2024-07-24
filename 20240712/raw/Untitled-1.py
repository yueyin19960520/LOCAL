#%%
import pickle
import matplotlib.pyplot as plt
import numpy as np
# print('#############################################')
# print('icohp_structures_all.pkl')
# with open('icohp_structures_all.pkl','rb') as file:
#     h = pickle.load(file)
#     print(type(h),len(h))
#     for i in h:
#         print('key %s'%(i))
#         print('value %s'%(h[i]))
#         break
print('#############################################')

def plot(l):
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
    x = list(data_range)
    my_x_ticks = np.linspace(min_, max_, 10)
    plt.xticks(my_x_ticks)
    plt.xlabel('adsorb energy')
    plt.ylabel('number')
    plt.scatter(x,y)
# plot([1,2,2,2,2,2,22,2,22,3,4,5])    

energy_list = []
print('raw_energy_data_dict_all.pkl')
with open('raw_energy_data_dict_all.pkl','rb') as file:
    h = pickle.load(file)
    print(type(h),len(h))
    for i in h:
        energy_list.append(h[i])
for i in h:
    print(i,h['QV1_012345_Au_Mo'])
    break
plot(energy_list)
# plot_dict({'Sc': 88, 'Ti': 70, 'V': 31, 'Cr': 9, 'Mn': 37, 'Fe': 30, 'Co': 53, 'Ni': 25, 'Cu': 27, 'Zn': 37, 'Y': 67, 'Zr': 53, 'Nb': 14, 'Mo': 11, 'Tc': 11, 'Ru': 7, 'Rh': 33, 'Pd': 21, 'Ag': 3, 'Cd': 15, 'Ce': 69, 'Hf': 47, 'Ta': 15, 'W': 6, 'Re': 6, 'Os': 6, 'Ir': 31, 'Pt': 24, 'Au': 17, 'Al': 86, 'Ga': 22, 'Ge': 25, 'In': 9, 'Sn': 16, 'Sb': 8, 'Tl': 10, 'Pb': 5, 'Bi': 16})

#%%
l1 = [('Zn', 0.011983151565617516), ('Cd', 0.011982340504912328), ('Sc', 0.010354812023163896), ('In', 0.010006055919932089), ('Ga', 0.009773822204679278), ('Ag', 0.008261464343067865), ('Pd', 0.005239181801960608), ('Ru', 0.005006137026002611), ('Y', 0.0045405881812234025), ('Pt', 0.0034945902250963813), ('Os', 0.003262626863411968), ('Ir', 0.0029133300530433705), ('Sb', 0.00268055563065377), ('Tl', 0.002446429440422185), ('Hf', 0.0019833137777585524), ('Bi', 0.001400972191431954), ('Ce', 0.0011687384761791472), ('Ge', 0.0009367751144947374), ('Re', 0.00012219981291532872), ('Sn', -0.0005750420399798853), ('Mo', -0.0018535440649281136), ('Zr', -0.0018543551256333043), ('Nb', -0.0023193632632757116), ('Au', -0.002436156004823109), ('W', -0.0029008937888971194), ('Ti', -0.003016875469739326), ('Rh', -0.003248568477855339), ('Cr', -0.0032534348420864816), ('Mn', -0.003946891745024144), ('Ni', -0.004993971115424753), ('Al', -0.005691753675456764), ('Co', -0.006273554554646565), ('Fe', -0.007202489415657799), ('Ta', -0.00720303012279459), ('Cu', -0.00778374958771081), ('Tc', -0.00917931470777483), ('Pb', -0.00918012576848002), ('V', -0.014643971385778321)]
def plot_rect(l,cm):
    x = [i[0] for i in l]
    y = [i[1] for i in l]
    # 构建x与颜色的映射关系
    color_map = {}
    for i, category in enumerate(x):
        color_map[category] = plt.cm.viridis(i / len(x))

    # 创建柱形图
    colors = [color_map[category] for category in x]
    plt.bar(x, y, color=colors)
    
    
plot_rect(l1)
#%%
def avg_e_E(element):
    element_E = []
    with open('raw_energy_data_dict_all.pkl','rb') as file:
        h = pickle.load(file)
        print(type(h),len(h))
        for k in h:
            if element in k:
                # print(k)
                element_E.append(h[k])
    print(len(element_E))
    print(sum(element_E)/len(element_E))
    return sum(element_E)/len(element_E)
avg_e_E('Tl')
#%%
res = []
for i in ["Sc", "Ti", "V" , "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", 
           "Y" , "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
           "Ce", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au",
           "Al", "Ga", "Ge", "In", "Sn", "Sb", "Tl", "Pb", "Bi"]:
    print(i)
    res.append([i,avg_e_E(i)])
    print()
a1 = sorted(res,key = lambda x:x[1],reverse = True)
print(a1)
#%%
x = ["Sc", "Ti", "V" , "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", 
           "Y" , "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
           "Ce", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au",
           "Al", "Ga", "Ge", "In", "Sn", "Sb", "Tl", "Pb", "Bi"]
def plot_rect(l):
    plt.figure(figsize=(10,5))
    # 构建x与颜色的映射关系
    x = ["Sc", "Ti", "V" , "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", 
           "Y" , "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
           "Ce", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au",
           "Al", "Ga", "Ge", "In", "Sn", "Sb", "Tl", "Pb", "Bi"]
    color_map = {}
    for i, category in enumerate(x):
        color_map[category] = plt.cm.viridis(i / len(x))
    x = [i[0] for i in l]
    y = [i[1] for i in l]
    # 创建柱形图
    colors = [color_map[category] for category in x]
    plt.ylabel('(unacc-all)/all')
    plt.bar(x, y, color=colors)
    plt.show()
l1 = []
for i in range(len(["Sc", "Ti", "V" , "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", 
           "Y" , "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
           "Ce", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au",
           "Al", "Ga", "Ge", "In", "Sn", "Sb", "Tl", "Pb", "Bi"])):
    l1.append((x[i],i))
plot_rect(l1)
#%%
import pickle
with open('icohp_structures_all.pkl','rb') as file:
    h = pickle.load(file)
    print(type(h),len(h))
    for i in h:
        print('QV1_012345_Au_Mo',h['QV1_012345_Au_Mo'])
        break
    
#%%
with open('raw_energy_data_dict_all.pkl','rb') as file:
    e = pickle.load(file)
with open('icohp_structures_all.pkl','rb') as file:
    cohp = pickle.load(file)
l1 = e.keys()
l2 = cohp.keys()
print(len(set(l1)-set(l2)))
print(len(set(l2)-set(l1)))
#%%
import torch
dataset = torch.load('../processed/POS2EwithoutCOHP.pt')
for i in range(len(dataset)):
    if type(dataset[i].edge_index)==None:
        print(type(dataset[i].edge_index))