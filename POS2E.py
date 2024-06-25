import sys
sys.path.append("./scripts/")
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from nets import *
from functions import *
from dataset import *
from training_utils import *
import itertools

from torch_geometric.loader import DataLoader
from POS2COHP import POS2COHP
from torch.utils.data import Subset
import time
from scipy.spatial import KDTree
from mendeleev import element
from Device import device
# device = "cuda:0" if torch.cuda.is_available() else "cpu"
import random
class POS2E():
    def __init__(self, COHP_info_dict, split_ratio=0.8, batch_size=48, dim=256, epochs=300, verbose=True,
                 linear_dim_list = [], conv_dim_list = []):

        self.COHP_info_dict = COHP_info_dict
        self.split_ratio = split_ratio
        self.batch_size = batch_size
        self.dim = dim
        self.linear_dim_list = linear_dim_list
        self.conv_dim_list = conv_dim_list
        self.epochs =  epochs
        self.verbose = verbose

    def build_raw_data_dict(self):

        if not os.path.exists(os.path.join("raw", "raw_energy_data_dict_all.pkl")):
            self.raw_data_dict = build_raw_DSAC_file("./", "structures_all")
            print("Raw data dict prepare done!")
        else:
            print("Raw data dict already exist!")
            file_get = open(os.path.join("raw","raw_energy_data_dict_all.pkl"),'rb') 
            self.raw_data_dict = pickle.load(file_get) 
            file_get.close()
        return None

    def train_single_model(self, suffix, dataset, setting, PRED,pool_type,
                           aug,maximum_num_atoms):
        print(setting)
        pos2e_dataset = POS2E_Dataset("./", dataset, PRED, self.raw_data_dict, setting, suffix,
                                      Data_Augmentation=aug, maximum_num_atoms=maximum_num_atoms)#.shuffle()
        #pos2e_dataset = POS2E_Dataset_Augmentation(pos2e_dataset, suffix)

        pos2e_tr_dataset = pos2e_dataset[:int(len(pos2e_dataset)*self.split_ratio)].shuffle()
        pos2e_te_dataset = pos2e_dataset[int(len(pos2e_dataset)*self.split_ratio):].shuffle()

        pos2e_tr_loader = DataLoader(pos2e_tr_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)  
        pos2e_te_loader = DataLoader(pos2e_te_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True) 

        print("Net-by-Net whole dataset:%s, Trainig set:%s, Test set:%s."%(len(pos2e_dataset), len(pos2e_tr_dataset), len(pos2e_te_dataset)))

        node_feats = 41 if setting["Fake_Carbon"] else 40

        model = CONT2E_Net(dim=self.dim, linear_dim_list=self.linear_dim_list,
                           conv_dim_list=self.conv_dim_list,
                            adj_conv=False, in_features=node_feats, bias=False,
                               conv=pyg_GCNLayer, dropout=0., pool=GraphMultisetTransformer,
                               pool_ratio=0.25, pool_heads=4, pool_seq=["GMPool_G"], pool_layer_norm=False,
                               pool_type=pool_type)
        print(model)
        epochs = self.epochs

        optimizer = torch.optim.AdamW(model.parameters(), lr=10 ** -3, weight_decay=10 ** -4) #-3,-4
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
        loss_func = MSELoss()
        print("Parameters of Model: %s"%sum(list(map(lambda v:v.view(-1,1).shape[0], list(model.state_dict().values())))))

        best = 1e5
        with open("./models/POS2E_Net_%s.txt"%suffix, 'w') as file:
            file.close()

        for i in range(epochs):
            #tr_temp_dataset = pos2e_tr_dataset.shuffle()[:int(len(pos2e_tr_dataset) * (1/79))]
            #te_temp_dataset = pos2e_te_dataset.shuffle()[:int(len(pos2e_te_dataset) * (1/79))]
            #pos2e_tr_loader = DataLoader(tr_temp_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)  
            #pos2e_te_loader = DataLoader(te_temp_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True) 

            tr_loss, tr_res = cont2e_train(model, pos2e_tr_loader, loss_func, optimizer)
            te_loss, te_res = cont2e_evaluate(model, pos2e_te_loader, loss_func)
                    
            scheduler.step()
            learning_rate = round(optimizer.state_dict()['param_groups'][0]['lr'],9)
            training_info = "epoch:%s, Training MAE:%.3f(eV), Training Loss:%.6f, Test MAE:%.3f(eV), Test Loss:%.6f, Learning Rate:%s"%(i, tr_res, tr_loss, te_res, te_loss, learning_rate)
            
            if self.verbose:
                print(training_info)
            
            with open("./models/POS2E_Net_%s.txt"%suffix, 'a') as file:
                file.write(training_info + "\n")   

            if learning_rate <= 1e-5 and te_res < best:
                torch.save(model, "./models/POS2E_Net_%s.pth"%suffix)
                best = te_res
                print("Saved Model!") if self.verbose else None
        return pos2e_dataset, model
    
    def train_single_model_with_active_learning(self, suffix, dataset, setting, PRED, verbose,
                                                base_lr,base_weight_decay, lr_iteration_decay_ratio,weight_decay_decay_ratio,
                                                Metals,selection_ratio,analogue_n):
        print('############################################################')
        print('train_single_model_with_active_learning!')
        print(setting)
        '''数据集'''
        pos2e_dataset = POS2E_Dataset("./", dataset, PRED, self.raw_data_dict, setting, suffix)#.shuffle()
        #pos2e_dataset = POS2E_Dataset_Augmentation(pos2e_dataset, suffix)

        initial_train_size = int(len(pos2e_dataset) * 0.5)
        pos2e_train_indices = list(range(initial_train_size))
        pos2e_pool_indices = list(range(initial_train_size, int(len(pos2e_dataset) * 0.9)))#pool 是慢慢用的
        pos2e_te_indices = list(range(int(len(pos2e_dataset) * 0.9), len(pos2e_dataset)))#这百分之十是测试集，训练的时候完全不动

        print("Net-by-Net whole dataset:%s."%(len(pos2e_dataset)))

        '''超参数'''
        node_feats = 41 if setting["Fake_Carbon"] else 40
        base_lr = base_lr
        base_weight_decay = base_weight_decay
        epochs = self.epochs
        loss_func = MSELoss()
        selection_ratio = selection_ratio
        lr_iteration_decay_ratio = lr_iteration_decay_ratio
        weight_decay_decay_ratio = weight_decay_decay_ratio
        best = 1000

        '''4个iteration'''
        for iteration in range(4): 
            pos2e_tr_dataset = Subset(pos2e_dataset, pos2e_train_indices)
            pos2e_pool_dataset = Subset(pos2e_dataset, pos2e_pool_indices)
            pos2e_te_dataset = Subset(pos2e_dataset, pos2e_te_indices)
            print('##########################################################################################################')
            print("Iteration:%s, Trainig set:%s, Pool set:%s, Test set:%s."%(iteration, len(pos2e_tr_dataset),
                                                                             len(pos2e_pool_dataset), len(pos2e_te_dataset)))

            pos2e_tr_loader = DataLoader(pos2e_tr_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
            pos2e_pool_loader = DataLoader(pos2e_pool_dataset, batch_size=self.batch_size, shuffle=False)
            pos2e_te_loader = DataLoader(pos2e_te_dataset, batch_size=self.batch_size, shuffle=False)

            if iteration == 0:
                """model = CONT2E_Net(in_features=node_feats, linear_dims=[64,128,256], conv_dims=[256,512,512,256],
                                   conv=pyg_GCNLayer, dropout=0., bias=True, 
                                   pool=GraphMultisetTransformer, pool_dropout=0., pool_ratio=0.25, pool_heads=4, 
                                   pool_seq=["GMPool_G"], pool_layer_norm=False)"""
                # model = CONT2E_Net_whole(dim=256, N_linear=1, N_conv=3, adj_conv=False, in_features=node_feats, bias=False,
                #                   conv=pyg_GCNLayer, dropout=0., pool=GraphMultisetTransformer,
                #                   pool_ratio=0.25, pool_heads=4, pool_seq=["GMPool_G"], pool_layer_norm=False)
                model = CONT2E_Net(dim=self.dim, linear_dim_list=self.linear_dim_list,
                           conv_dim_list=self.conv_dim_list,
                            adj_conv=False, in_features=node_feats, bias=False,
                               conv=pyg_GCNLayer, dropout=0., pool=GraphMultisetTransformer,
                               pool_ratio=0.25, pool_heads=4, pool_seq=["GMPool_G"], pool_layer_norm=False,pool_type='all')
                print(model)
                lr = base_lr
                weight_decay = base_weight_decay
                
            lr = base_lr * (lr_iteration_decay_ratio ** iteration)
            weight_decay = base_weight_decay * (weight_decay_decay_ratio ** iteration)
            
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)

            

            print('Iteration:%s '%(iteration))

            '''开始训练'''
            for epoch in range(epochs):
                if verbose:
                    start_time = time.time()  # 记录当前时间作为epoch开始时间
                tr_loss, tr_res = cont2e_train(model, pos2e_tr_loader, loss_func, optimizer)
                te_loss, te_res = cont2e_evaluate(model, pos2e_te_loader, loss_func)
                # vl_loss, vl_res = cont2e_evaluate(model, pos2e_pool_loader, loss_func)
                
                learning_rate = round(optimizer.state_dict()['param_groups'][0]['lr'],9)
                training_info = f'it:{iteration},ep:{epoch}, Tr MAE:{tr_res:.3f}(eV), Tr Loss:{tr_loss:.3f}, Te MAE:{te_res:.3f}(eV), Te Loss:{te_loss:.3f}, LR:{learning_rate}'
                print(training_info)
                scheduler.step()
                with open("./models/POS2E_Net_%s.txt"%(suffix), 'a') as file:
                    file.write(training_info + "\n")   

                if learning_rate <= 3e-4 and te_res < best:
                    torch.save(model, "./models/POS2E_Net_%s.pth"%(suffix))
                    best = te_res
                    print("Saved Model!") if self.verbose else None
                if verbose:
                    end_time = time.time()  # 记录当前时间作为epoch结束时间
                    epoch_time = end_time - start_time  # 计算epoch运行时间
                    print(f"Epoch {epoch} took {epoch_time} seconds")
            
            '''计算训练误差'''
            tr_losses = cont2e_cal(model,pos2e_tr_loader) # 
            print('tr_loss_shape',tr_losses.shape)
            
            '''选择被认为不好的，只选择训练集中的以避免任何对测试集使用'''
            # tr_top_indices = np.argsort(tr_losses)[-int(len(pos2e_dataset) * selection_ratio):] # 训练集中表现不好的，总数的0.03，否则每轮加的数量不一样
            print('random')
            tr_top_indices = random.sample(list(range(len(tr_losses))), int(len(pos2e_dataset) * 0.03))
            te_losses = cont2e_cal(model,pos2e_te_loader) # 
            print('te_loss_shape',te_losses.shape)
            # te_top_indices = np.argsort(te_losses)[-int(len(te_losses) * selection_ratio):] # 测试集中表现不好的
            ################################################################################################################################################
            '''获取数据嵌入向量及结构名称'''
            tr_embeddings, tr_names = zip(*[(
                model(data.to(device), return_embedding=True).to("cpu").detach().numpy(), 
                ['%s_%s'%(data.to("cpu").slab[idx],data.to("cpu").metal_pair[idx]) for idx in range(len(data.to("cpu").metal_pair))]
                                            )
                                                          for data in pos2e_tr_loader])
            pool_embeddings, pool_names = zip(*[(
                model(data.to(device), return_embedding=True).to("cpu").detach().numpy(), 
                ['%s_%s'%(data.to("cpu").slab[idx],data.to("cpu").metal_pair[idx]) for idx in range(len(data.to("cpu").metal_pair))]
                                            )
                                                          for data in pos2e_pool_loader])
            te_embeddings, te_names = zip(*[(
                model(data.to(device), return_embedding=True).to("cpu").detach().numpy(), 
                ['%s_%s'%(data.to("cpu").slab[idx],data.to("cpu").metal_pair[idx]) for idx in range(len(data.to("cpu").metal_pair))]
                                            )
                                                          for data in pos2e_te_loader])
            # print('pool_names',len(pool_names))
            tr_embeddings_list = np.vstack(tr_embeddings)
            pool_embeddings_list = np.vstack(pool_embeddings)
            te_embeddings_list = np.vstack(te_embeddings)
            # pool_names 是111个列表之元组
            tr_names_list = []
            for i in tr_names:
                tr_names_list += i
            pool_names_list = []
            for i in pool_names:
                pool_names_list += i
            te_names_list = []
            for i in te_names:
                te_names_list += i
            # print('pool_name_list',pool_names_list)
            '''建立KDTree'''
            tr_kd_tree = KDTree(tr_embeddings_list)
            pool_kd_tree = KDTree(pool_embeddings_list)
            te_kd_tree = KDTree(te_embeddings_list)
            selected_indices = []
            selected_indices_tr = []
            selected_indices_te = []

            # selected_indices的取值是[6635,13000]，而analogue_idx和pool_names_list和tr_top_indices的取值是[0,5308]
            selected_pool_structures_train = []
            print('训练集较差',len(tr_top_indices),'个')

            '''在pool中寻找与训练集中表现不好的的类似物'''
            for i in tr_top_indices:
                dd,analogue_idx = pool_kd_tree.query(tr_embeddings_list[i],analogue_n,workers=2)
                if verbose:
                    print(f'训练集中预测较差的{tr_names_list[i]}在pool找到的类似物是{[pool_names_list[j] for j in analogue_idx]}')
                selected_indices_tr += [pos2e_pool_indices[j] for j in analogue_idx] # 这里改一下参照物
                selected_pool_structures_train += [pool_names_list[j] for j in analogue_idx] # 存找到的类似物
            # for i in te_top_indices:
            #     dd,analogue_idx = pool_kd_tree.query(te_embeddings_list[i],3,workers=2)
            #     if verbose:
            #         print(f'测试集中预测较差的{te_names_list[i]}在pool找到的类似物是{[pool_names_list[j] for j in analogue_idx]}')
            #     selected_indices_te.append(pos2e_pool_indices[analogue_idx])
            # 不用test信息
            selected_indices = selected_indices_tr#+selected_indices_te
            
            print('不同结构找到的类似物重复数量: ',len(selected_indices)-len(set(selected_indices)))

            '''这一步是试图寻找训练集中表现不好的结构的共性，但这种表示方法还是不够好，或许应该使用符号回归找找描述符'''
            selected_num_dict = {}
            for i in selected_pool_structures_train:
                selected_num_dict[i] = selected_pool_structures_train.count(i)
            selected_num_dict_order=sorted(selected_num_dict.items(),key=lambda x:x[1],reverse=True)
            print('类似物被找到的次数',selected_num_dict_order)





            ################################################################################################################################################
            '''画图, 这部分其实应该用符号回归找描述符代替，因为本来就是在试图用结构的某个特征分类'''
            # 预测的不好的数据集
            '''预测的不好的的吸附能密度分布曲线'''
            unaccurate_structures_dataset = Subset(pos2e_dataset, selected_indices)
            unaccurate_energy_list = []
            unaccurate_elements_list = []
            unaccurate_elements_pair_list = []
            for i in range(len(unaccurate_structures_dataset)):
                unaccurate_energy_list.append(unaccurate_structures_dataset.__getitem__(i).y)
                unaccurate_elements_list.append(unaccurate_structures_dataset.__getitem__(i).metal_pair.split('_')[0])
                unaccurate_elements_list.append(unaccurate_structures_dataset.__getitem__(i).metal_pair.split('_')[1])
                unaccurate_elements_pair_list.append(unaccurate_structures_dataset.__getitem__(i).metal_pair)
            plot(unaccurate_energy_list,'adsorb_energy','numberritio','unacc_E')
            result = count_element(unaccurate_elements_list,Metals=Metals)
            a1 = sorted(result.items(),key = lambda x:x[1],reverse = True)
            print('unacc',a1)
            
            # pool中的全部数据
            '''全部的吸附能密度分布曲线'''
            all_elements_list = []
            all_energy_list = []
            all_elements_pair_list = []
            for i in range(len(pos2e_pool_dataset)):
                all_energy_list.append(pos2e_pool_dataset.__getitem__(i).y)
                all_elements_list.append(pos2e_pool_dataset.__getitem__(i).metal_pair.split('_')[0])
                all_elements_list.append(pos2e_pool_dataset.__getitem__(i).metal_pair.split('_')[1])
                all_elements_pair_list.append(pos2e_pool_dataset.__getitem__(i).metal_pair)
            plot(all_energy_list,'adsorb_energy','numberritio','pool_E')
            result_2 = count_element(all_elements_list,Metals=Metals)
            a2 = sorted(result_2.items(),key = lambda x:x[1],reverse = True)
            print('all',a2)

            '''预测的不好的数据中一个元素出现的频率与pool中的频率之差/pool'''
            diff_between_all_and_unaccurate_elements = {}
            for k in result_2:
                diff_between_all_and_unaccurate_elements[k] = (result[k] - result_2[k])/result_2[k]#预测的不好的数据中一个元素出现的频率与pool中的频率之差/pool
            sort_diff_elem = sorted(diff_between_all_and_unaccurate_elements.items(),key = lambda x:x[1],reverse = True)
            print('(unacc-all)/all',sort_diff_elem)
            plot_rect(sort_diff_elem,Metals=Metals)

            '''全部的与预测的不好的的两金属原子半径之和密度分布曲线'''
            all_radius_sum = []
            unaccurate_radius_sum = []
            atom_radius_dict = {}
            for i in Metals:
                atom_radius_dict[i] = element(i).atomic_radius_rahm #这个包太慢先存下来
            for i in all_elements_pair_list:
                # print(i)
                all_radius_sum.append(atom_radius_dict[i.split('_')[0]]+atom_radius_dict[i.split('_')[1]])
                # print(element(i.split('_')[0]).atomic_radius,element(i.split('_')[1]).atomic_radius,element(i.split('_')[0]).atomic_radius+element(i.split('_')[1]).atomic_radius)
            for i in unaccurate_elements_pair_list:
                unaccurate_radius_sum.append(atom_radius_dict[i.split('_')[0]]+atom_radius_dict[i.split('_')[1]])
            plot(all_radius_sum, 'sum_of_r','numberritio','all_radius_sum')
            plot(unaccurate_radius_sum,'sum_of_r','number','unaccurate_radius_sum')

            

            '''我在考虑到底是应该把新数据点加到原有的里还是只训新的相当于微调'''
            # 不能微调，允许同一个结构多次出现以强调
            pos2e_train_indices.extend(selected_indices)
            print('add %s structures to training dataset'%(len(selected_indices)))
            # pos2e_train_indices.extend(list(set(selected_indices)))
            # pos2e_train_indices = list(set(selected_indices))

            '''不在使用后从pool中删除，因为如果它会被重复选中就说明网络需要它蕴含的信息'''
            # pool中结构可重复使用
            # for idx in list(set(selected_indices)):
            #     # print(idx)
            #     pos2e_pool_indices.remove(idx)
        return pos2e_dataset, model

    def train_all_models(self):
        for suffix, [dataset, model, setting, PRED] in self.COHP_info_dict.items():
            pos2e_dataset, model = self.train_single_model(suffix, dataset, setting, PRED)
        print("Finish the training of all POS2E models.")

    def get_all_models(self):
        dataset_model_dict = {}
        for suffix, [dataset, model, setting, PRED] in self.COHP_info_dict.items():
            dataset = POS2E_Dataset("./", dataset, PRED, self.raw_data_dict, setting, suffix).shuffle()
            model = torch.load("./models/POS2E_Net_%s.pth"%suffix).to("cpu")
            dataset_model_dict[suffix] = [dataset, model, setting]
        return dataset_model_dict

    def build_bridge_for_KDT(self):
        # 返回 POS2E_Dataset, 刚训练的模型，setting
        dataset_model_dict = self.get_all_models()

        for suffix, [dataset, model, setting] in dataset_model_dict.items():
            data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

            model.eval()
            with torch.no_grad():
                PRED = list(map(lambda data:(list(np.abs(np.subtract(model(data).numpy(), data.y.numpy()))), data.slab, data.metal_pair), data_loader))
            DIFF, slabs, metal_pairs = np.hstack(PRED)
            name_list = [x+"_"+y for x,y in zip(slabs, metal_pairs)]
            PRED_DICT = dict(zip(name_list, DIFF))

            dataset_model_dict[suffix].append(PRED_DICT)

        dataset_model_dict_with_PRED_DICT = dataset_model_dict
        return dataset_model_dict_with_PRED_DICT

    def _measure_model(self, sorted_DIFF, top_k):
        MAE = np.average([x[1] for x in sorted_DIFF[:round(len(sorted_DIFF)*top_k)]])
        names_with_MAE = [x[0] for x in sorted_DIFF[:round(len(sorted_DIFF)*top_k)]]
        return MAE, names_with_MAE

    def get_all_models_sorted(self, top_k=1):
        # 返回不同结构的MAE
        dataset_model_dict_with_PRED_DICT = self.build_bridge_for_KDT()

        model_list = []
        for suffix, [dataset, model, setting, PRED_DICT] in dataset_model_dict_with_PRED_DICT.items():
            sorted_DIFF = sorted([(k,np.float32(v)) for k,v in PRED_DICT.items()], key=lambda x:x[1], reverse=True)
            top_k_MAE, bad_predictions = self._measure_model(sorted_DIFF, top_k)
            print(suffix, ": %.6f"%top_k_MAE)
            model_list.append((dataset, model, setting, PRED_DICT, top_k_MAE, bad_predictions, suffix))

        return sorted(model_list, key=lambda x:x[4], reverse=True)


if __name__ == "__main__":
    Metals = ["Sc", "Ti", "V" , "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", 
              "Y" , "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
              "Ce", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au",
              "Al", "Ga", "Ge", "In", "Sn", "Sb", "Tl", "Pb", "Bi"]
    Slabs = ["C","N"]
    Element_List = Metals + Slabs

    setting_dict = {"Fake_Carbon": None, "Binary_COHP": None, "Hetero_Graph": None, "threshold": -0.6}

    pos2cohp = POS2COHP(Element_List, setting_dict,
                        split_ratio=0.9, batch_size=48, hidden_feats=[256,256,256,256], 
                        predictor_hidden_feats=128, epochs=300, verbose=True)
    pos2cohp.build_raw_data_dict()
    #pos2cohp.train_all_models()

    dataset_model_dict_with_PRED = pos2cohp.build_bridge_for_E()
    print(dataset_model_dict_with_PRED.keys())

    pos2e = POS2E(dataset_model_dict_with_PRED, split_ratio=0.9, batch_size=48, dim=256, epochs=300, verbose=True)
    pos2e.build_raw_data_dict() 
    pos2e.train_all_models()

    sorted_model_with_infos = pos2e.get_all_models_sorted(top_k=0.05)

    bad_predictions = sorted_model_with_infos[0][5]

    print(bad_predictions)