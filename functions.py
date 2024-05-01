import numpy as np
import pandas as pd
import pickle
import os
import re
import torch
import copy
from pymatgen.io.vasp import Poscar
from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Structure, Lattice


def build_raw_DSAC_file(path, folder):
    temp_raw_data_dict = {}
    
    file_get = open(os.path.join("metals_bulk", "energy_per_atom_dict"), "rb")
    metal_energy_dict = pickle.load(file_get)
    file_get.close()

    structure_path = os.path.join(path, folder)
    for file in os.listdir(structure_path):
        ele_combo = file.split("_")[-2:]
        outcar_path = os.path.join(structure_path, file, "OUTCAR")
        if os.path.exists(outcar_path):
            outcar_content = os.popen('grep -a "reached" %s | tail -n 1'%(outcar_path)).read()
            if "reached required accuracy - stopping structural energy minimisation" in outcar_content:
                line = os.popen('grep -a "without" %s | tail -n 1'%(outcar_path)).read()
                absolute_energy = float(line.split('=')[2].split()[0])
                    
                contcar_path = os.path.join(structure_path, file, "CONTCAR")
                structure = Poscar.from_file(contcar_path).structure
                distance_matrix = structure.distance_matrix

                temp_el = list(filter(lambda x:x != None, (list(map(lambda x:x.specie.name if x.specie.name in ["C","N"] else None, structure.sites)))))
                fake_slab_energy = temp_el.count("C") * (-9.2230488) + temp_el.count("N") * (-8.316268445)
                
                stability_energy = absolute_energy - metal_energy_dict[ele_combo[0]] - metal_energy_dict[ele_combo[1]] - fake_slab_energy
                    
                temp_raw_data_dict[file] = stability_energy
            else:
                stability_energy = "nan"
        else:
            stability_energy = "nan"

    file_save = open(os.path.join(path, "raw", "raw_energy_data_dict_all.pkl"),'wb') 
    pickle.dump(temp_raw_data_dict, file_save) 
    file_save.close()
    return temp_raw_data_dict


def get_hole_atoms(structure,distance_matrix):
    #distance_matrix = structure.distance_matrix
    CN_list = []
    for idx, site in enumerate(structure):
        if site.specie.name in ["C","N"]:
            threshold = 1.5
            CN = list(filter(lambda x:x!=None,
                                 (list(map(lambda x:(structure[x].specie.name) if distance_matrix[idx][x] < threshold and 
                                                                                   structure[x].specie.name in ["C","N"] and 
                                                                                   x != idx else None, 
                                           range(len(structure)))))))
            CN_list.append((idx, CN))
    hole_atoms_idx = list(map(lambda x:x[0], list(filter(lambda x:len(x[1])<3, CN_list))))
    #assert len(hole_atoms_idx) == 7
    return hole_atoms_idx


def get_coordinated_environment(structure, distance_matrix):
    verbosed_distance_matrix = []
    coordinated_idx_matrix = []
    N_idx = []
    
    hole_atoms_idx = get_hole_atoms(structure,distance_matrix)

    for idx, site in enumerate(structure.sites):
        ele = site.specie.name
        if ele in ["N","C"]:
            threshold = 1.5
        else: 
            threshold = 4

        distance_list = distance_matrix[idx]
        distance_list_with_idx_ele = list(map(lambda x:(distance_list[x],x,structure[x].specie.name), np.arange(len(distance_list))))
        
        if ele in ["N","C"]:
            cool_distance_list_with_idx_ele = list(filter(lambda x:x[0]<threshold, distance_list_with_idx_ele))
        else:
            #cool_distance_list_with_idx_ele = list(filter(lambda x:0<x[0]<threshold and x[2]=="N", distance_list_with_idx_ele))
            cool_distance_list_with_idx_ele = list(filter(lambda x:x[0]<threshold and x[1] in hole_atoms_idx, distance_list_with_idx_ele))
        
        coordinated_idx_list = list(map(lambda x:x[1], cool_distance_list_with_idx_ele))
        coordinated_idx_matrix.append(coordinated_idx_list)
    
    temp_coordinated_idx_matrix = copy.deepcopy(coordinated_idx_matrix)
    for idx,coor_idx_list in enumerate(temp_coordinated_idx_matrix):
        for jdx in coor_idx_list:
            if idx not in temp_coordinated_idx_matrix[jdx]:
                coordinated_idx_matrix[jdx].append(idx)
    return coordinated_idx_matrix
    

def check_validaiton(coordinated_idx_matrix):
    flag_list = []
    
    for idx, coordinated_idx_list in enumerate(coordinated_idx_matrix):
        if len(coordinated_idx_list)< 2:
            flag_list.append(False)
        for coordinated_idx in coordinated_idx_list:
            flag = True if idx in coordinated_idx_matrix[coordinated_idx] else False
            flag_list.append(flag)
      
    return True if len(set(flag_list))==1 else False



###################################################################################################################
################################################### NET_by_NET ####################################################
###################################################################################################################


def build_raw_COHP_file(path, structure_set):
    icohp_list_dict = {}
    lobster_path = os.path.join(path, structure_set)
    folders = [f for f in os.listdir(lobster_path)]
    #folders = [f for f in os.listdir(lobster_path) if os.path.isdir(os.path.join(lobster_path, f))]

    for folder in folders:
        if os.path.isfile(os.path.join(lobster_path,folder,'ICOHPLIST.lobster')):
            result = rd_icohplist(os.path.join(lobster_path,folder,'ICOHPLIST.lobster'))
            if result != {}:
                icohp_list_dict[folder] = result

    with open(os.path.join("raw", "icohp_%s.pkl")%(structure_set),'wb') as f:
        pickle.dump(icohp_list_dict, f)
    print("%s contains %s data points."%(structure_set, len(icohp_list_dict)))
    return None


def rd_icohplist(f):
    icohp={}
    with open(f,'r') as file:
        lines = file.readlines()
    for i in lines:
        line_list = i.split()
        if "_" not in line_list[1] and "_" not in line_list[2] and "COHP" not in line_list[0]:
            en1,en2 = line_list[1], line_list[2] 
            key = en1 + '_' + en2 if int(en1[-2:]) < int(en2[-2:]) else en2 + '_' + en1 
            if key in icohp:
                icohp[key] = float(icohp[key]) + float(line_list[7]) # add two spin ICOHP
            else:
                icohp[key]=line_list[7]
    return icohp


def get_connectivity(structure):
    distance_matrix = structure.distance_matrix
    edge_index = []
    threshold = {'N':1.5, 'C':1.5, 'M':2.4}
    for idx,site in enumerate(structure):
        if site.specie.name in ['N', 'C']:
            indices = np.where(distance_matrix[idx]<threshold[site.specie.name])
            edge_index+=((idx,j) for j in indices[0])
        else:
            indices = np.where(distance_matrix[idx]<threshold['M'])
            edge_index+=((idx,j) for j in indices[0])
    for i in edge_index:
        if (i[1],i[0]) not in edge_index:
            edge_index += ((i[1],i[0]),)  
        
    np.conn = np.array(edge_index)
    res = pd.DataFrame(list(np.conn[:,0])).value_counts()
    #assert set(list(res.values)) == {4,5}  #make sure the connectivity is valid
    return edge_index


def get_MCN_edge_index_and_COHP(elements, cohp_res, connectivity):
    M_index = [idx for idx,i in enumerate(elements) if i!='N' and i!='C']
    temp = [x for x in connectivity if set(M_index).difference(set(x)) != set(M_index)]
    CN_index = sorted(set([x for x in [i for j in temp for i in j] if x not in M_index]))

    MCN_edge_index = []
    for i in CN_index:
        for j in M_index:
            MCN_edge_index.append([i,j])
            MCN_edge_index.append([j,i])
    
    MCN_icohp = []
    if cohp_res != None:
        for idx1, idx2 in MCN_edge_index:
            key = "%s%s_%s%s"%(elements[idx1],idx1+1,elements[idx2],idx2+1) if idx1 < idx2 else "%s%s_%s%s"%(elements[idx2],idx2+1,elements[idx1],idx1+1) 
            try:
                MCN_icohp.append(float(cohp_res[key]))
            except:
                print(cohp_res)
                print(elements)
                assert 1 ==2
    return torch.Tensor(MCN_edge_index).long().T, MCN_icohp


def filter_pairs(tensor, specific_values):
    filtered_pairs = []
    for i in range(tensor.size(1)):  # Iterate through columns (pairs)
        if not (tensor[0, i].item() in specific_values or tensor[1, i].item() in specific_values):
            filtered_pairs.append((tensor[0, i].item(), tensor[1, i].item()))
    return torch.tensor(filtered_pairs).T



def split_batch_data(batch, batch_pred):
    #batch = batch.to("cpu")
    #batch_pred = batch.to("cpu")
    lengths = list(batch.cohp_num.numpy())
    slices = np.cumsum(lengths)[:-1]
    pred_slice = np.split(np.array(batch_pred), slices)
    return pred_slice


def extract_COHP(graph, icohp_dicts):
    metal_pair = graph.metal_pair
    slab = graph.slab
    icohp_dict = icohp_dicts[slab][metal_pair]
    valid_cohp = list(map(lambda x:tuple(map(int, re.findall(r'\d+', x))) if icohp_dict[x] <= -0.4 else None, icohp_dict.keys()))
    valid_cohp = list(filter(lambda x:x!=None,valid_cohp))
    inter_valid_cohp = sum(map(lambda t: [(t[0]-1, t[1]-1), (t[1]-1, t[0]-1)], valid_cohp), [])
    return torch.Tensor(inter_valid_cohp).to(torch.int64).transpose(0, 1)



def name2structure(path, name):
    qv, C_idx, ele1, ele2 = name.split("_")
    ori = Poscar.from_file(os.path.join("sample_space", "%s.vasp"%qv)).structure
    ALL_N_idx = list(filter(lambda x:x!=None, list(map(lambda x:x if ori[x].specie.name == "N" else None, range(len(ori))))))
    C_idx = [int(c) for c in C_idx]
    changed_C_idx = np.array(ALL_N_idx)[C_idx]
    none = list(map(lambda idx: ori.replace(idx,"C"), changed_C_idx))
    ori.replace(-2, ele1)
    ori.replace(-1, ele2)
    temp_path = os.path.join(path, name)
    os.mkdir(temp_path) if not os.path.exists(temp_path) else None
    writer = Poscar(ori)
    writer.write_file(os.path.join(temp_path, "POSCAR"))
    return None 



def build_raw_COHP_file_old(path, structure_sets):
    for structure_set in structure_sets:
        icohp_list_dict = {}
        lobster_path = os.path.join(path, structure_set, "structures")
        folders = [f for f in os.listdir(lobster_path)]
        #folders = [f for f in os.listdir(lobster_path) if os.path.isdir(os.path.join(lobster_path, f))]

        for folder in folders:
            if os.path.isfile(os.path.join(lobster_path,folder,'ICOHPLIST.lobster')):
                result = rd_icohplist(os.path.join(lobster_path,folder,'ICOHPLIST.lobster'))
                if result != {}:
                    icohp_list_dict[folder] = result

        with open(os.path.join("raw", "icohp_%s.pkl")%(structure_set),'wb') as f:
            pickle.dump(icohp_list_dict, f)
        print("%s contains %s data points."%(structure_set, len(icohp_list_dict)))
    return None


def build_raw_DSAC_file_old(path, structure_set, Metals, metal_energy_dict, symmetry_dict):
    temp_raw_data_dict = {}

    #slab_path = os.path.join(path, structure_set, "slab", "OUTCAR")
    #slab_energy = float(os.popen('grep -a "without" %s | tail -n 1'%(slab_path)).read().split('=')[2].split()[0])
        
    energy_matrix = pd.DataFrame("nan", index=Metals, columns=Metals)

    sub_path = os.path.join(path, structure_set, "structures")
    for file in os.listdir(sub_path):
        ele_combo = file.split("_") if structure_set != "samples" else file.split("_")[-2:]
        outcar_path = os.path.join(sub_path, file, "OUTCAR")
        outcar_content = os.popen('grep -a "reached" %s | tail -n 1'%(outcar_path)).read()
        if "reached required accuracy - stopping structural energy minimisation" in outcar_content:
            line = os.popen('grep -a "without" %s | tail -n 1'%(outcar_path)).read()
            absolute_energy = float(line.split('=')[2].split()[0])
                
            contcar_path = os.path.join(sub_path, file, "CONTCAR")
            structure = Poscar.from_file(contcar_path).structure
            distance_matrix = structure.distance_matrix

            temp_el = list(filter(lambda x:x != None, (list(map(lambda x:x.specie.name if x.specie.name in ["C","N"] else None, structure.sites)))))
            fake_slab_energy = temp_el.count("C") * (-9.2230488) + temp_el.count("N") * (-8.316268445)
            
            stability_energy = absolute_energy - metal_energy_dict[ele_combo[0]] - metal_energy_dict[ele_combo[1]] - fake_slab_energy
                
            temp_raw_data_dict[file] = (structure, distance_matrix, stability_energy)
        else:
            stability_energy = "nan"
        
        if symmetry_dict[structure_set]:
            energy_matrix.loc[ele_combo[0], ele_combo[1]] = stability_energy
            energy_matrix.loc[ele_combo[1], ele_combo[0]] = stability_energy
        else:
            energy_matrix.loc[ele_combo[0], ele_combo[1]] = stability_energy
    energy_matrix.to_csv(os.path.join(path, structure_set, "%s.csv"%structure_set)) if structure_set != "samples" else None

    file_save = open(os.path.join(path, "raw","%s_raw_data_dict.pkl"%structure_set),'wb') 
    pickle.dump(temp_raw_data_dict, file_save) 
    file_save.close()
    return temp_raw_data_dict



def padding_structure(structure, x_unit=2.45, y_unit=4.27, x_layer=1, y_layer=1):  #x_unit=2.45, y_unit=4.27
    '''to extend the graphene in x and y direction'''
    coords = [site.coords for site in structure.sites]
    x_new_points = []
    for coord in coords:
        if coord[0] > np.max(np.array(coords)[:,0]) - x_unit: # 如果它是最右边那2列的
            for xr in range(1,x_layer+1):
                new_point = np.array([coord[0] + x_unit*xr, coord[1], coord[2]])
                x_new_points.append(new_point)
            
    temp_coords = coords + x_new_points # 向右扩展x_layer层

    y_new_points = []
    for coord in temp_coords:
        if coord[1] < y_unit: # 如果它是最下边那4行的
            for yr in range(1, y_layer+1):
                new_point = np.array([coord[0], coord[1] - y_unit*yr, coord[2]])
                y_new_points.append(new_point)
        
    padding = np.array([0, y_unit * y_layer, 0]) # 向下扩展y_layer层
        
    lattice = Lattice(structure.lattice.matrix + np.array([x_unit * x_layer, y_unit * y_layer, 0]) * np.identity(3))
    new_structure = Structure(lattice, [], [])
    for site in structure:
        new_structure.append(site.specie, site.coords+padding, coords_are_cartesian=True)
    for coord in x_new_points+y_new_points:
        new_structure.append(Element("C"), coord+padding, coords_are_cartesian=True)
    
    #filename = filename.split(".")[0] + "_%s%s"%(str(x_layer),str(y_layer)) + filename.split(".")[1]
    #new_structure.to(fmt="poscar", filename=os.path.join(path, filename))
    return new_structure


def enumerate_padding_structure(structure, maximum_num_atoms = 300):
    num_atoms = len(structure)
    x_layer, y_layer = 0, 0
    augmented_structures, check_num_atoms = [], []

    while num_atoms < maximum_num_atoms:
        y_layer += 1
        padded_structure = padding_structure(structure, x_layer=x_layer, y_layer=y_layer)
        num_atoms = len(padded_structure)
        # y_l一直加直到超过限制，然后x_l+1，y_l从零再开始
        while num_atoms < maximum_num_atoms:
            augmented_structures.append(padded_structure)
            
            y_layer += 1
            padded_structure = padding_structure(structure, x_layer=x_layer, y_layer=y_layer)
            num_atoms = len(padded_structure)

            if num_atoms >= maximum_num_atoms:
                x_layer += 1
                y_layer = 0
                padded_structure = padding_structure(structure, x_layer=x_layer, y_layer=y_layer)
                num_atoms = len(padded_structure)

                if num_atoms < maximum_num_atoms: 
                    augmented_structures.append(padded_structure)
                    break
        else:
            y_layer -= 1
            x_layer += 1
            padded_structure = padding_structure(structure, x_layer=x_layer, y_layer=y_layer)
            num_atoms = len(padded_structure)
            if num_atoms < maximum_num_atoms:
                augmented_structures.append(padded_structure)
    # print(len(augmented_structures))
    return augmented_structures


"""def remove_node(node_id, data):
    new_data = copy.deepcopy(data)
    new_data.x = torch.cat([new_data.x[:node_id], new_data.x[node_id + 1:]], dim=0)

    mask = (new_data.edge_index[0] != node_id) & (new_data.edge_index[1] != node_id)
    new_data.edge_index = new_data.edge_index[:, mask]
    for i in range(2):
        new_data.edge_index[i][new_data.edge_index[i] > node_id] -= 1
    new_data.y -= (-9.2230488)
    return new_data"""