import numpy as np
import pandas as pd
import pickle
import os
import re
import torch
import copy
import random
import datetime
from pymatgen.io.vasp import Poscar
from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Structure, Lattice
from Device import device
from net_utils import GCNLayer, GCNLayer_edge, GMT
import torch.nn.functional as F


root_path = os.path.dirname(os.path.dirname(__file__))

configs_str_mapping = {"F.relu":F.relu,
                       "GCNLayer":GCNLayer,
                       "GCNLayer_edge":GCNLayer_edge,
                       "GMT":GMT}


def build_raw_COHP_file(path, folder):
    icohp_list_dict = {}
    lobster_path = os.path.join(path, folder)
    folders = [f for f in os.listdir(lobster_path)]
    #folders = [f for f in os.listdir(lobster_path) if os.path.isdir(os.path.join(lobster_path, f))]

    for file in folders:
        if os.path.isfile(os.path.join(lobster_path, file, 'ICOHPLIST.lobster')):
            result = rd_icohplist(os.path.join(lobster_path, file, 'ICOHPLIST.lobster'))
            if result != {}:
                icohp_list_dict[file] = result

    with open(os.path.join(path, "raw", "icohp_structures_all.pkl"),'wb') as f:
        pickle.dump(icohp_list_dict, f)
    print("%s contains %s data points."%(folder, len(icohp_list_dict)))
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


def build_raw_DSAC_file(path, folder):
    temp_raw_data_dict = {}
    
    file_get = open(os.path.join(path, "metals_bulk", "energy_per_atom_dict"), "rb")
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
    return None#temp_raw_data_dict



def build_raw_DSAC_file_looping(path, folder, loop):
    temp_raw_data_dict = {}
    
    file_get = open(os.path.join(path, "metals_bulk", "energy_per_atom_dict"), "rb")
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

    file_save = open(os.path.join(path, "raw", "raw_energy_data_dict_loop%s.pkl"%loop),'wb') 
    pickle.dump(temp_raw_data_dict, file_save) 
    file_save.close()
    return None


def build_raw_COHP_file_looping(path, folder, loop):
    icohp_list_dict = {}
    lobster_path = os.path.join(path, folder)
    folders = [f for f in os.listdir(lobster_path)]
    #folders = [f for f in os.listdir(lobster_path) if os.path.isdir(os.path.join(lobster_path, f))]

    for file in folders:
        if os.path.isfile(os.path.join(lobster_path, file, 'ICOHPLIST.lobster')):
            result = rd_icohplist(os.path.join(lobster_path, file, 'ICOHPLIST.lobster'))
            if result != {}:
                icohp_list_dict[file] = result

    with open(os.path.join(path, "raw", "icohp_structures_loop%s.pkl"%loop),'wb') as f:
        pickle.dump(icohp_list_dict, f)
    print("%s contains %s data points."%(folder, len(icohp_list_dict)))
    return None




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
    batch = batch.to("cpu")
    batch_pred = batch_pred.to("cpu")
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


def name2structure(name,save=False,path="./next_loop"):
    qv, C_idx, ele1, ele2 = name.split("_")
    ori = Poscar.from_file(os.path.join("sample_space", "%s.vasp"%qv)).structure
    ALL_N_idx = list(filter(lambda x:x!=None, list(map(lambda x:x if ori[x].specie.name == "N" else None, range(len(ori))))))
    C_idx = [int(c) for c in C_idx]
    changed_C_idx = np.array(ALL_N_idx)[C_idx]
    none = list(map(lambda idx: ori.replace(idx,"C"), changed_C_idx))
    ori.replace(-2, ele1)
    ori.replace(-1, ele2)
    if not save:
        return ori
    else:
        temp_path = os.path.join(path, name)
        os.mkdir(temp_path) if not os.path.exists(temp_path) else None
        writer = Poscar(ori)
        writer.write_file(os.path.join(temp_path, "POSCAR"))
        return None 


def padding_structure(structure, x_unit=2.45, y_unit=4.27, x_layer=1, y_layer=1):  #x_unit=2.45, y_unit=4.27
    coords = [site.coords for site in structure.sites]
    x_new_points = []
    for coord in coords:
        if coord[0] > np.max(np.array(coords)[:,0]) - x_unit:
            for xr in range(1,x_layer+1):
                new_point = np.array([coord[0] + x_unit*xr, coord[1], coord[2]])
                x_new_points.append(new_point)
            
    temp_coords = coords + x_new_points

    y_new_points = []
    for coord in temp_coords:
        if coord[1] < y_unit:
            for yr in range(1, y_layer+1):
                new_point = np.array([coord[0], coord[1] - y_unit*yr, coord[2]])
                y_new_points.append(new_point)
        
    padding = np.array([0, y_unit * y_layer, 0])
        
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
 
    return augmented_structures


### NEW ADDING BY YUEYIN 0724 ###
def get_physical_encoding_dict():
    Metals = ["Sc", "Ti", "V" , "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", 
           "Y" , "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
           "Ce", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au",
           "Al", "Ga", "Ge", "In", "Sn", "Sb", "Tl", "Pb", "Bi"]
    Slabs = ["C","N"]
    Element_list = Metals + Slabs

    file_path = os.path.join(root_path,"raw", "physical_encodings_dict.pkl")
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            physical_encodings_dict = pickle.load(f)
    else:
        df_radii = pd.read_excel(os.path.join(root_path,"raw","pre_set.xlsx"),sheet_name='Radii_X')
        df_ip = pd.read_excel(os.path.join(root_path,"raw","pre_set.xlsx"),sheet_name='IP')
        IP_dict = df_ip.iloc[:9,:].set_index('ele').to_dict('dict')
        pyykko_dict = df_radii.iloc[:,1:5].set_index('symbol').T.to_dict('dict')
        element_cache = {symbol: Element(symbol) for symbol in Element_list}

        physical_encodings_dict = {}
        for ele in Element_list:
            physical_encodings_dict[ele] = [element_cache[ele].atomic_radius_rahm, 
                                            element_cache[ele].mass,
                                            element_cache[ele].electronegativity('pauling'),
                                            element_cache[ele].electronegativity('allred-rochow'),
                                            element_cache[ele].electronegativity('cottrell-sutton'),
                                            element_cache[ele].electronegativity('gordy'),
                                            element_cache[ele].electronegativity(scale='martynov-batsanov'),
                                            element_cache[ele].electronegativity('mulliken'),
                                            element_cache[ele].electronegativity('nagle'),
                                            element_cache[ele].vdw_radius,
                                            element_cache[ele].vdw_radius_alvarez,
                                            element_cache[ele].vdw_radius_mm3,
                                            element_cache[ele].ionenergies[1],
                                            element_cache[ele].dipole_polarizability,
                                            element_cache[ele].heat_of_formation,
                                            pyykko_dict[ele]['single'],
                                            pyykko_dict[ele]['double'],
                                            pyykko_dict[ele]['triple'],
                                            IP_dict[ele]['IP1'],
                                            IP_dict[ele]['IP2'],
                                            IP_dict[ele]['IP3'],
                                            IP_dict[ele]['IP4']]
        physical_encodings_dict["Fc"] = copy.deepcopy(physical_encodings_dict["C"])

        # Normalization 
        df = pd.DataFrame(physical_encodings_dict)
        z_score_normalize = lambda row:(row - row.mean())/row.std()
        physical_encodings_dict = df.apply(z_score_normalize, axis=1).to_dict(orient='list')

        with open(file_path, "wb") as f:
            pickle.dump(physical_encodings_dict, f)

    return physical_encodings_dict


def split_list(input_list, n, split_ratio):
    first_piece_size = int(len(input_list) * split_ratio)
    remaining_size = len(input_list) - first_piece_size
    piece_size = remaining_size // (n - 1)
    pieces = []
    pieces.append(input_list[:first_piece_size])
    for i in range(n - 1):
        start_index = first_piece_size + i * piece_size
        if i == n - 2:
            pieces.append(input_list[start_index:])
        else:
            pieces.append(input_list[start_index:start_index + piece_size])
    return pieces[0], pieces[1:]


def setting2suffix(setting):
    suffix = "%s_%s_%s"%("FcN" if setting["Fake_Carbon"] else "CN", 
                         "BC" if setting["Binary_COHP"] else "RG",
                         "Hetero" if setting["Hetero_Graph"] else "Homo")
    if setting["encode"] == "physical":
        suffix += "_phys"
    elif setting["encode"] == "onehot":
        suffix += "_oneh"
    else:
        suffix += "_both"
    return suffix


## NEW AFTER 20240820 ##
def restart(criteria=None, new_split=False, split_ratio=[], Random=False, rest_split_ratio=[], loop=0):

    if criteria != None:
        processed_path = os.path.join(root_path,"processed")
        #none = list(map(lambda f:print(f"old file has been deleted: {f}") if criteria in f else None,
                        #os.listdir(processed_path)))
        #none = list(map(lambda f:os.remove(os.path.join(processed_path, f)) if criteria in f else None,
                        #os.listdir(processed_path)))

    if new_split:
        temp_dict = {'QV1': '012345', 'QV2': '012345', 'QV3': '012345', 'QV4': '0123456', 'QV5': '01234567', 'QV6': '0123456'}
        def filter_strings(keys_list,temp_dict=temp_dict):
            return [string for string in keys_list if string.split('_')[1] in [temp_dict.get(string.split('_')[0], ""), ""]]
        """
        if not os.path.exists(os.path.join(root_path, "raw", "icohp_structures_all.pkl")):
            build_raw_COHP_file(root_path, "structures_all")
            print("Raw data dict prepare done!")
        else:
            print("Raw data dict already exist!")

        if not os.path.exists(os.path.join(root_path, "raw", "raw_energy_data_dict_all.pkl")):
            build_raw_DSAC_file(root_path, "structures_all")
            print("Raw data dict prepare done!")
        else:
            print("Raw data dict already exist!")
        """

        #if loop > 0:
        for _loop in range(loop+1):
            if not os.path.exists(os.path.join(root_path, "raw", "icohp_structures_loop%s.pkl"%_loop)):
                build_raw_COHP_file_looping(root_path, "structures_loop/loop%s"%_loop, loop=_loop)
                print("looping Raw data dict prepare done!")
            else:
                print("looping Raw data dict already exist!")

            if not os.path.exists(os.path.join(root_path, "raw", "raw_energy_data_dict_loop%s.pkl"%_loop)):
                build_raw_DSAC_file_looping(root_path, "structures_loop/loop%s"%_loop, loop=_loop)
                print("looping Raw data dict prepare done!")
            else:
                print("looping Raw data dict already exist!")

        raw_file = os.path.join(root_path,"raw","icohp_structures_loop0.pkl")
        with open(raw_file, "rb") as pklf:
            icohp_list_dict = pickle.load(pklf)
        icohp_list_keys = list(icohp_list_dict.keys())

        folder_name = os.path.join(root_path,"raw")
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y%m%d%H%M%S")
        filename = os.path.join(folder_name, f"list_{timestamp}.pkl")
 
        if Random:
            random.shuffle(icohp_list_keys)
            print(icohp_list_keys[:5])
            """
            all_CN_keys = filter_strings(icohp_list_keys)
            mixed_keys = list(set(icohp_list_keys).difference(all_CN_keys))
            icohp_list_keys = mixed_keys
            """
            data_num = len(icohp_list_keys)
            tr_list_keys = icohp_list_keys[:int(split_ratio[0]*data_num)]
            vl_list_keys = icohp_list_keys[int(split_ratio[0]*data_num):int(sum(split_ratio[:-1])*data_num)]
            te_list_keys = icohp_list_keys[int(sum(split_ratio[:-1])*data_num):]
        else:
            all_CN_keys = filter_strings(icohp_list_keys)
            mixed_keys = list(set(icohp_list_keys).difference(all_CN_keys))

            analogue_keys1 = ["_".join((qv, "", e1, e2)) for key in mixed_keys for qv, idx, e1, e2 in [key.split("_")]]
            analogue_keys2 = ["_".join((qv, temp_dict[qv], e1, e2)) for key in mixed_keys for qv, idx, e1, e2 in [key.split("_")]]
            analogue_keys = analogue_keys1 + analogue_keys2

            tr_list_keys = list(set(icohp_list_keys).difference(set(analogue_keys)))
            vlte_keys = list(set(icohp_list_keys).difference(set(tr_list_keys)))

            data_num = len(vlte_keys)
            tr_list_keys = vlte_keys[:int(split_ratio[0]*data_num)] + tr_list_keys
            random.shuffle(tr_list_keys)
            vl_list_keys = vlte_keys[int(split_ratio[0]*data_num):int(sum(split_ratio[:-1])*data_num)]
            te_list_keys = vlte_keys[int(sum(split_ratio[:-1])*data_num):]


        splitted_keys = {"train": tr_list_keys,
                         "valid": vl_list_keys,
                         "test": te_list_keys}

        with open(filename, 'wb') as file:
            pickle.dump(splitted_keys, file)
        print(f"New split dict is saved in: {filename}")
        return splitted_keys
    else:
        return None


def set_seed(seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed) 

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        print("No random seed provided. Running with non-deterministic behavior.")


def worker_init_fn(worker_id):
    np.random.seed(torch.initial_seed() % (2**32))


def icohp2df(data):
    df = pd.DataFrame([key.split('_') + [value] for key, value in data.items()], columns=['Row', 'Column', 'Value'])
    return df.pivot(index='Row', columns='Column', values='Value')


def expand_material_ids(ids):
    expanded_ids = set()
    for mid in ids:
        parts = mid.split('_')
        qv, cb, m1, m2 = parts[0], parts[1], parts[2], parts[3]
        if qv == 'QV4':
            expanded_ids.add(mid)
        else:
            expanded_ids.add(f"{qv}_{cb}_{m1}_{m2}")
            expanded_ids.add(f"{qv}_{cb}_{m2}_{m1}")
    return expanded_ids


def equal_id(mid):
    parts = mid.split('_')
    qv, cb, m1, m2 = parts[0], parts[1], parts[2], parts[3]
    if qv == 'QV4':
        return mid
    else:
        return f"{qv}_{cb}_{m2}_{m1}" 


def get_selected_names():
    selected_names = []

    mom_path = os.path.join(root_path, "raw")
    files = [f for f in os.listdir(mom_path) if "selected_names_" in f]
    for file in files:
        with open(os.path.join(mom_path,file), "rb") as f:
            temp = pickle.load(f)
            selected_names += temp

    return selected_names



#### New Adding in 20240902 ####
"""
def evaluate_model(pos2cohp_model_path, 
                   pos2e_model_path, 
                   root_path, 
                   configs,
                   device='cpu'):

    setting = dict(zip(list(configs["setting_dict"].keys()), list(map(lambda x:x[0], list(configs["setting_dict"].values())))))
    Element_List = configs["Metals"] + configs["Slabs"]

    # Determine node features based on configuration settings
    if setting["encode"] == "physical":
        node_feats = 22
    elif setting["encode"] == "onehot":
        node_feats = 41 if setting["Fake_Carbon"]else 40
    else:
        node_feats = 63 if setting["Fake_Carbon"] else 62
    
    # Load the pos2cohp model with configurations
    config = configs["POS2COHP"]
    
    pos2cohp_model = POS2COHP_Net(in_feats=node_feats, 
                                  hidden_feats=config["hidden_feats"],
                                  activation=configs_str_mapping[config["activation"]],
                                  predictor_hidden_feats=config["predictor_hidden_feats"],
                                  n_tasks=1, 
                                  predictor_dropout=0.0)
    
    pos2cohp_model.load_state_dict(torch.load(pos2cohp_model_path))
    
    # Load raw energy data
    with open(os.path.join(root_path, "raw", "icohp_structures_all.pkl"), 'rb') as file_get:
        icohp_list_keys = pickle.load(file_get)
    
    # Load the dataset
    pos2cohp_dataset = POS2COHP_Dataset(root_path, Element_List, setting, icohp_list_keys)

    # Recalculate the predicted COHP values
    pos2cohp_dataloader = DataLoader(pos2cohp_dataset, batch_size=config["batch_size"], shuffle=False)
    pos2cohp_model.eval()
    pos2cohp_model.to(device)
    
    PRED = []
    with torch.no_grad():
        for data in pos2cohp_dataloader:
            data = data.to(device)  # Move data to the same device
            prediction = pos2cohp_model(data)
            PRED.extend(split_batch_data(data, prediction))

    # Load raw energy data
    with open(os.path.join(root_path, "raw", "raw_energy_data_dict_all.pkl"), 'rb') as file_get:
        raw_data_dict = pickle.load(file_get)

    # Initialize the POS2E_edge_Dataset
    POS2E_edge_dataset = POS2E_edge_Dataset(root=root_path, 
                                            setting=setting, 
                                            src_dataset=pos2cohp_dataset, 
                                            predicted_value=PRED, 
                                            raw_data_dict=raw_data_dict
    )

    # Perform prediction for POS2E model
    config = configs["POS2E"]
    
    pos2e_model = CONT2E_Net(in_features=node_feats, 
                             edge_dim=1,
                             bias=False,
                             linear_block_dims=config["linear_block_dims"],
                             conv_block_dims=config["conv_block_dims"],
                             adj_conv=config["adj_conv"],
                             conv=configs_str_mapping[config["conv_edge"]], #configs_str_mapping[config["conv_edge"]]
                             dropout=0.0, 
                             pool=configs_str_mapping[config["pool"]], 
                             pool_dropout=0.0,
                             pool_ratio=config["pool_ratio"], 
                             pool_heads=config["pool_heads"], 
                             pool_seq=config["pool_seq"],
                             pool_layer_norm=config["pool_layer_norm"],
                             pool_type=config["pool_type"])
    
    pos2e_model.load_state_dict(torch.load(pos2e_model_path))
    
    pos2e_model.eval()
    pos2e_model.to(device)
    POS2E_edge_dataloader = DataLoader(POS2E_edge_dataset, batch_size=config["batch_size"], shuffle=False)

    aes = []
    names = []
    with torch.no_grad():
        for data in POS2E_edge_dataloader:
            data = data.to(device)  # Move data to the same device
            prediction = pos2e_model(data).detach().cpu().numpy()
            true_values = data.y.to("cpu").numpy()
            ae = np.abs(np.subtract(prediction, true_values))
            aes.extend(ae)
            names.extend(['%s_%s' % (data.slab[idx], data.metal_pair[idx]) for idx in range(len(data.metal_pair))])
    
    # Check the sorted MAE
    sorted_result = sorted([(i, j) for i, j in zip(aes, names)], key=lambda x: x[0], reverse=True)

    return sorted_result

"""