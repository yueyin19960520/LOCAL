import os
import random 
from itertools import permutations, combinations, product
from pymatgen.core import  Lattice
from pymatgen.io.vasp import Poscar
from pymatgen.core import Structure
import argparse
import copy
import numpy as np


def modify(structure, ele_pair):
    new_structure = copy.deepcopy(structure)
    ori_lattice = structure.lattice
    lattice = Lattice.from_parameters(a=ori_lattice.a, b=ori_lattice.b, c=15, 
                                                    alpha=ori_lattice.alpha, beta=ori_lattice.beta, gamma=ori_lattice.gamma)
    new_structure = Structure(lattice, new_structure.species, new_structure.cart_coords, coords_are_cartesian=True)

    MM_distance = new_structure[-1].distance(new_structure[-2])
    MM_vector = new_structure[-1].coords - new_structure[-2].coords
    #sum_pyykko_radius = (pyykko_dict[ele_pair[0]] + pyykko_dict[ele_pair[1]])/100

    new_structure.replace(-2, ele_pair[0])
    new_structure.replace(-1, ele_pair[1])
    
    """
    height = 0.5
    
    if MM_distance < sum_pyykko_radius:
        enlarge_ratio = (sum_pyykko_radius/MM_distance - 1)
        new_coords_1 = new_structure.sites[-1].coords + (enlarge_ratio/2)*MM_vector + np.array([0., 0., height])
        new_coords_2 = new_structure.sites[-2].coords - (enlarge_ratio/2)*MM_vector + np.array([0., 0., height])
    else:
        new_coords_1 = new_structure.sites[-1].coords + np.array([0., 0., height])
        new_coords_2 = new_structure.sites[-2].coords + np.array([0., 0., height])
        
    new_structure[-1].coords = new_coords_1
    new_structure[-2].coords = new_coords_2
    new_MM_distance = new_structure[-1].distance(new_structure[-2])

    new_coords_1 += np.array([0, 0, 0.2], dtype=new_coords_1.dtype)
    new_coords_2 += np.array([0, 0, 0.2], dtype=new_coords_2.dtype)
    """
    return new_structure


def check_dir_POSCAR(path,check_pair):
    structure = Poscar.from_file(path+"/%s/POSCAR"%check_pair).structure
    ele1 = structure[-2].specie.name
    ele2 = structure[-1].specie.name
    return check_pair == "%s_%s"%(ele1,ele2)



if __name__ == "__main__":
    global pyykko_dict
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, required=True, help="folder name")
    args = parser.parse_args()

    name = args.name

    path = os.path.abspath(os.getcwd())
    path = os.path.join(path, name)

    ori_structure = Poscar.from_file(path + "/%s.vasp"%name).structure
    path = os.path.join(path, "POSCARs")
    print(path)
    os.mkdir(path) if not os.path.exists(path) else None

    element_list = ["Sc", "Ti", "V" , "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", 
                                "Y" , "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
                                "Ce", "Hf", "Ta", "W",  "Re", "Os", "Ir", "Pt", "Au", 
                                "Al", "Ga", "Ge", "In", "Sn", "Sb", "Tl", "Pb", "Bi"]

    #element_list = ["Sc", "Ti", "V" , "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn"]

    """pyykko_dict = {'Sc': 148, 'Ti': 136, 'V' : 134, 'Cr': 122, 'Mn': 119, 'Fe': 116, 'Co': 111, 'Ni': 110, 'Cu': 112, 'Zn': 118, 
                               'Y' : 163, 'Zr': 154, 'Nb': 147, 'Mo': 138, 'Tc': 128, 'Ru': 125, 'Rh': 125, 'Pd': 120, 'Ag': 128, 'Cd': 136, 
                               'Ce': 163, 'Hf': 152, 'Ta': 146, 'W' : 137, 'Re': 131, 'Os': 129, 'Ir': 122, 'Pt': 123, 'Au': 124, 'Hg': 133,
                               'Al': 126, 'Ga': 124, 'Ge': 121, 'In': 142, 'Sn': 140, 'Sb': 140, 'Tl': 144, 'Pb': 144, 'Bi': 151}"""

    symmetry_dict = {"QV1_C0N6":True,"QV1_C6N0":True,"QV2_C0N6":True,"QV2_C6N0":True,"QV3_C0N6":True,"QV3_C6N0":True,
                    "QV4":False, "QV5":True, "QV6":True}

    if symmetry_dict[name]:
        ele_pairs = [(x, y) for x, y in product(element_list, repeat=2) if x <= y]
        assert (len(element_list)*(len(element_list)+1))/2 == len(ele_pairs)
    else:
        ele_pairs = [(x, y) for x, y in product(element_list, repeat=2)]
        assert (len(element_list)**2) == len(ele_pairs)

    for ele_pair in ele_pairs:
        temp_path = os.path.join(path, "%s_%s"%(ele_pair[0], ele_pair[1]))
        os.mkdir(temp_path) if not os.path.exists(temp_path) else None
        structure = modify(ori_structure, ele_pair)
        writer = Poscar(structure)
        writer.write_file(temp_path + "/POSCAR")

    check_pair = random.sample(os.listdir(path), 1)[0]    
    print("The %s file contains %s sub files."%(name, len(os.listdir(path))))
    print("Check the dir_name of the corresponding POSCAR is : %s"%check_dir_POSCAR(path, check_pair))
