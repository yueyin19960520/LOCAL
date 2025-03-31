import os
import datetime
import argparse
import random
import time


def prepare_rerun(root):
    l1 = [f for f in os.listdir(root) if os.path.isdir(os.path.join(root,f))]
    fi = []
    un = []

    incar_lines = ["############\n","IMIX = 4\n","AMIX = 0.2\n","BMIX = 0.001\n"]

    for i in l1:
        incar = os.path.join(root, i, "INCAR")

        un.append(i)
        with open(incar, "a") as incar_file:
            incar_file.writelines(incar_lines)
        
    for f in un:
        folder = os.path.join(root,f)
        clean_directory(folder)
    return None

def clean_directory(directory):
    keep_files = {"POSCAR", "INCAR","OUTCAR.opt"}
    for file in os.listdir(directory):
        if file not in keep_files:
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

if __name__ == "__main__":
    p_dir=os.getcwd()
    root = os.path.join(p_dir,"static_pool")  #name is opt_pool????
    prepare_rerun(root)
