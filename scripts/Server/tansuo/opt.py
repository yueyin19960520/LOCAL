import os
import datetime
import argparse
import random
import time


def construct_run_list(root):
    l1 = [f for f in os.listdir(root) if os.path.isdir(os.path.join(root,f))]
    fi = []
    un = []
    for i in l1:
        kpoints = os.path.join(root, i, "KPOINTS")
        outcar = os.path.join(root, i, "OUTCAR")
        if os.path.exists(kpoints): 
            content = os.popen('grep "reached required accuracy - stopping structural energy minimisation" %s' % (outcar)).read()
            if 'reached required accuracy - stopping structural energy minimisation' in content:
                fi.append(i)
            #else:
                #un.append(i)
        else:
            un.append(i)

    random.shuffle(un)
    return un


def run(p_dir, folder, root):
    os.chdir(root)

    folder = os.path.join(root, folder)
    incar = os.path.join(p_dir,"INPUT","INCAR.opt")
    kpoints = os.path.join(p_dir,"INPUT","KPOINTS")
    os.chdir(folder)

    if os.path.isfile(os.path.join(folder, "KPOINTS")):
        return
    if not os.path.isfile(os.path.join(folder, "INCAR")):
        os.system("cp %s %s"%(incar, os.path.join(folder,"INCAR")))

    os.system("cp %s %s"%(kpoints, os.path.join(folder,"KPOINTS")))                      # KPOINTS
    os.system('echo -e "1\n103\n" | vaspkit >/dev/null 2>&1')    # POTCAR 
            
    os.system("module load compilers/intel/oneapi-2023/config soft/vasp/vasp.6.3.2")
    os.system("mpirun vasp_std > vasp.out 2>vasp.err")


if __name__ == "__main__":
    p_dir=os.getcwd()
    root = os.path.join(p_dir,"opt_pool")  #name is opt_pool????
    
    sleep_t = random.randint(10,30)
    time.sleep(sleep_t*5)
    
    os.system("module load compilers/intel/oneapi-2023/config soft/vasp/vasp.6.3.2")
    
    while True:
        rerun_list = construct_run_list(root)
        if rerun_list == []:
            break
        else:
            run(p_dir, rerun_list[0], root)
