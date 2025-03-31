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
        lobsterout = os.path.join(root, i, "lobsterout")
        if os.path.exists(lobsterout):
            content = os.popen('grep "finished in" %s | tail -n 1'%(lobsterout)).read()
            if 'finished in' in content:
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
    lobsterin = os.path.join(p_dir, "INPUT", "lobsterin")
    os.chdir(folder)

    if os.path.isfile(os.path.join(folder, "lobsterin")):
        return
    
    os.system("cp %s %s"%(lobsterin, os.path.join(folder,"lobsterin")))                      # KPOINTS
    os.system("lobster") 

if __name__ == "__main__":
    p_dir=os.getcwd()
    root = os.path.join(p_dir,"cohp_pool")  #name is opt_pool????
    
    sleep_t = random.randint(10,30)
    time.sleep(sleep_t*1)
    
    os.system("module load VASP/6.3.0")
   
    while True:
        rerun_list = construct_run_list(root)
        if rerun_list == []:
            break
        else:
            run(p_dir, rerun_list[0], root)