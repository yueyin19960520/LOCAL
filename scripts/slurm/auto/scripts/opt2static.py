import os
import argparse
import re
import random

path = os.path.abspath(os.getcwd())

if __name__ == "__main__":
    opt_pool = os.path.join(path, "opt_pool")
    static_pool = os.path.join(path, "static_pool")

    folders = [f for f in os.listdir(opt_pool) if os.path.isdir(os.path.join(opt_pool,f))]
    
    for folder in folders:
        temp_path = os.path.join(opt_pool, folder)
        keyword = "reached required accuracy - stopping structural energy minimisation"

        outcar = os.path.join(temp_path, "OUTCAR")
        contcar = os.path.join(temp_path, "CONTCAR")

        with open(outcar, "r") as f:
            content = f.read()
            f.close()
        if keyword in content:
            #finished_jobs.append(folder)
            dst = os.path.join(static_pool, folder)
            poscar = os.path.join(dst,"POSCAR")
            new_outcar = os.path.join(dst,"OUTCAR.opt")
            os.mkdir(dst) if not os.path.exists(dst) else None
            os.system("cp %s %s"%(outcar, new_outcar))
            os.system("cp %s %s"%(contcar, poscar))
            os.system("rm -rf %s"%temp_path)
