import os
import argparse
import re
import random

path = os.path.abspath(os.getcwd())

if __name__ == "__main__":
    cohp_pool = os.path.join(path, "cohp_pool")
    finished_pool = os.path.join(path, "finished_pool")

    folders = [f for f in os.listdir(cohp_pool) if os.path.isdir(os.path.join(cohp_pool,f))]
    
    for folder in folders:
        temp_path = os.path.join(cohp_pool, folder)
        keyword = "finished in"

        lobsterout = os.path.join(temp_path, "lobsterout")
        contcar = os.path.join(temp_path, "CONTCAR")
        old_outcar = os.path.join(temp_path, "OUTCAR.opt")
        icohplist = os.path.join(temp_path, "ICOHPLIST.lobster")

        with open(lobsterout, "r") as f:
            content = f.read()
            f.close()
        if keyword in content:
            #finished_jobs.append(folder)
            dst = os.path.join(finished_pool, folder)
            os.mkdir(dst) if not os.path.exists(dst) else None
            os.system("cp %s %s"%(old_outcar, dst))
            os.system("cp %s %s"%(contcar, dst))
            os.system("cp %s %s"%(icohplist, dst))
            os.system("rm -rf %s"%temp_path)
