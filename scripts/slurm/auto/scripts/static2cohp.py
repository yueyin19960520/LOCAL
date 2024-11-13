import os
import argparse
import re
import random

path = os.path.abspath(os.getcwd())

if __name__ == "__main__":
    static_pool = os.path.join(path, "static_pool")
    cohp_pool = os.path.join(path, "cohp_pool")

    folders = [f for f in os.listdir(static_pool) if os.path.isdir(os.path.join(static_pool,f))]
    
    for folder in folders:
        temp_path = os.path.join(static_pool, folder)
        keyword = "aborting loop because EDIFF is reached"

        outcar = os.path.join(temp_path, "OUTCAR")
        contcar = os.path.join(temp_path, "CONTCAR")
        old_outcar = os.path.join(temp_path, "OUTCAR.opt")
        kpoints = os.path.join(temp_path, "KPOINTS")
        wavecar = os.path.join(temp_path, "WAVECAR")
        vasprun = os.path.join(temp_path, "vasprun.xml")
        potcar = os.path.join(temp_path, "POTCAR")

        with open(outcar, "r") as f:
            content = f.read()
            f.close()
        if keyword in content:
            #finished_jobs.append(folder)
            dst = os.path.join(cohp_pool, folder)
            os.mkdir(dst) if not os.path.exists(dst) else None
            os.system("cp %s %s"%(outcar, dst))
            os.system("cp %s %s"%(contcar, dst))
            os.system("cp %s %s"%(old_outcar, dst))
            os.system("cp %s %s"%(kpoints, dst))
            os.system("cp %s %s"%(wavecar, dst))
            os.system("cp %s %s"%(potcar, dst))
            os.system("cp %s %s"%(vasprun, dst))
            os.system("rm -rf %s"%temp_path)
