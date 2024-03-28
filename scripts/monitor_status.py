import os
import argparse
import re
import random


path = os.path.abspath(os.getcwd())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str, required=True, help='the name of parent folder')
    args = parser.parse_args()

    name = args.name

    finish_path = os.path.join(path, name+"_finish")
    path = os.path.join(path,name)

    os.mkdir(finish_path) if not os.path.exists(finish_path) else None
   
    folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path,f))]

    unconverged_jobs = []    # unconverged jobs
    finished_jobs = []   # reached minimum energy jobs
    untorched_jobs = []  # jobs even not in the queue 
    waiting_jobs = []    # jobs waitting in the queue
   
    num_files = []

    for folder in folders:
        temp_path = os.path.join(path,folder)
        content = os.listdir(temp_path)
        num_files.append(len(content))

        if len(content) == 2:
            untorched_jobs.append(folder)
        if len(content) > 2 and len(content) < 7:
            waiting_jobs.append(folder)
        if len(content) >= 7:
            keyword = "reached required accuracy - stopping structural energy minimisation"
            outcar = os.path.join(temp_path, "OUTCAR")
            contcar = os.path.join(temp_path, "CONTCAR")
            with open(outcar, "r") as f:
                content = f.read()
                f.close()
            if keyword in content:
                #finished_jobs.append(folder)
                dst = os.path.join(finish_path, folder)
                os.mkdir(dst) if not os.path.exists(dst) else None
                os.system("cp %s %s"%(outcar, dst))
                os.system("cp %s %s"%(contcar, dst))
                os.system("rm -rf %s"%temp_path)
            else:
                #print(folder)
                unconverged_jobs.append(folder)

    finished_folders = [f for f in os.listdir(finish_path) if os.path.isdir(os.path.join(finish_path,f))]
    finished_jobs = finished_folders

    now_folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    total_jobs = len(now_folders) + len(finished_folders)
    #assert total_jobs == len(unconverged_jobs) + len(finished_jobs) + len(waiting_jobs) + len(untorched_jobs) 

    print(set(num_files))
    #print(waiting_jobs)
    print("Total Jobs:%s, Unconverged Jobs:%s, Finished Jobs:%s, Waiting Jobs:%s, Untorched Jobs:%s"
          %(total_jobs, len(unconverged_jobs), len(finished_jobs), len(waiting_jobs), len(untorched_jobs)))
