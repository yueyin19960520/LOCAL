import paramiko
import requests
import argparse
import re
import os
import time


import paramiko
import requests
import argparse
import re
import os
import time


class electron_server:
    def __init__(self, hostname='192.168.215.102', username='yinyue', password='ggapbe2020', root=None, total_jobs=10):
        self.hostname = hostname
        self.username = username
        self.password = password
        self.ssh = self._connect()
        self.sftp = self.ssh.open_sftp()
        self.root = root
        self.total_jobs = total_jobs

    def _connect(self, retry_count=100000, delay=60):
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        for attempt in range(retry_count):
            try:
                ssh.connect(hostname=self.hostname, username=self.username, password=self.password)
                return ssh
            except:
                if attempt < retry_count - 1:  # not the last attempt
                    print(attempt)
                    time.sleep(delay)  # wait before retrying
                    continue
                else:
                    print("...")  # re-raise the exception if the last attempt fails
        return ssh

    def check_if_folder_exists(self, folder):
        folder_path = os.path.join("/home/yinyue/DSAC/", folder)
        try:
            sftp = self.ssh.open_sftp()
            try:
                sftp.stat(folder_path)
                return True
            except IOError:
                print(f"Folder {folder_path} does not exist.")
                return False
            finally:
                sftp.close()
        except Exception as e:
            print(f"Error checking folder: {e}")
            return False

    def check_jobs(self):
        stdin, stdout, stderr = self.ssh.exec_command('squeue | grep yinyue')
        output = stdout.read().decode('utf-8')
        LONG, MEDIUM, SHORT = len(re.findall("long", output)), len(re.findall("medium", output)), len(re.findall("short", output))
        NEW = len(re.findall("new", output))
        return LONG, MEDIUM, SHORT, NEW     
            
    
    def submit_job(self, filename="general.sub", job_name="HIGHT", partition="new", nodes="1", ntasks="16", python_file="job.py"):
        filename = f"./{self.root}/scripts/{filename}"
        self.modify_sub_file(remote_file_path=filename, job_name=job_name, partition=partition, nodes=nodes, ntasks=ntasks, python_file=python_file)
        self.ssh.exec_command(f'cd {self.root} && sbatch ./scripts/general.sub && cd ../')    
        
        
    def submit_jobs_with_limits(self, filename="general.sub", job_name="HIGHT", nodes="1", ntasks="16", python_file="job.py"):
        max_jobs = {"long": 1, "medium": 2, "short":0, "new": 2}

        for partition in list(max_jobs.keys()):
            while True:
                current_jobs = dict(zip(list(max_jobs.keys()), self.check_jobs()))

                if current_jobs[partition] < max_jobs[partition]:
                    print(f"Submitting job to {partition} partition. Current: {current_jobs[partition]}, Limit: {max_jobs[partition]}.")
                    self.submit_job(filename, job_name, partition, nodes, ntasks, python_file)
                    time.sleep(1)  # Optional: sleep to avoid overwhelming the server
                else:
                    print(f"Reached the limit for {partition} jobs. Current: {current_jobs[partition]}, Limit: {max_jobs[partition]}.")
                    break  # Exit the loop for this partition when the job limit is reached

        
    def _opt2static(self):
        self.ssh.exec_command(f'cd %s && python ./scripts/opt2static.py && cd ../'%self.root)
        
    def _static2cohp(self):
        self.ssh.exec_command(f'cd %s && python ./scripts/static2cohp.py && cd ../'%self.root)
        
    def _cohp2finished(self):
        self.ssh.exec_command(f'cd %s && python ./scripts/cohp2finished.py && cd ../'%self.root)
        
        
    def _check_process(self, pool):
        stdin, stdout, stderr = self.ssh.exec_command('cd %s && ls %s | wc -l && cd ..'%(self.root, pool))
        rest_folder = stdout.read().decode('utf-8')
        return int(rest_folder)  
    

    def calculate_finished_jobs(self, pool=None, stop_time=1800):
        stdin, stdout, stderr = self.ssh.exec_command(f'cd %s && python ./scripts/check.py %s %s | wc -l && cd ../'%(self.root,pool,stop_time))
        return int(stdout.read().decode('utf-8'))
    
    
    def cancel_all_jobs(self):
        self.ssh.exec_command(f'scancel -u {self.username}')
    
    def close(self):
        self.ssh.close()
        
        
    def modify_sub_file(self, remote_file_path, job_name=None, partition=None, nodes=None, ntasks=None, python_file=None):
        with self.sftp.open(remote_file_path, 'r') as file:
            sub_content = file.readlines()

        for i, line in enumerate(sub_content):
            if job_name and line.startswith('#SBATCH -J'):
                sub_content[i] = f'#SBATCH -J {job_name}\n'
            if partition and line.startswith('#SBATCH -p'):
                sub_content[i] = f'#SBATCH -p {partition}\n'
            if nodes and line.startswith('#SBATCH -N'):
                sub_content[i] = f'#SBATCH -N {nodes}\n'
            if ntasks and line.startswith('#SBATCH -n'):
                sub_content[i] = f'#SBATCH -n {ntasks}\n'
            if python_file and 'python ./scripts/' in line:
                sub_content[i] = f'python ./scripts/{python_file}\n'

        with self.sftp.open(remote_file_path, 'w') as file:
            file.writelines(sub_content)
            
            
    def autotune(self):
        stages = [("opt", self._opt2static, "static_pool", 0.95),
                  ("static", self._static2cohp, "cohp_pool", 0.90),
                  ("cohp", self._cohp2finished, "finished_pool", 0.90)]

        for stage_name, transition_function, pool_name, completion_threshold in stages:
            while True:
                self.ssh = self._connect()
                self.sftp = self.ssh.open_sftp()
            
                total_jobs = self.total_jobs
                finished_jobs = self.calculate_finished_jobs(pool=stage_name+"_pool")
                completion_ratio = finished_jobs / total_jobs
                print(f"Current completion for {stage_name} stage: {completion_ratio*100:.2f}%")
                
                if completion_ratio >= completion_threshold:
                    print(f"Transitioning to next stage: {stage_name} -> {pool_name}")
                    self.cancel_all_jobs()
                    time.sleep(60)
                    transition_function()
                    break
                else:
                    None

                self.submit_jobs_with_limits(filename="general.sub", job_name=stage_name.upper(), nodes="1", ntasks="16", python_file=f"{stage_name}.py")
                self.close()
                print(time.localtime())
                time.sleep(30*60)

        print("All stages completed. Jobs are finished.")

    def sent_message(self, lunch_info): 
        #res = self.check_jobs()
        value1 = "Server"
        value2 = "Electron"
        value3 = lunch_info#"long:{},medium:{},short:{},new:{}".format(*res)
        event_name = "Push"
        key = "cbgG6OygzBbJGpXxeSbJgz"
        url = "https://maker.ifttt.com/trigger/"+event_name+"/with/key/"+key+""
        payload = "{\n    \"value1\": \""+value1+"\",  \n  \"value2\": \""+value2+"\",  \n  \"value3\": \""+value3+"\"    \n}"
        headers = {'Content-Type': "application/json",'User-Agent': "PostmanRuntime/7.15.0", 'Accept': "*/*",
        'Cache-Control': "no-cache",'Postman-Token': "a9477d0f-08ee-4960-b6f8-9fd85dc0d5cc,d376ec80-54e1-450a-8215-952ea91b01dd",
        'Host': "maker.ifttt.com",'accept-encoding': "gzip, deflate",
        'content-length': "63",'Connection': "keep-alive",'cache-control': "no-cache"}
        response = requests.request("POST", url, data=payload.encode('utf-8'), headers=headers)
        return None


"""
class tansuo1000_server:
    def __init__(self, hostname='192.168.11.1', username='lijun', keyfile="D:/tsinghua/lijun_new_ssh"):
        self.hostname = hostname
        self.username = username
        self.keyfile = keyfile
        self.ssh = self._connect()

    def _connect(self, retry_count=100000, delay=60):
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        #ssh.connect(hostname=self.hostname, username=self.username, keyfile=self.keyfile)
        
        for attempt in range(retry_count):
            try:
                ssh.connect(hostname=self.hostname, username=self.username, key_filename=self.keyfile)
                return ssh
            except:
                if attempt < retry_count - 1:  # not the last attempt
                    print(attempt)
                    time.sleep(delay)  # wait before retrying
                    continue
                else:
                    print("...")  # re-raise the exception if the last attempt fails
        return ssh

    def check_if_folder_exists(self, folder):
        folder_path = os.path.join("/home/lijun/WORK/yiny/DSAC/", folder)
        try:
            sftp = self.ssh.open_sftp()
            try:
                sftp.stat(folder_path)
                return True
            except IOError:
                print(f"Folder {folder_path} does not exist.")
                return False
            finally:
                sftp.close()
        except Exception as e:
            print(f"Error checking folder: {e}")
            return False

    def check_jobs(self):
        stdin, stdout, stderr = self.ssh.exec_command('squeue')
        output = stdout.read().decode('utf-8')
        list(map(lambda line: print(' | '.join(line.split())), output.split('\n')))
        jobs = len(re.findall("HIGHT", output))
        print(f"Number of jobs: {jobs}. \n")
        return jobs

    def check_folder(self, folder):
        assert self.check_if_folder_exists(folder)
        stdin, stdout, stderr = self.ssh.exec_command(f'cd ./WORK/yiny/DSAC && python ./scripts/monitor_status.py -n {folder} && cd ~')
        output = stdout.read().decode('utf-8')
        print(output)
        return None

    def run(self,folder):
        self.ssh.exec_command(f'cd ./WORK/yiny/DSAC && sed -i "s/folder/{folder}/g" run.sub && cd ~')
        self.ssh.exec_command(f'cd ./WORK/yiny/DSAC && sbatch run.sub && cd ~')
        self.ssh.exec_command(f'cd ./WORK/yiny/DSAC && sed -i "s/{folder}/folder/g" run.sub && cd ~')
        time.sleep(5)
        return None

    def autorun(self, folder, MAX_JOB=12):
        assert self.check_if_folder_exists(folder)
        lunch_info = None
        JOB = self.check_jobs()
        for _ in range(MAX_JOB - JOB):
            self.run(folder)
        NEW_JOB = self.check_jobs()

        lunched_jobs = (NEW_JOB-JOB)
        if lunched_jobs > 0:
            lunch_info = "Lunched %s jobs."%(NEW_JOB-JOB)
            print(lunch_info)
        else:
            print("No job lunched.")
        return lunch_info

    def sent_message(self, lunch_info): 
        #res = self.check_jobs()
        value1 = "Server"
        value2 = "Tansuo1000"
        value3 = lunch_info#"long:{},medium:{},short:{},new:{}".format(*res)
        event_name = "Push"
        key = "cbgG6OygzBbJGpXxeSbJgz"
        url = "https://maker.ifttt.com/trigger/"+event_name+"/with/key/"+key+""
        payload = "{\n    \"value1\": \""+value1+"\",  \n  \"value2\": \""+value2+"\",  \n  \"value3\": \""+value3+"\"    \n}"
        headers = {'Content-Type': "application/json",'User-Agent': "PostmanRuntime/7.15.0", 'Accept': "*/*",
        'Cache-Control': "no-cache",'Postman-Token': "a9477d0f-08ee-4960-b6f8-9fd85dc0d5cc,d376ec80-54e1-450a-8215-952ea91b01dd",
        'Host': "maker.ifttt.com",'accept-encoding': "gzip, deflate",
        'content-length': "63",'Connection': "keep-alive",'cache-control': "no-cache"}
        response = requests.request("POST", url, data=payload.encode('utf-8'), headers=headers)
        return None

    def close(self):
        self.ssh.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Agent for lunching jobs on server.')
    parser.add_argument('-s', '--server', type=str, required=True, help='Electron or Tansuo1000')
    parser.add_argument('-f', '--folder', type=str, required=True, help='folder name in server')
    parser.add_argument('-d', '--days', type=int, default=15, help='how many days to automate.')
    args = parser.parse_args()
    assert args.server in ['Electron', 'Tansuo1000']

    hours = args.days * 24
    folder = args.folder

    for i in range(hours):
        if args.server == "Electron":
            server= electron_server()
        if args.server == "Tansuo1000":
            server= tansuo1000_server()
        print("Connection is Successful!")
        lunch_info = server.autorun(folder)
        if lunch_info is not None:
            server.sent_message(lunch_info)
        server.check_folder(folder)
        server.close()
        time.sleep(3600)
"""