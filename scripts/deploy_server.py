import paramiko
import requests
import argparse
import re
import os
import time


class electron_server:
    def __init__(self, hostname='192.168.215.102', username='yinyue', password='ggapbe2020'):
        self.hostname = hostname
        self.username = username
        self.password = password
        self.ssh = self._connect()

    def _connect(self, retry_count=100000, delay=60):
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        #ssh.connect(hostname=self.hostname, username=self.username, password=self.password)

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
        list(map(lambda line: print(' | '.join(line.split())), output.split('\n')))
        LONG, MEDIUM, SHORT = len(re.findall("long", output)), len(re.findall("medium", output)), len(re.findall("short", output))
        NEW = len(re.findall("new", output))
        print(f"Long jobs: {LONG}, Medium jobs: {MEDIUM}, New jobs: {NEW}, Short jobs: {SHORT}. \n")
        return LONG, MEDIUM, SHORT, NEW

    def check_folder(self, folder):
        assert self.check_if_folder_exists(folder)
        stdin, stdout, stderr = self.ssh.exec_command(f'cd DSAC && python ./scripts/monitor_status.py -n {folder} && cd ../')
        output = stdout.read().decode('utf-8')
        print(output)
        return None

    def run(self, queue, folder):
        assert queue in ["short", "medium", "long", "new"]
        self.ssh.exec_command(f'cd DSAC && sed -i "s/NONE/{queue}/g" run.sub && cd ../')
        self.ssh.exec_command(f'cd DSAC && sed -i "s/folder/{folder}/g" run.sub && cd ../')
        self.ssh.exec_command(f'cd DSAC && sbatch run.sub && cd ../')
        self.ssh.exec_command(f'cd DSAC && sed -i "s/{queue}/NONE/g" run.sub && cd ../')
        self.ssh.exec_command(f'cd DSAC && sed -i "s/{folder}/folder/g" run.sub && cd ../')
        return None

    def autorun(self, folder, MAX_LONG=2, MAX_MEDIUM=4, MAX_SHORT=0, MAX_NEW=10):
        assert self.check_if_folder_exists(folder)
        LONG, MEDIUM, SHORT, NEW = self.check_jobs()
        lunch_info = None
        for _ in range(MAX_LONG - LONG):
            self.run("long", folder)
        for _ in range(MAX_MEDIUM - MEDIUM):
            self.run("medium", folder)
        for _ in range(MAX_SHORT - SHORT):
            self.run("short", folder)
        for _ in range(MAX_NEW - NEW):
            self.run("new", folder)
        NEW_LONG, NEW_MEDIUM, NEW_SHORT, NEW_NEW = self.check_jobs()

        lunched_jobs = (NEW_LONG-LONG) + (NEW_MEDIUM-MEDIUM) + (NEW_SHORT-SHORT) + (NEW_NEW-NEW)
        if lunched_jobs > 0:
            lunch_info = "Lunched %s long jobs, %s medium jobs, %s short jobs, %s new jobs."%(NEW_LONG-LONG, NEW_MEDIUM-MEDIUM, NEW_SHORT-SHORT, NEW_NEW-NEW)
            print(lunch_info)
        else:
            print("No job lunched.")
        return lunch_info

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

    def close(self):
        self.ssh.close()


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