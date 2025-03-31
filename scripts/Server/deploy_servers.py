import paramiko
import requests
import argparse
import re
import os
import time
import yaml
from multiprocessing import Process


class Server:
    def __init__(self, hostname='', username='', password='', key_file='', 
                 root='', partitions=[], core_allocation=[], job_allocation=[], 
                 job_system='', local_dir_path=''):

        self.hostname = hostname
        self.username = username
        self.password = password
        self.key_file = key_file
        self.root = root
        self.partitions = partitions
        self.job_system = job_system
        self.local_dir_path = local_dir_path
        
        stages = [["opt", self._opt2static, "static_pool"],
                  ["static", self._static2cohp, "cohp_pool"],
                  ["cohp", self._cohp2finished, "finished_pool"]]
        for stage,ca,ja in zip(stages,core_allocation,job_allocation):
            stage.append(ca)
            stage.append(dict(zip(partitions, ja)))
        self.stages = stages


    # 1. Connection Management
    def _connect(self, retry_count=100000, delay=60):
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        for attempt in range(retry_count):
            try:
                if not self.key_file:
                    ssh.connect(hostname=self.hostname, username=self.username, password=self.password)
                else:
                    ssh.connect(hostname=self.hostname, username=self.username, key_filename=self.key_file)
                return ssh
            except paramiko.SSHException as e:
                print(f"SSHException on attempt {attempt + 1}: {e}")
            except Exception as e:
                print(f"Unexpected error on attempt {attempt + 1}: {e}")
            time.sleep(delay)
        raise ConnectionError(f"Failed to connect to {self.hostname} after {retry_count} retries.")

    def is_connected(self):
        try:
            stdin, stdout, stderr = self.ssh.exec_command('echo Connection Test')
            output = stdout.read().decode('utf-8').strip()
            if output == "Connection Test":
                return True
            else:
                return False
        except Exception as e:
            print(f"Connection check failed: {e}")
            return False
    
    def close(self):
        if hasattr(self, 'ssh'):
            self.ssh.close()
    

    # 2. Utility Methods
    def _check_root_path(self):
        self.ssh = self._connect()
        self.sftp = self.ssh.open_sftp()
        dirs = set(self.sftp.listdir(self.root))
        
        required_dirs = set(['opt_pool', 'static_pool', 'cohp_pool', 'finished_pool', 'scripts', 'INPUT', 'timestamp'])
        assert dirs.intersection(required_dirs) == required_dirs
        print(f"'{self.root}' contains: {dirs}")
        self.close()

    def modify_sub_file(self, remote_file_path, job_name=None, partition=None, nodes=None, ntasks=None, python_file=None):
        with self.sftp.open(remote_file_path, 'r') as file:
            sub_content = file.readlines()
        walltime = {"short":36, "medium":240, "long":240}

        modified_content = []
        for line in sub_content:
            # SLURM system
            if job_name and line.startswith('#SBATCH -J'):
                modified_content.append(f'#SBATCH -J {job_name}\n')
            elif partition and line.startswith('#SBATCH -p'):
                modified_content.append(f'#SBATCH -p {partition}\n')
            elif nodes and line.startswith('#SBATCH -N'):
                modified_content.append(f'#SBATCH -N {nodes}\n')
            elif ntasks and line.startswith('#SBATCH -n'):
                modified_content.append(f'#SBATCH -n {ntasks}\n')
            ### tansuo
            elif ntasks and line.startswith('#SBATCH --ntasks-per-node='):
                modified_content.append(f'#SBATCH --ntasks-per-node={ntasks}\n')
            ### hetian
            elif job_name and line.startswith('#SBATCH --job-name='):
                modified_content.append(f'#SBATCH --job-name={job_name}\n')
            elif partition and line.startswith('#SBATCH --partition='):
                modified_content.append(f'#SBATCH --partition={partition}\n')
            elif nodes and line.startswith('#SBATCH --nodes='):
                modified_content.append(f'#SBATCH --nodes={nodes}\n')
            ### tushu
            elif job_name and line.startswith('#PBS -N'):
                modified_content.append(f'#PBS -N {job_name}\n')
            elif nodes and ntasks and line.startswith('#PBS -l nodes='):
                modified_content.append(f'#PBS -l nodes={nodes}:ppn={ntasks}\n')
            elif partition and line.startswith('#PBS -q'):
                modified_content.append(f'#PBS -q {partition}\n')
            elif line.startswith('#PBS -l walltime='):
                modified_content.append(f'#PBS -l walltime={walltime[partition]}:00:00\n')                

            elif python_file and 'python ./scripts/' in line:
                modified_content.append(f'python ./scripts/{python_file}\n')
            else:
                # Keep lines that do not match any of the conditions
                modified_content.append(line)

        with self.sftp.open(remote_file_path, 'w') as file:
            file.writelines(modified_content)      

    def _suit_for_module_load(self, module_load="fhbquybbf"):
        scripts_path = os.path.join(self.root, "scripts").replace("\\", "/")
        file_list = ["opt.py", "static.py", "cohp.py", "general.sub"]

        self.ssh = self._connect()
        self.sftp = self.ssh.open_sftp()

        for file_name in file_list:
            file_path = f"{scripts_path}/{file_name}"
            try:
                self.sftp.stat(file_path)  # Raises FileNotFoundError if not found
            except FileNotFoundError:
                print(f"File not found on remote server: {file_path}")
                continue

            temp_file = file_path + ".tmp"
            try:
                with self.sftp.open(file_path, "r") as infile, self.sftp.open(temp_file, "w") as outfile:
                    for line in infile:
                        if "module load" in line:
                            leading_whitespace = line[:len(line) - len(line.lstrip())]
                            if file_name.endswith(".py"):
                                new_line = f'{leading_whitespace}os.system("{module_load}")\n'
                            else:
                                new_line = f"{leading_whitespace}{module_load}\n"
                            outfile.write(new_line)
                        else:
                            outfile.write(line)

                self.sftp.remove(file_path)  # Remove the original file before renaming
                self.sftp.rename(temp_file, file_path)
                print(f"Updated file on remote server: {file_path}")
            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                try:
                    self.sftp.remove(temp_file)
                except FileNotFoundError:
                    pass
        self.sftp.close()
        self.ssh.close()

    def _check_process(self, pool):
        stdin, stdout, stderr = self.ssh.exec_command('cd %s && ls %s | wc -l && cd ..'%(self.root, pool))
        rest_folder = stdout.read().decode('utf-8')
        return int(rest_folder)  

    """
    def filling_opt_pool(self):
        self.ssh = self._connect()
        self.sftp = self.ssh.open_sftp()
        opt_pool = os.path.join(self.root, "opt_pool").replace("\\", "/")
        
        for sub_dir in os.listdir(self.local_dir_path):
            local_sub_dir_path = os.path.join(self.local_dir_path, sub_dir)
            local_file_path = os.path.join(local_sub_dir_path, "POSCAR")

            remote_dir_path = os.path.join(opt_pool, sub_dir).replace("\\", "/")
            remote_file_path = os.path.join(remote_dir_path, "POSCAR").replace("\\", "/")
            try:
                self.sftp.listdir(remote_dir_path)
            except FileNotFoundError:
                self.sftp.mkdir(remote_dir_path)
                self.sftp.put(local_file_path, remote_file_path)

        self.sftp.close()
        self.ssh.close()
    """
    def filling_opt_pool(self):
        """
        Fills the remote 'opt_pool' directory with files from the local directory.
        Ensures that all necessary directories and files are properly handled.
        """
        try:
            # Establish SSH and SFTP connections using context manager for safety
            self.ssh = self._connect()
            with self.ssh.open_sftp() as sftp:
                opt_pool = os.path.join(self.root, "opt_pool").replace("\\", "/")

                # Check if local directory exists
                if not os.path.exists(self.local_dir_path):
                    raise FileNotFoundError(f"Local directory path {self.local_dir_path} does not exist.")

                # Process each subdirectory in the local directory
                for sub_dir in os.listdir(self.local_dir_path):
                    local_sub_dir_path = os.path.join(self.local_dir_path, sub_dir)
                    local_file_path = os.path.join(local_sub_dir_path, "POSCAR")
                    remote_dir_path = os.path.join(opt_pool, sub_dir).replace("\\", "/")
                    remote_file_path = os.path.join(remote_dir_path, "POSCAR").replace("\\", "/")

                    # Ensure local file exists before attempting transfer
                    if not os.path.exists(local_file_path):
                        print(f"Warning: {local_file_path} does not exist, skipping.")
                        continue

                    # Create remote directory if not exists
                    try:
                        sftp.listdir(remote_dir_path)
                    except FileNotFoundError:
                        print(f"Directory {remote_dir_path} not found, creating it.")
                        try:
                            sftp.mkdir(remote_dir_path)
                            print(f"Created remote directory: {remote_dir_path}")
                        except Exception as e:
                            print(f"Error creating remote directory {remote_dir_path}: {e}")
                            continue

                        # Transfer the file
                        try:
                            print(f"Transferring {local_file_path} to {remote_file_path}...")
                            sftp.put(local_file_path, remote_file_path)
                            print(f"Successfully transferred {local_file_path} to {remote_file_path}.")
                        except Exception as e:
                            print(f"Error transferring {local_file_path} to {remote_file_path}: {e}")
                        
        except Exception as e:
            print(f"Error in filling_opt_pool: {e}")


    # 3. Job Submission and Management
    def check_jobs(self, job_name): 
        if self.job_system == "slurm":
            stdin, stdout, stderr = self.ssh.exec_command('squeue | grep {}'.format(job_name))
        elif self.job_system == "pbs":
            stdin, stdout, stderr = self.ssh.exec_command('qstat | grep {}'.format(job_name))
        else:
            print("Error: Job Schedule System syntax error.")
        output = stdout.read().decode('utf-8')
        partition_counts = {}
        for partition in self.partitions:
            count = len(re.findall(partition, output))
            partition_counts[partition] = count
        return list(partition_counts.values())
            
    
    def submit_job(self, filename="general.sub", job_name="HIGHT", partition="new", nodes="1", ntasks="16", python_file="job.py"):
        filename = f"./{self.root}/scripts/{filename}"
        self.modify_sub_file(remote_file_path=filename, job_name=job_name, partition=partition, nodes=nodes, ntasks=ntasks, python_file=python_file)
          
        if self.job_system == "slurm":
            self.ssh.exec_command(f'cd {self.root} && sbatch ./scripts/general.sub && cd ../')  
        elif self.job_system == "pbs":
            self.ssh.exec_command(f'cd {self.root} && qsub ./scripts/general.sub && cd ../')  
        else:
            print("Error: Job Schedule System syntax error.")
        
        
    def submit_jobs_with_limits(self, filename="general.sub", job_name="HIGHT", nodes="1", ntasks="16", python_file="job.py", partition_limits=None):
        max_job_submit = sum(list(partition_limits.values()))
        submitted_count = 0

        for partition in list(partition_limits.keys()):
            while True:
                current_jobs = dict(zip(list(partition_limits.keys()), self.check_jobs(job_name)))

                if current_jobs[partition] < partition_limits[partition]:
                    print(f"Submitting job to {partition} partition. Current: {current_jobs[partition]}, Limit: {partition_limits[partition]}.")
                    self.submit_job(filename, job_name, partition, nodes, ntasks, python_file)
                    submitted_count += 1
                    if submitted_count > max_job_submit:
                        while True:
                            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Error: Job submission is stucked on {self.hostname}. Check the server.")
                            time.sleep(60*60)
                    time.sleep(5)  # Optional: sleep to avoid overwhelming the server
                else:
                    print(f"Reached the limit for {partition} jobs. Current: {current_jobs[partition]}, Limit: {partition_limits[partition]}.")
                    break  # Exit the loop for this partition when the job limit is reached
    
    def cancel_all_jobs(self):
        self.ssh.exec_command(f'scancel -u {self.username}')


    # 4. Stage Handling and Autotune
    def _opt2static(self):
        self.ssh.exec_command(f'cd %s && python ./scripts/opt2static.py && cd ../'%self.root)
        
    def _static2cohp(self):
        self.ssh.exec_command(f'cd %s && python ./scripts/static2cohp.py && cd ../'%self.root)
        
    def _cohp2finished(self):
        self.ssh.exec_command(f'cd %s && python ./scripts/cohp2finished.py && cd ../'%self.root)

    def check_current_stage(self, max_retries=100, retry_delay=600):
        stages = {"finished_pool": "finished", "cohp_pool": "cohp", "static_pool": "static", "opt_pool": "opt"}

        for attempt in range(max_retries):
            if not hasattr(self, 'ssh') or not self.ssh.get_transport().is_active():
                self.ssh = self._connect()
            if not hasattr(self, 'sftp'):
                self.sftp = self.ssh.open_sftp()

            for pool, stage in stages.items():
                try:
                    pool_path = os.path.join(self.root, pool).replace("\\", "/")
                    file_count = len(self.sftp.listdir(pool_path))

                    if file_count > 0:
                        return stage
                    if file_count == 0 and stage == "opt":
                        return "opt"

                except Exception as e:
                    print(f"Error accessing {pool} on attempt {attempt + 1}: {e}")
                    continue

            # Wait before retrying if stage cannot be determined
            if attempt < max_retries - 1:
                print(f"Unable to determine stage. Retrying in {retry_delay} seconds...")
                time.sleep(retry_delay)
            else:
                print("Maximum retries reached. Defaulting to 'opt' stage.")
                return "opt"  # Default to 'opt' if all retries fail           
        
    def get_finished_jobs_from_pool(self, pool=None, stop_time=1800):
        stdin, stdout, stderr = self.ssh.exec_command(f'cd %s && python ./scripts/check.py %s %s | wc -l && cd ../'%(self.root,pool,stop_time))
        return int(stdout.read().decode('utf-8'))
    

    def get_total_jobs_from_pool(self, pool=None, max_retries=100, retry_delay=600):
        retries = 0
        while retries < max_retries:
            try:
                self.ssh = self._connect()
                self.sftp = self.ssh.open_sftp()

                pool_path = os.path.join(self.root, pool).replace("\\", "/")
                directories = self.sftp.listdir(pool_path)
                return len([d for d in directories if d not in [".", ".."]])
            except Exception as e:
                print(f"Error accessing pool {pool}: {e}. Retry {retries + 1} of {max_retries}.")
                retries += 1
                time.sleep(retry_delay)

        print(f"Failed to access pool {pool} after {max_retries} retries. Returning 0.")
        return 0  

        
    def autotune(self):
        current_stage = self.check_current_stage()
        stage_names = [stage[0] for stage in self.stages]
        if current_stage == "finished":
            print("All stages are already completed.")
            return
        
        start_index = stage_names.index(current_stage)
        for stage_name, transition_function, target_pool, Ncore, partition_limits in self.stages[start_index:]:
            current_pool = stage_name+"_pool"
               
            while True:
                print(self.hostname)
                self.ssh = self._connect()
                self.sftp = self.ssh.open_sftp()

                #if stage_name == "opt":
                    #self.filling_opt_pool()
            
                total_jobs = self.get_total_jobs_from_pool(pool=current_pool)
                finished_jobs = self.get_finished_jobs_from_pool(pool=current_pool)
                completion_ratio = finished_jobs / total_jobs
                print(f"Current completion for {current_pool} stage: {completion_ratio*100:.2f}%")
                
                if completion_ratio > 0.99:#>= completion_threshold:
                    print(f"Transitioning to next stage: {current_pool} -> {target_pool}")
                    #self.cancel_all_jobs()
                    transition_function()
                    time.sleep(10*60) # aiting for all files are moved
                    break
                else:
                    self.submit_jobs_with_limits(filename="general.sub", job_name=stage_name.upper(), nodes="1", ntasks=f"{Ncore}",
                                                 python_file=f"{stage_name}.py", partition_limits=partition_limits)
                    self.close()
                    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]")
                    print("##################################################")
                    time.sleep(60*60)

        print("All stages completed. Jobs are finished.")


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as file:
        return yaml.safe_load(file)


def deploy(server_name, config):
    if server_name not in config["servers"]:
        print(f"Error: Server '{server_name}' not found in the configuration.")
        return

    server_config = config["servers"][server_name]

    server_instance = Server(
        hostname=server_config.get("hostname"),
        username=server_config.get("username"),
        password=server_config.get("password"),
        key_file=server_config.get("key_file"),
        root=server_config.get("root"),
        job_system=server_config.get("job_system"),
        partitions=server_config.get("partitions", []),
        core_allocation=server_config.get("core_allocation", []),
        job_allocation=server_config.get("job_allocation", []),
        local_dir_path=server_config.get("local_dir_path"))

    server_instance.autotune()


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Deploy and manage multiple servers.")
    parser.add_argument(
        "-s", "--servers", required=True, nargs="+",
        help="List of server names to deploy (e.g., tansuo electron hetian)"
    )
    parser.add_argument("-c", "--config", default="config.yaml", help="Path to the configuration YAML file")
    args = parser.parse_args()

    # Load configurations
    config = load_config(args.config)

    # Create and start a process for each server
    processes = []
    for server_name in args.servers:
        p = Process(target=deploy, args=(server_name, config))
        p.start()
        processes.append(p)
        time.sleep(1*60)

    # Wait for all processes to complete
    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
