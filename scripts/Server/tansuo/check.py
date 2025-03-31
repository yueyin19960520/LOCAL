import os
import time
import argparse
from datetime import datetime, timedelta

def is_job_finished(job_dir, threshold_seconds):
    """Check if a job is finished based on two criteria:
    1. No files in the directory have been modified within the specified threshold (in seconds).
    2. The number of files in the directory is greater than 10.
    """
    current_time = time.time()
    threshold_time = current_time - threshold_seconds

    file_count = 0
    for root, dirs, files in os.walk(job_dir):
        file_count += len(files)
        for file in files:
            file_path = os.path.join(root, file)
            if os.path.getmtime(file_path) > threshold_time:
                # If any file was modified within the threshold time, job is not finished
                return False

    # Check if the number of files is greater than 10
    if file_count > 10:
        return True
    else:
        return False

def check_finished_jobs(root, threshold_seconds):
    """Check all job directories in the root directory to see if they are finished."""
    job_dirs = [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    finished_jobs = []

    for job_dir in job_dirs:
        if is_job_finished(job_dir, threshold_seconds):
            finished_jobs.append(job_dir)

    return finished_jobs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check finished jobs based on file modification time and file count.")
    parser.add_argument('root', type=str, help="The root directory containing job directories.")
    parser.add_argument('threshold', type=int, help="Time threshold in seconds. Jobs with no file changes within this time are considered finished.")

    args = parser.parse_args()

    # Check finished jobs in the root directory
    finished_jobs = check_finished_jobs(args.root, args.threshold)

    if finished_jobs:
        for job in finished_jobs:
            print(f"- {job}")
    else:
        None