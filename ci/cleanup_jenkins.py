# Copyright (c) 2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#


import re
from pathlib import Path
import shutil
import psutil
import argparse

def get_free_space_gb(drive):
    """Return the free space of the given drive in GB."""
    usage = psutil.disk_usage(drive)
    return usage.free / (1024 ** 3)

def get_sorted_folders(path, pattern):
    """Return the oldest folder matching the pattern in the given path."""
    folders = [f for f in Path(path).iterdir() if f.is_dir() and re.search(pattern, f.name)]
    if not folders:
        return None
    folders.sort(key=lambda f: f.stat().st_mtime, reverse=False)
    return folders

def main(no_dry_run=False, min_free_space_gb=50):
    drive = 'C:\\'
    workspace_path = 'C:\\'
    pattern = r'PR-[0-9]'
    sorted_folders = get_sorted_folders(workspace_path, pattern)
    for oldest_folder in sorted_folders:
        if get_free_space_gb(drive) > min_free_space_gb:
            break
        print(oldest_folder)
        if not oldest_folder:
            print("No folders matching the pattern found.")
            break
        print(f"Deleting folder: {oldest_folder}")
        try:
            if no_dry_run:
                shutil.rmtree(oldest_folder)
            print(f"Deleted folder: {oldest_folder}")
        except Exception as e:
            print(f"Error deleting folder {oldest_folder}: {e}")
        
    print("Free space is " + str(get_free_space_gb(drive)) + " Exiting script.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Cleanup directories script.")
    parser.add_argument('--no-dry-run', action='store_true', help="Delete the folders instead of just listing.")
    parser.add_argument('--min-free-space-gb', type=int, default=50, help="Minimum free space in GB to maintain.")
    args = parser.parse_args()

    if not args.no_dry_run:
        print("Dry run mode enabled. No folders will be deleted.")
    main(no_dry_run=args.no_dry_run, min_free_space_gb=args.min_free_space_gb)

