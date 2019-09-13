import time
import os
import shutil


def wait_endpoint_setup(container):
    start_time = time.time()
    tick = start_time
    running = False
    logs = ""
    while tick - start_time < 300:
        tick = time.time()
        try:
            logs = str(container.logs())
            if "Server listens on port" in logs:
                running = True
                break
        except Exception as e:
            time.sleep(1)
    print("Logs from container: ", logs)
    return running


def copy_model(model, version, destination_path):
    dir_to_cpy = destination_path + str(version)
    if not os.path.exists(dir_to_cpy):
        os.makedirs(dir_to_cpy)
        shutil.copy(model[0], dir_to_cpy + '/model.bin')
        shutil.copy(model[1], dir_to_cpy + '/model.xml')
    return dir_to_cpy
