from datetime import datetime, timedelta
import os 
import time

while 1:
    os.system("python3 3D-ResNets-PyTorch-TimeCycle/test_all.py --result_path new_implementation --begin_epoch 4 --gpu_id 1 && python3 3D-ResNets-PyTorch-TimeCycle/test_all.py --result_path new_implementation --begin_epoch 4 --gpu_id 1")

    dt = datetime.now() + timedelta(hours=1)
    dt = dt.replace(minute=10)

    while datetime.now() < dt:
        time.sleep(1)