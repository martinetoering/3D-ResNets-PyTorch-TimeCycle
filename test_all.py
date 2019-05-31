import os
import sys
import json



if __name__ == '__main__':
    folder = sys.argv[1]
    for file in os.listdir(folder):
        if file.endswith(".pth"):
            result_folder = folder.split("_hmdb51")[0]
            result_folder = result_folder
            print("PAT H file:", result_folder)
            path_file = os.path.join(folder, file)
            os.system("python3 /home/martine/3D-ResNets-PyTorch-TimeCycle/main.py --no_train --resume_path " + str(path_file) + " --result_path " + str(result_folder) + " --print_per_epoch")