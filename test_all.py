import os
import sys
import json



if __name__ == '__main__':
    folder = sys.argv[1]
    video_list = sys.argv[2]
    annotation = sys.argv[3]
    epoch_1 = sys.argv[4]
    epoch_2 = sys.argv[5]

    for file in sorted(os.listdir(folder)):
        if file.endswith(".pth"):
            original_result_path = folder

            result_folder = folder.split("_hmdb51")[0]
            result_folder = result_folder
            
            path_file = os.path.join(folder, file)

            number = file.split("_")[1]
            number = int(number.split(".")[0])
            
            results_file = "results_{}.txt".format(number)
            results_file_path = os.path.join(original_result_path, results_file)
        
            if number > int(epoch_1) and number < int(epoch_2):
                
                if os.path.isfile(results_file_path) is False:

                    print("Checkpoint:", file)
                    print("Result folder:", original_result_path)
                    print("Number:", number)
                    print("Results file path:", results_file_path)

                    # exit() 

                    os.system("python3 /home/martine/3D-ResNets-PyTorch-TimeCycle/main.py --list " + video_list + " --annotation_path " + annotation + " --no_train --resume_path " + str(path_file) + " --result_path " + str(result_folder) + " --general_eval_file" + " --print_per_epoch --gpu_id 1")
            else:
                continue