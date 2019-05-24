
import numpy as np
import os
from pathlib import Path
import json

src = '/home/mtoering/data/hmdb_videos/jpg'
output = '/home/mtoering/data/hmdb_videos/jpg_256'
ann = '/home/mtoering/data/hmdb51_1.json'
outlist = 'hmdb_new.txt'
foldername = ''

with open(ann) as annotation:
    d = json.loads(annotation.read())

#print(d["database"])


fout = open(outlist, 'w')

for classname in os.listdir(src):
    class_path = os.path.join(src, classname)
    class_path2 = os.path.join(output, classname)
    for video_folder in os.listdir(class_path):
        if video_folder in d["database"]:
            if d["database"][video_folder]["subset"] == "training" or d["database"][video_folder]["subset"] == "validation":

                video_path = os.path.join(class_path, video_folder)
                for filename in os.listdir(video_path):
                    all_str = os.path.join(video_path, filename) + '\n'
                    #fout.write(all_str)
                video_path2 = os.path.join(class_path2, video_folder)
                for filename2 in os.listdir(video_path):
                    all_str2 = os.path.join(video_path2, filename2) + '\n'
                    fout.write(all_str2)

                # video_path2 = os.path.join(class_path2, video_folder)
                # fname = video_path
                # path = os.path.join(video_path, 'n_frames')
                # file = open(path, 'r')
                # fnms = file.read()
                # file.close()
                # outstr = fname + ' ' + str(fnms) + '\n' # Output is e.g. videos/O/5/E/v_vAqaXZuAO5E/075 3
                # outstr2 = video_path2 + '\n'
                # fout.write(outstr2)
                # print("done")

fout.close()

