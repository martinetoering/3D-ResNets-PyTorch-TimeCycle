import argparse
import fnmatch
import glob
import json
import os
import shutil
import subprocess
import uuid

from joblib import delayed
from joblib import Parallel
import pandas as pd

file_src = '/home/mtoering/data/hmdb_all.txt'
file_output = '/home/mtoering/data/hmdb_new.txt'
folder_path = '/home/mtoering/data/hmdb_videos/jpg'
output_path = '/home/mtoering/data/hmdb_videos/jpg_256'


file_list = []

f = open(file_src, 'r')

for line in f:
    rows = line.split()
    fname = rows[0]
    file_list.append(fname)

f.close()

output_list = []

f = open(file_output, 'r')

for line in f:
    rows = line.split()
    fname = rows[0]
    output_list.append(fname)

f.close()


def download_clip(inname, outname):

    status = False
    inname = '"%s"' % inname
    outname = '"%s"' % outname
    command = "ffmpeg  -loglevel panic -i {} -filter:v scale=\"trunc(oh*a/2)*2:256\" -q:v 1 -c:a copy {}".format( inname, outname)
    try:
        output = subprocess.check_output(command, shell=True,
                                         stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as err:
        return status, err.output

    # Check if the video was successfully saved.
    status = os.path.exists(outname)
    return status, 'Downloaded'


def download_clip_wrapper(row):
    """Wrapper for parallel processing purposes."""

    videoname = row
    videoname2 = row2
    
    for filename in os.listdir(videoname):

        inname = filename
        outname = filename

    if os.path.isdir(outname) is False:
        try:
            os.makedirs( outname, 0755 )
        except:
            print(outname)

    outname = outname + '/clip.mp4'


    downloaded, log = download_clip(inname, outname)
    return downloaded


status_lst = Parallel(n_jobs=15)(delayed(download_clip_wrapper)(row) for row, row2 in zip(file_list, output_list)
