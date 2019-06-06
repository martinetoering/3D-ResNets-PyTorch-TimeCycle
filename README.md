Run experiment

```
python3 main.py --annotation_path hmdb51_1.json --list hmdb_1.txt \ 
&& python3 main.py --annotation_path hmdb51_2.json --list hmdb_2.txt \ 
&& python3 main.py --annotation_path hmdb51_3.json --list hmdb_3.txt
```

python3 main.py --timecycle_weight 50 --binary_class_weight 4 --annotation_path hmdb51_2.json --list hmdb_2.txt --result_path result --videoLen 3 --frame_gap 4 --predDistance 0 --val


python3 test_all.py <checkpoint_folder_path>




____________________


# 3D ResNets for Action Recognition

# TimeCycle


