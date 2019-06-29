# Video Classification from scratch
## Thesis project

![](3D-ResNets-PyTorch-TimeCycle/master/figures/Multi-branch_network.png)



### Preprocess - Follow 3D-ResNets-PyTorch and change and run utils/generate_filelist.py


### Train ResNet 50 model on split 1 of HMDB-51

python3 main.py --timecycle_weight 25 --binary_class_weight 2 --annotation_path hmdb51_1.json --list hmdb_1.txt --result_path res50_bin_test --videoLen 3 --frame_gap 4 --predDistance 0 --gpu_id 0





_____________________________________________________________________________


Acknowledgements
 
3D ResNets for Action Recognition: https://github.com/kenshohara/3D-ResNets-PyTorch

TimeCycle: https://github.com/martinetoering/3D-ResNets-PyTorch-TimeCycle


