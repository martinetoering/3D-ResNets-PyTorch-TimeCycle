import argparse

def str_to_bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--loss_weight',
        default=75,
        type=int,
        help='Weight to multiply TimeCycle loss with before adding to classification loss')

    parser.add_argument(
        '--root_path',
        default='/home/martine/data',
        type=str,
        help='Root directory path of data')
    parser.add_argument(
        '--video_path',
        default='hmdb_videos/jpg',
        type=str,
        help='Directory path of Videos')
    parser.add_argument( 
        '--list', 
        default='hmdb_1.txt',
        help='path to video list', 
        type=str)
    parser.add_argument(
        '--annotation_path',
        default='hmdb51_1.json',
        type=str,
        help='Annotation file path')
    parser.add_argument(
        '--result_path',
        default='test',
        type=str,
        help='Result directory path')

    parser.add_argument(
        '--batch_size', 
        default=4, 
        type=int, 
        help='Batch Size')
    parser.add_argument(
        '--n_epochs',
        default=100,
        type=int,
        help='Number of total epochs to run')
    parser.add_argument(
        '--begin_epoch',
        default=1,
        type=int,
        help=
        'Training begins at this epoch. Previous trained model indicated by resume_path is loaded.'
    )

    parser.add_argument(
        '--n_val_samples',
        default=3,
        type=int,
        help='Number of validation samples for each activity')
    parser.add_argument(
        '--resume_path',
        default='',
        type=str,
        help='Save data (.pth) of previous training')
    parser.add_argument(
        '--pretrain_path', 
        default='', 
        type=str, 
        help='Pretrained model (.pth)')
    parser.add_argument(
        '--ft_begin_index',
        default=0,
        type=int,
        help='Begin block index of fine-tuning')

    parser.add_argument(
        '--learning_rate',
        default=2e-4,
        type=float,
        help=
        'Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument(
        '--momentum', 
        default=0.5, 
        type=float, 
        metavar='M',
        help='momentum')
    parser.add_argument(
        '--weight_decay',
        default=1e-4, 
        type=float,
        metavar='W', 
        help='weight decay (default: 1e-4)')
    parser.add_argument(
        '--lr_patience',
        default=10,
        type=int,
        help='Patience of LR scheduler. See documentation of ReduceLROnPlateau.'
    )

    parser.add_argument(
        '--videoLen', 
        default=3, 
        type=int,
        help='')
    parser.add_argument(
        '--frame_gap', 
        default=4, 
        type=int,
        help='')
    parser.add_argument(
        '--hist', 
        default=1, 
        type=int,
        help='')
    parser.add_argument(
        '--sample_size',
        default=240,
        type=int,
        help='Height and width of inputs')
    parser.add_argument(
        '--sample_duration',
        default=13,
        type=int,
        help='Temporal duration of inputs')
    parser.add_argument(
        '--predDistance', 
        default=0, type=int,
        help='predict how many frames away')
    parser.add_argument(
        '--seperate2d', 
        type=int, 
        default=0, 
        help='manual seed')
    parser.add_argument(
        '--T', 
        default=512**-.5, 
        type=float,
        help='temperature')
    parser.add_argument(
        '--gridSize', 
        default=9, 
        type=int,
        help='temperature')
    parser.add_argument(
        '--lamda', 
        default=0.1, 
        type=float,
        help='temperature')
    parser.add_argument(
        '--pretrained_imagenet', 
        type=str_to_bool, 
        nargs='?', 
        const=True, 
        default=False,
        help='pretrained_imagenet')


    # Miscs
    parser.add_argument(
        '--manualSeed', type=int, help='manual seed')

    #Device options
    parser.add_argument(
        '--gpu-id', 
        default='0', 
        type=str,
        help='id(s) for CUDA_VISIBLE_DEVICES')

    # 3D-ResNets-PyTorch
    parser.add_argument(
        '--dataset',
        default='hmdb51',
        type=str,
        help='Used dataset (activitynet | kinetics | ucf101 | hmdb51)')
    parser.add_argument(
        '--n_classes',
        default=51,
        type=int,
        help=
        'Number of classes (activitynet: 200, kinetics: 400, ucf101: 101, hmdb51: 51)'
    )
    parser.add_argument(
        '--n_finetune_classes',
        default=51,
        type=int,
        help=
        'Number of classes for fine-tuning. n_classes is set to the number when pretraining.'
    )
    # parser.add_argument(
    #     '--initial_scale',
    #     default=1.0,
    #     type=float,
    #     help='Initial scale for multiscale cropping')
    # parser.add_argument(
    #     '--n_scales',
    #     default=5,
    #     type=int,
    #     help='Number of scales for multiscale cropping')
    # parser.add_argument(
    #     '--scale_step',
    #     default=0.84089641525,
    #     type=float,
    #     help='Scale step for multiscale cropping')
    # parser.add_argument(
    #     '--train_crop',
    #     default='corner',
    #     type=str,
    #     help=
    #     'Spatial cropping method in training. random is uniform. corner is selection from 4 corners and 1 center.  (random | corner | center)'
    # )
    # parser.add_argument('--momentum', default=0.9, type=float, help='Momentum')
    # parser.add_argument(
    #     '--dampening', 
    #     default=0.9, 
    #     type=float, 
    #     help='dampening of SGD')
    # parser.add_argument(
    #     '--weight_decay', default=1e-3, type=float, help='Weight Decay')
    # parser.add_argument(
    #     '--mean_dataset',
    #     default='activitynet',
    #     type=str,
    #     help=
    #     'dataset for mean values of mean subtraction (activitynet | kinetics)')
    # parser.add_argument(
    #     '--mean_norm',
    #     action='store_true',
    #     help='If true, inputs are normalized by mean.')
    # parser.set_defaults(mean_norm=False)
    # parser.add_argument(
    #     '--std_norm',
    #     action='store_true',
    #     help='If true, inputs are normalized by standard deviation.')
    # parser.set_defaults(std_norm=False)
    # parser.add_argument(
    #     '--nesterov', action='store_true', help='Nesterov momentum')
    # parser.set_defaults(nesterov=False)

    parser.add_argument(
        '--no_train',
        action='store_true',
        help='If true, training is not performed.')
    parser.set_defaults(no_train=False)
    parser.add_argument(
        '--test_eval_all',
        action='store_true',
        help='If true, all epochs are tested and evaluated in validation step')
    parser.set_defaults(test_eval_all=False)
    parser.add_argument(
        '--val',
        action='store_true',
        help='If true, validation step with n_val_samples is performed (no testing or evaluating)')
    parser.set_defaults(val=False)
    parser.add_argument(
        '--no_test', action='store_true', help='If true, no test is performed')
    parser.set_defaults(no_test=False)
    parser.add_argument(
        '--test_subset',
        default='val',
        type=str,
        help='Used subset in test (val | test)')
    parser.add_argument(
        '--no_eval', action='store_true', help='If true, no evaluation is done.')
    parser.set_defaults(no_eval=False)
    parser.add_argument(
        '--top_k', default=1, type=int, help='Top 1 or Top 5 accuracy')
    parser.add_argument(
        '--print_per_epoch', action='store_true')
    parser.set_defaults(print_per_epoch=False)
    # parser.add_argument(
    #     '--scale_in_test',
    #     default=1.0,
    #     type=float,
    #     help='Spatial scale in test')
    # parser.add_argument(
    #     '--crop_position_in_test',
    #     default='c',
    #     type=str,
    #     help='Cropping method (c | tl | tr | bl | br) in test')
    parser.add_argument(
        '--no_softmax_in_test',
        action='store_true',
        help='If true, output for each clip is not normalized using softmax.')
    parser.set_defaults(no_softmax_in_test=False)
    parser.add_argument(
        '--no_cuda', action='store_true', help='If true, cuda is not used.')
    parser.set_defaults(no_cuda=False)
    parser.add_argument(
        '--n_threads',
        default=12,
        type=int,
        help='Number of threads for multi-thread loading')
    parser.add_argument(
        '--checkpoint',
        default=2,
        type=int,
        help='Trained model is saved at every this epochs.')
    # parser.add_argument(
    #     '--no_hflip',
    #     action='store_true',
    #     help='If true holizontal flipping is not performed.')
    # parser.set_defaults(no_hflip=False)
    # parser.add_argument(
    #     '--norm_value',
    #     default=1,
    #     type=int,
    #     help=
    #     'If 1, range of inputs is [0-255]. If 255, range of inputs is [0-1].')
    parser.add_argument(
        '--model',
        default='resnet',
        type=str,
        help='(resnet | preresnet | wideresnet | resnext | densenet | ')
    parser.add_argument(
        '--model_depth',
        default=50,
        type=int,
        help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    # parser.add_argument(
    #     '--resnet_shortcut',
    #     default='B',
    #     type=str,
    #     help='Shortcut type of resnet (A | B)')
    # parser.add_argument(
    #     '--wide_resnet_k', default=2, type=int, help='Wide resnet k')
    # parser.add_argument(
    #     '--resnext_cardinality',
    #     default=32,
    #     type=int,
    #     help='ResNeXt cardinality')
    # parser.add_argument(
    #     '--manual_seed', default=1, type=int, help='Manually set random seed')
    parser.add_argument(
        '--name_general_eval_file',
        action='store_true')
    args = parser.parse_args()

    return args
