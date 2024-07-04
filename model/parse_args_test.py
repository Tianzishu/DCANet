from model.utils import *

def parse_args():
    """Training Options for Segmentation Experiments"""
    parser = argparse.ArgumentParser(description='Dense_Nested_Attention_Network_For_SIRST')
    # choose model
    parser.add_argument('--model', type=str, default='DCANet',
                        help='model name:  UNet')
    parser.add_argument('--channel_size', type=str, default='two',
                        help='one,  two,  three,  four')
    parser.add_argument('--backbone', type=str, default='resnet_34',
                        help='vgg10, resnet_10,  resnet_18,  resnet_34 ')

    # data and pre-process
    parser.add_argument('--dataset', type=str, default='NUDT-SIRST',
                        help='dataset name: NUDT-SIRST,NUAA-SIRST,ISTDD,IRSTD-1K')
    parser.add_argument('--st_model', type=str, default='NUDT-SIRST_DNANet_20_06_2024_13_52_33S_wDS')
    parser.add_argument('--model_dir', type=str,
                        default = './result/NUDT-SIRST_DNANet_20_06_2024_13_52_33S_wDS/mIoU__DCANet_NUDT-SIRST_epoch.pth.tar')
    parser.add_argument('--mode', type=str, default='TXT', help='mode name:  TXT, Ratio')
    parser.add_argument('--test_size', type=float, default='0.5', help='when --mode==Ratio')
    parser.add_argument('--root', type=str, default='./dataset')
    parser.add_argument('--suffix', type=str, default='.png')
    parser.add_argument('--split_method', type=str, default='50_50',
                        help='50_50(for NUDT-SIRST NUAA-SIRST),800_200(for IRSTD-1K),8000_500(for ISTDD)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--in_channels', type=int, default=3,
                        help='in_channel=3 for pre-process')
    parser.add_argument('--base_size', type=int, default=256,
                        help='256, 512, 1024')
    parser.add_argument('--crop_size', type=int, default=256,
                        help='256, 512, 1024')


    parser.add_argument('--test_batch_size', type=int, default=1,
                        metavar='N', help='input batch size for \
                        testing (default: 32)')

    # cuda and logging
    parser.add_argument('--gpus', type=str, default='0',
                        help='Training with GPUs, you can specify 1,3 for example.')

    # ROC threshold
    parser.add_argument('--ROC_thr', type=int, default=10,
                        help='crop image size')


    args = parser.parse_args()

    # the parser
    return args
