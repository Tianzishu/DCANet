# Basic module
from tqdm                  import tqdm
from model.parse_args_test import parse_args
import scipy.io as scio

# Torch and visulization
from torchvision      import transforms
from torch.utils.data import DataLoader

# Metric, loss .etc
from model.utils  import *
from model.metric import *
from model.loss   import *
from model.load_param_data import load_dataset1, load_param, load_dataset_eva

# Model
from model.model_DCANet import  *
class Trainer(object):
    def __init__(self, args):

        # Initial
        self.args  = args
        self.ROC   = ROCMetric(1, args.ROC_thr)
        # self.PD_FA = PD_FA(1,255)
        self.PD_FA = PD_FA(1,10, args.crop_size)
        self.mIoU  = mIoU(1)
        self.save_prefix = '_'.join([args.model, args.dataset])
        nb_filter, num_blocks = load_param(args.channel_size, args.backbone)

        # Read image index from TXT
        if args.mode    == 'TXT':
            dataset_dir = args.root + '/' + args.dataset
            #train_img_ids, val_img_ids, test_txt=load_dataset_eva(args.root, args.dataset,args.split_method)
            val_img_ids, test_txt = load_dataset_eva(args.root, args.dataset, args.split_method)

        self.val_img_ids, _ = load_dataset1(args.root, args.dataset, args.split_method)

        # Preprocess and load data
        input_transform = transforms.Compose([
                          transforms.ToTensor(),
                          transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        testset         = TestSetLoader_size (dataset_dir,img_id=val_img_ids,base_size=args.base_size, crop_size=args.crop_size, transform=input_transform,suffix=args.suffix)
        self.test_data  = DataLoader(dataset=testset,  batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers,drop_last=False)

        # Choose and load model (this paper is finished by one GPU)
        if args.model == 'DCANet':
            model = DCANet(num_classes=1, input_channels=args.in_channels, block=Res_SimAM_block, num_blocks=num_blocks,
                           nb_filter=nb_filter, deep_supervision=args.deep_supervision)
        elif args.model == 'DCANet_CB':
            model = DCANet(num_classes=1, input_channels=args.in_channels, block=Res_CBAM_block, num_blocks=num_blocks,
                           nb_filter=nb_filter, deep_supervision=args.deep_supervision)
        elif args.model == 'DNANet':
            model = DNANet(num_classes=1, input_channels=args.in_channels, block=Res_CBAM_block, num_blocks=num_blocks,
                           nb_filter=nb_filter, deep_supervision=args.deep_supervision)
        elif args.model == 'ACM':
            model = ACM(args.in_channels, layers=[args.blocks] * 3, fuse_mode=args.fuse_mode, tiny=False, classes=1)
        elif args.model == 'ALCNet':
            model = ALCNet(layers=[4] * 4, channels=[8, 16, 32, 64, 128], shift=13, pyramid_mod='AsymBi',
                           scale_mode='Single',
                           act_dilation=16, fuse_mode='AsymBi', pyramid_fuse='Single', r=2, classes=1)
        elif args.model == 'ISNet':
            model = ISNet(layer_blocks=[4] * 3, channels=[8, 16, 32, 64], num_classes=1)
        elif args.model == 'ResUNet':
            model = res_UNet(num_classes=1, input_channels=3, block=Res_block, num_blocks=[2, 2, 2, 2],
                             nb_filter=[8, 16, 32, 64, 128])
        elif args.model == 'AGPCNet':
            model = agpcnet(backbone='resnet18', scales=(10, 6, 5, 3), reduce_ratios=(16, 4), gca_type='patch',
                            gca_att='post', drop=0.1)
        model.apply(weights_init_xavier)
        print("Model Initializing")
        self.model      = model

        # DATA_Evaluation metrics
        self.best_recall    = [0,0,0,0,0,0,0,0,0,0,0]
        self.best_precision = [0,0,0,0,0,0,0,0,0,0,0]

        # Checkpoint
        #checkpoint        = torch.load(args.root.split('dataset')[0] +args.model_dir)
        checkpoint = torch.load(args.model_dir)
        target_image_path = dataset_dir + '/' +'visulization_result' + '/' + args.st_model + '_visulization_result'
        target_dir        = dataset_dir + '/' +'visulization_result' + '/' + args.st_model + '_visulization_fuse'
        eval_image_path   = './result/'+ args.st_model +'/'+ 'visulization_result'
        eval_fuse_path    = './result/'+ args.st_model +'/'+ 'visulization_fuse'

        #make_visulization_dir(target_image_path, target_dir)
        make_visulization_dir(eval_image_path, eval_fuse_path)

        # Load trained model
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model = self.model.to('cuda')
        # Test
        self.model.eval()
        tbar = tqdm(self.test_data)
        losses = AverageMeter()
        with torch.no_grad():
            num = 0
            for i, ( data, labels, size) in enumerate(tbar):
                data = data.cuda()
                labels = labels.cuda()
                pred = self.model(data)
                loss = SoftIoULoss(pred, labels)
                #save_Ori_intensity_Pred_GT(pred, labels,target_image_path, val_img_ids, num, args.suffix,args.crop_size)
                save_resize_pred(pred, size, args.crop_size, eval_image_path, self.val_img_ids, num, args.suffix)
                #save_Pred_GT_for_split_evalution(pred, labels, eval_image_path, self.val_img_ids, num, args.suffix, args.crop_size)
                num += 1
                losses.    update(loss.item(), pred.size(0))
                self.ROC.  update(pred, labels)
                self.mIoU. update(pred, labels)
                self.PD_FA.update(pred, labels)
                _, mean_IOU = self.mIoU.get()

            FA, PD    = self.PD_FA.get(len(val_img_ids), args.crop_size)
            test_loss = losses.avg

            # scio.savemat(dataset_dir + '/' +  'value_result'+ '/' +args.st_model  + '_PD_FA_' + str(255),
            #              {'number_record1': FA, 'number_record2': PD})

            print('test_loss, %.4f' % (test_loss))
            print('mean_IOU:', mean_IOU)
            print('PD:',PD)
            print('FA:',FA)
            self.best_iou = mean_IOU

''
def main(args):
    trainer = Trainer(args)

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args = parse_args()
    main(args)
