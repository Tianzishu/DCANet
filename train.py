# torch and visulization
import torch
from tqdm             import tqdm
import  numpy as np
import torch.optim    as optim
from torch.optim      import lr_scheduler
from torchvision      import transforms
from torch.utils.data import DataLoader
from model.parse_args_train import  parse_args

# metric, loss .etc
from model.utils import *
from model.metric import *
from model.loss import *
from model.load_param_data import  load_dataset, load_param

# model
from model.model_DNANet import  *
from model.model_DCANet import  *
from model.model_ACM import  ACM
from model.model_alcnet import ASKCResNetFPN as ALCNet
from model.model_ISNet.ISNet import ISNet
from model.model_res_UNet import res_UNet
from model.model_res_UNet import Res_block
from model.model_AGPCNet import agpcnet


#from model.DANet import  Res_DA_block
#from model.PSA import  Res_PSA_block
#from model.epsanet_dca import EPSABlock
#from model.epsanet import EPSABlock
#from model.epsanet import EPSANet
#from model.pspnet import PSPNet
import warnings
warnings.filterwarnings('ignore')


#from torchsummary import summary

class Trainer(object):
    def __init__(self, args):
        # Initial
        self.args = args
        self.ROC  = ROCMetric(1, 10)
        self.mIoU = mIoU(1)
        self.save_prefix = '_'.join([args.model, args.dataset])
        self.save_dir    = args.save_dir
        nb_filter, num_blocks = load_param(args.channel_size, args.backbone)

        # Read image index from TXT
        if args.mode == 'TXT':

            dataset_dir = args.root + '/' + args.dataset
            train_img_ids, val_img_ids, test_txt = load_dataset(args.root, args.dataset, args.split_method)

        # Preprocess and load data
        input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        trainset        = TrainSetLoader(dataset_dir,img_id=train_img_ids,base_size=args.base_size,crop_size=args.crop_size,transform=input_transform,suffix=args.suffix)
        testset         = TestSetLoader (dataset_dir,img_id=val_img_ids,base_size=args.base_size, crop_size=args.crop_size, transform=input_transform,suffix=args.suffix)
        self.train_data = DataLoader(dataset=trainset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.workers,drop_last=True)
        self.test_data  = DataLoader(dataset=testset,  batch_size=args.test_batch_size, num_workers=args.workers,drop_last=False)

        # Choose and load model (this paper is finished by one GPU)
        if args.model   == 'DCANet':
            model       = DCANet(num_classes=1, input_channels=args.in_channels, block=Res_SimAM_block, num_blocks=num_blocks, nb_filter=nb_filter, deep_supervision=args.deep_supervision)
        elif args.model   == 'DCANet_CB':
            model       = DCANet(num_classes=1, input_channels=args.in_channels, block=Res_CBAM_block, num_blocks=num_blocks, nb_filter=nb_filter, deep_supervision=args.deep_supervision)
        elif args.model   == 'DNANet':
            model       = DNANet(num_classes=1, input_channels=args.in_channels, block=Res_CBAM_block, num_blocks=num_blocks, nb_filter=nb_filter, deep_supervision=args.deep_supervision)
        elif args.model == 'ACM':
            model = ACM(args.in_channels, layers=[args.blocks] * 3, fuse_mode=args.fuse_mode, tiny=False, classes=1)
        elif args.model == 'ALCNet':
            model = ALCNet(layers=[4] * 4, channels=[8, 16, 32, 64, 128], shift=13, pyramid_mod='AsymBi',
                         scale_mode='Single',
                         act_dilation=16, fuse_mode='AsymBi', pyramid_fuse='Single', r=2, classes=1)
        elif args.model == 'ISNet':
            model = ISNet(layer_blocks=[4]*3, channels=[8,16,32,64], num_classes=1)
        elif args.model == 'ResUNet':
            model = res_UNet(num_classes=1, input_channels=3, block=Res_block, num_blocks=[2, 2, 2, 2],
                           nb_filter=[8, 16, 32, 64, 128])
        elif args.model == 'AGPCNet':
            model = agpcnet(backbone='resnet18', scales=(10, 6, 5, 3), reduce_ratios=(16, 4), gca_type='patch', gca_att='post', drop=0.1)
        #device = torch.device("cuda:0")
        model           = model.cuda()
      #  summary(model, (args.in_channels, args.base_size, args.crop_size))
        model.apply(weights_init_xavier)
        print("Model Initializing", args.model,args.dataset)
        self.model      = model

#        summary(self.model, (args.in_channels, args.base_size, args.crop_size))

        # Optimizer and lr scheduling

        if args.optimizer   == 'Adam':
            self.optimizer  = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        elif args.optimizer == 'Adagrad':
            self.optimizer  = torch.optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        if args.scheduler   == 'CosineAnnealingLR':
            self.scheduler  = lr_scheduler.CosineAnnealingLR( self.optimizer, T_max=args.epochs, eta_min=args.min_lr)
        self.scheduler.step()

        # Evaluation metrics
        self.best_iou       = 0
        self.best_recall    = [0,0,0,0,0,0,0,0,0,0,0]
        self.best_precision = [0,0,0,0,0,0,0,0,0,0,0]


    # Training
    def training(self,epoch):
        tbar = tqdm(self.train_data)
        self.model.train()
        losses = AverageMeter()
        for i, ( data, labels) in enumerate(tbar):
            self.get_grad(i=i,epoch=epoch)
            data   = data.cuda()
            labels = labels.cuda()
            if args.deep_supervision == 'True':
                preds= self.model(data)
                loss = 0
                for pred in preds:
                    loss += SoftIoULoss(pred, labels)
                loss /= len(preds)
            else:
               pred = self.model(data)
               loss = SoftIoULoss(pred, labels)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.update(loss.item(), pred.size(0))
            tbar.set_description('Epoch %d, training loss %.4f' % (epoch, losses.avg))
        self.train_loss = losses.avg

    # Testing
    def testing (self, epoch):
        tbar = tqdm(self.test_data)
        self.model.eval()
        self.mIoU.reset()
        losses = AverageMeter()

        with torch.no_grad():
            for i, ( data, labels) in enumerate(tbar):
                data = data.cuda()
                labels = labels.cuda()
                if args.deep_supervision == 'True':
                    preds = self.model(data)
                    loss = 0
                    for pred in preds:
                        loss += SoftIoULoss(pred, labels)
                    loss /= len(preds)
                    pred =preds[-1]
                else:
                    pred = self.model(data)
                    loss = SoftIoULoss(pred, labels)
                losses.update(loss.item(), pred.size(0))
                self.ROC .update(pred, labels)
                self.mIoU.update(pred, labels)
                ture_positive_rate, false_positive_rate, recall, precision = self.ROC.get()
                _, mean_IOU = self.mIoU.get()
                tbar.set_description('Epoch %d, test loss %.4f, mean_IoU: %.4f' % (epoch, losses.avg, mean_IOU ))
            test_loss=losses.avg
        save_model(mean_IOU, self.best_iou, self.save_dir, self.save_prefix,
                   self.train_loss, test_loss, recall, precision, epoch, self.model.state_dict(),self.grad_block)


def main(args):
    trainer = Trainer(args)
    for epoch in range(args.start_epoch, args.epochs):
        trainer.training(epoch)
        trainer.testing(epoch)


if __name__ == "__main__":
    args = parse_args()
    main(args)





