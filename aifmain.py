import warnings
warnings.filterwarnings("ignore")
from glob import glob
from sklearn.model_selection import GroupKFold, StratifiedKFold
import cv2
# from skimage import io
import torch
from torch import nn
import os
from datetime import datetime
import time
import random
#import cv2
import torchvision
from torchvision import transforms
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
import copy 

import matplotlib.pyplot as plt
from torch.utils.data import Dataset,DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
# from torch.cuda.amp import autocast, GradScaler
from torch.nn.modules.loss import _WeightedLoss
import torch.nn.functional as F

import timm
import time 

import sklearn
import warnings
import joblib
from sklearn.metrics import roc_auc_score, log_loss
from sklearn import metrics
#import warnings
#import cv2
#import pydicom
from efficientnet_pytorch import EfficientNet
from scipy.ndimage.interpolation import zoom

from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)

from albumentations.pytorch import ToTensorV2

from fmix import sample_mask, make_low_freq_image, binarise_mask

from pathlib import Path

def preprocess_data(): 
    ###### fmix package ready ######### 
    package_paths = []
    path_fMix = './FMix/'
    package_paths.append(path_fMix)
    
    for pth in package_paths:
    sys.path.append(pth)
    
    #### data preparation check ##### 
    path          = './cassava/'
    trainfilename = 'train.csv'

    from pathlib import Path
    TRAIN_PATH = Path('./cassava/train')
    
    for dir_path in TRAIN_PATH.glob('*'):
        print(dir_path)
        print(len(list(dir_path.glob('*.jpg'))))

    ######## train extra data preparation  ############# 
    map_dict = {'cbb': 0, 'cbsd': 1, 'cgm': 2, 'cmd': 3, 'healthy': 4}

    labels = []
    image_ids = []
    for dir_path in TRAIN_PATH.glob('*'):
        for img_path in dir_path.glob('*.jpg'):
    #         print(img_path)
    #         print(img_path.name)
    #         print(img_path.parent.name)
            label = map_dict[img_path.parent.name]
            labels.append(label)
            concatstr = '{}{}{}{}'.format('train/',img_path.parent.name, "/", img_path.name)  
            #image_ids.append(dir_path,img_path.name)
            image_ids.append(concatstr)  
    img2label = {'image_id': image_ids, 'label': labels}
    train_extra = pd.DataFrame.from_dict(img2label)

    ########### orginal train data preparation  ######### 
    path_train = path + trainfilename
    train = pd.read_csv(path_train, sep=',')
    
    ############### put path + file name and merge with extra data ###########
    train['image_id'] = 'train_images/' + train['image_id']
    train_merge = pd.concat([train, train_extra])
    
    ########## reset index for merged dataset ################## 
    train_merge.reset_index(drop=True, inplace=True)
    
    ##################### constant values ############## 
    CFG = {
    'fold_num': 5,
    'seed': 777,
    'model_arch': 'tf_efficientnet_b6_ns',
    #'img_size': 512,    #original one 
    #'wimg_size': 800,    #changed by jw
    #'himg_size': 600,    #changed by jw
    'img_size':  512,    #changed by jw
    'num_classes': 5,
    'epochs': 1,
    'train_bs': 12,
    'valid_bs': 12,
    'test_bs': 12,
    'T_0': 10,
    'lr': 1e-4,
    'min_lr': 1e-6,
    'weight_decay':1e-6,
    'num_workers': 4,
    'accum_iter': 2, # suppoprt to do batch accumulation for backprop with effectively larger batch size
    'verbose_step': 1,
    #'device': 'cuda:0',
    'patience': 20,     # check patience by jwp 
    'alpha' : 0.25 ,   # Focal Loss
    'gamma' : 1.5 ,     # Focal Loss 
    'fmix_flag' : 'fmix' , 
    'cutmix_flag' : 'cutmix' , 
    'mixup_flag' : 'mixup'  
    
    }

    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    
# get_img and get_img_rgb is the same, which one looks readable ? 


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
# get_img and get_img_rgb is the same, which one looks readable ? 

def get_img(path):
    im_bgr = cv2.imread(path)
    im_rgb = im_bgr[:, :, ::-1]
    #im_rgb = cv2.resize (im_rgb, (CFG['img_size'],CFG['img_size'])) 
    im_rgb = cv2.resize (im_rgb, (CFG['img_size'],CFG['img_size'])) 
    #im_rgb = cv2.resize (im_rgb, (CFG['img_size'],800)) 
    #im_rgb = cv2.resize (im_rgb, (800,CFG['img_size'])) 
    #print(im_rgb)
    return im_rgb


def get_img_rgb(path):
    im_bgr = cv2.imread(path)
    im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
    # im_rgb = cv2.resize (im_rgb, (512,512)) 
    #print(im_rgb)
    return im_rgb

class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in 
        Focal Loss for Dense Object Detection.
            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])
        The losses are averaged across observations for each minibatch.
        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5), 
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.
    """
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = torch.ones(class_num, 1)*alpha
            else:
                # self.alpha = Variable(alpha).cuda()
                self.alpha = Variable(torch.ones(class_num, 1)*alpha).cuda()
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()

        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p 

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss
    
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print, flag='default'):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.flag = flag
    def __call__(self, val_loss, model, fold, epoch,flag):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, fold, epoch,flag)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, fold, epoch, flag)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, fold, epoch, flag):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  fold: {fold} Saving model ...')
        
        torch.save(model.state_dict(),'model/{}_epoch{}_fold{}_{}'.format(CFG['model_arch'], epoch, fold,flag))
        self.val_loss_min = val_loss
        
def rand_bbox(size, lam):
    W = size[0]
    H = size[1]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


class PlantDataset(Dataset):
    def __init__(self, df, data_root, 
                 transforms=None, 
                 output_label=True, 
                 one_hot_label=False,
                 do_fmix=False, 
                 fmix_params={
                     'alpha': 1., 
                     'decay_power': 3., 
                     'shape': (CFG['img_size'], CFG['img_size']),
                     'max_soft': True, 
                     'reformulate': False
                 },
                 do_cutmix=False,
                 cutmix_params={
                     'alpha': 1,
                 },
                 do_mixup=False,
                 mixup_params={
                     'alpha': 1,
                 }
                ):
        
        super().__init__()
        self.df = df.reset_index(drop=True).copy()
        self.transforms    = transforms
        self.data_root     = data_root
        self.do_fmix       = do_fmix
        self.fmix_params   = fmix_params
        self.do_cutmix     = do_cutmix
        self.cutmix_params = cutmix_params
        self.do_mixup      = do_cutmix
        self.mixup_params  = mixup_params
        
        self.num_classes = CFG['num_classes']
        
        self.output_label  = output_label
        self.one_hot_label = one_hot_label
        
        
        if output_label == True:
            self.labels = self.df['label'].values
            #print(self.labels)
            
            if one_hot_label is True:
                self.labels = np.eye(self.num_classes)[self.labels]
                #print(self.labels)
            
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index: int):
        
        # get labels
        if self.output_label:
            target    = self.labels[index]
            
          
        img  = get_img("{}/{}".format(self.data_root, self.df.loc[index]['image_id']))
        
        if self.transforms:
            img = self.transforms(image=img)['image']
        
        if self.do_fmix and np.random.uniform(0., 1., size=1)[0] > 0.5:
            with torch.no_grad():
                #lam, mask = sample_mask(**self.fmix_params)
                #print('debug :::::::::::::::::::fmix started ::::::::::::::')
                
                fmix_ix = np.random.choice(self.df.index, size=1)[0]
                
                #if target == self.labels[fmix_ix]:
                
                fmix_img  = get_img("{}/{}".format(self.data_root, self.df.iloc[fmix_ix]['image_id']))
                lam, mask = sample_mask(self.fmix_params['alpha'], self.fmix_params['decay_power'], self.fmix_params['shape'], 
                                        self.fmix_params['max_soft'])

                if self.transforms:
                    fmix_img = self.transforms(image=fmix_img)['image']

                mask_torch = torch.from_numpy(mask)

                # fmix image
                img = mask_torch*img+(1.- mask_torch)*fmix_img

                #print('#############debug#############: img label: {} and fmix label: {} '.format(target, self.labels[fmix_ix]))  

                # mix target
                rate   = mask.sum()/CFG['img_size']/CFG['img_size']
                target = rate * target + (1.-rate)*self.labels[fmix_ix]
                #print(target, mask, img)
                #assert False

        if self.do_cutmix and np.random.uniform(0., 1., size=1)[0] > 0.5:
            #print(img.sum(), img.shape)
            with torch.no_grad():
                #print('debug :::::::::::::::::::cutmix started ::::::::::::::',img)
                cmix_ix = np.random.choice(self.df.index, size=1)[0]
                
                #if target == self.labels[cmix_ix]:
                    
                cmix_img  = get_img("{}/{}".format(self.data_root, self.df.iloc[cmix_ix]['image_id']))
                if self.transforms:
                    cmix_img = self.transforms(image=cmix_img)['image']

                lam = np.clip(np.random.beta(self.cutmix_params['alpha'], self.cutmix_params['alpha']),0.3,0.4)
                bbx1, bby1, bbx2, bby2 = rand_bbox((CFG['img_size'], CFG['img_size']), lam)

                img[:, bbx1:bbx2, bby1:bby2] = cmix_img[:, bbx1:bbx2, bby1:bby2]
                #print('#############debug#############: img label: {} and cmix label: {} '.format(target, self.labels[cmix_ix]))  

                rate = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (CFG['img_size'] * CFG['img_size']))
                target = rate*target + (1.-rate)*self.labels[cmix_ix]
        
        if self.do_mixup and np.random.uniform(0., 1., size=1)[0] > 0.5:
            #print(img.sum(), img.shape)
            with torch.no_grad():
                #print('debug :::::::::::::::::::mixup started ::::::::::::::')
                mixup_ix = np.random.choice(self.df.index, size=1)[0]
                
                #if target == self.labels[mixup_ix]:
                mixup_img  = get_img("{}/{}".format(self.data_root, self.df.iloc[mixup_ix]['image_id']))

                if self.transforms:
                    mixup_img = self.transforms(image=mixup_img)['image']

                lam = np.clip(np.random.beta(self.mixup_params['alpha'], self.mixup_params['alpha']),0.3,0.7)

                img = (lam*img) + ((1-lam)*mixup_img)
                #print('#############debug#############: img label: {} and mixup label: {} '.format(target, self.labels[mixup_ix]))  
                rate   = lam
                target = rate*target + (1.-rate)*self.labels[mixup_ix]
                
                #print('debug target mixup and original label value : {}, {}'.format(target, self.labels[mixup_ix]))
                
                
            #print('-', img.sum())
            #print(target)
            #assert False
                            
        # do label smoothing
        #print(type(img), type(target))
        if self.output_label == True:
            return img, target
        else:
            return img

def get_train_transforms():
    return Compose([
            RandomResizedCrop(CFG['img_size'], CFG['img_size']),
            Transpose(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ShiftScaleRotate(p=0.5),
            HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            CoarseDropout(p=0.5),
            Cutout(p=0.5),
            ToTensorV2(p=1.0),
        ], p=1.)
  
        
def get_valid_transforms():
    return Compose([
            CenterCrop(CFG['img_size'], CFG['img_size'], p=1.),
            Resize(CFG['img_size'], CFG['img_size']),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], max_pixel_value=255.0, p=1.0),
            ToTensorV2(p=1.0),
        ], p=1.)


class PlantImgClassifier(nn.Module):
    def __init__(self, model_arch, n_class, pretrained=False):
        super().__init__()
        self.model = timm.create_model(model_arch, pretrained=pretrained)
        n_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(n_features, n_class)

    def forward(self, x):
        x = self.model(x)
        return x
    
def prepare_dataloader(df, trn_idx, val_idx, data_root, fmix_flag, cutmix_flag, mixup_flag):
    
#    from catalyst.data.sampler import BalanceClassSampler
    
    train_ = df.loc[trn_idx,:].reset_index(drop=True)
    valid_ = df.loc[val_idx,:].reset_index(drop=True)
        
    train_ds = PlantDataset(train_, data_root, transforms=get_train_transforms(), output_label=True, 
                            one_hot_label=False, do_fmix=fmix_flag, do_cutmix=cutmix_flag, do_mixup=mixup_flag)
    valid_ds = PlantDataset(valid_, data_root, transforms=get_valid_transforms(), output_label=True)
    
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=CFG['train_bs'],
        pin_memory=False,
        drop_last=False,
        shuffle=True,        
        num_workers=CFG['num_workers'],
        #sampler=BalanceClassSampler(labels=train_['label'].values, mode="downsampling")
    )
    val_loader = torch.utils.data.DataLoader(
        valid_ds, 
        batch_size=CFG['valid_bs'],
        num_workers=CFG['num_workers'],
        shuffle=False,
        pin_memory=False,
    )
    return train_loader, val_loader

def train_one_epoch(epoch, model, loss_fn, optimizer, train_loader, device, scheduler=None, schd_batch_update=False):
    model.train()

    t = time.time()
    running_loss = None

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    for step, (imgs, image_labels) in pbar:
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).long()
        image_preds = model(imgs)   #output = model(input)
        loss = loss_fn(image_preds, image_labels)
        loss.backward()

        if running_loss is None:
            running_loss = loss.item()
        else:
            running_loss = running_loss * .99 + loss.item() * .01

        if ((step + 1) %  CFG['accum_iter'] == 0) or ((step + 1) == len(train_loader)):
            optimizer.step()
            optimizer.zero_grad() 

            if scheduler is not None and schd_batch_update:
                scheduler.step()

        if ((step + 1) % CFG['verbose_step'] == 0) or ((step + 1) == len(train_loader)):
            description = f'epoch {epoch} loss: {running_loss:.4f}'

            pbar.set_description(description)
                
    if scheduler is not None and not schd_batch_update:
        scheduler.step()
    
    return running_loss 
    
def valid_one_epoch(epoch, model, loss_fn, val_loader, device, scheduler=None, schd_loss_update=False):
    model.eval()

    t = time.time()
    loss_sum = 0
    sample_num = 0
    image_preds_all = []
    image_targets_all = []
    
    pbar = tqdm(enumerate(val_loader), total=len(val_loader))
    for step, (imgs, image_labels) in pbar:
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).long()
        
        image_preds = model(imgs)   #output = model(input)
        image_preds_all += [torch.argmax(image_preds, 1).detach().cpu().numpy()]
        image_targets_all += [image_labels.detach().cpu().numpy()]
        
        loss = loss_fn(image_preds, image_labels)
      
        loss_sum += loss.item()*image_labels.shape[0]
        sample_num += image_labels.shape[0]  

        if ((step + 1) % CFG['verbose_step'] == 0) or ((step + 1) == len(val_loader)):
            description = f'epoch {epoch} loss: {loss_sum/sample_num:.4f}'
            pbar.set_description(description)
    
    image_preds_all = np.concatenate(image_preds_all)
    image_targets_all = np.concatenate(image_targets_all)
    print('validation multi-class accuracy = {:.4f}'.format((image_preds_all==image_targets_all).mean()))
    
    if scheduler is not None:
        if schd_loss_update:
            scheduler.step(loss_sum/sample_num)
        else:
            scheduler.step()

    return loss_sum/sample_num  



if __name__ == '__main__':
    
    preprocess_data()
    
    starttime = time.time()
    print('Planned epochs : {} Start Time {}'.format(CFG['epochs'], time.strftime("%H:%M:%S", time.gmtime(starttime)))) 
     # for training only, need nightly build pytorch
    
    early_stopping = EarlyStopping(patience=CFG['patience'], verbose=True)
     
    seed_everything(CFG['seed'])
    

    #device = torch.device(CFG['device'])
        
    model = PlantImgClassifier(CFG['model_arch'], CFG['num_classes'], pretrained=True).to(device)
    #model.load_state_dict(torch.load('./model/{}_epoch{}_fold{}'.format(CFG['model_arch'],0, 1)))
    
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG['lr'], weight_decay=CFG['weight_decay'])
    
    
    #loss_tr = FocalLoss(CFG['num_classes'], alpha=CFG['alpha'], gamma=CFG['gamma'], size_average=True)  
    #loss_fn = FocalLoss(CFG['num_classes'], alpha=CFG['alpha'], gamma=CFG['gamma'], size_average=True)  
    loss_tr = nn.CrossEntropyLoss().to(device) #MyCrossEntropyLoss().to(device)
    loss_fn = nn.CrossEntropyLoss().to(device)

    for epoch in range(CFG['epochs']):
            
        folds = StratifiedKFold(n_splits=CFG['fold_num'], shuffle=True, random_state=CFG['seed']).split(np.arange(train.shape[0]),
                                                                                                        train.label.values)
    
        for fold, (trn_idx, val_idx) in enumerate(folds):

            ############# fmix training ###########
            print('Training epoch {} with fold {} fmix started'.format(epoch, fold))

            print(len(trn_idx), len(val_idx))
            train_loader, val_loader = prepare_dataloader(train, trn_idx, val_idx, data_root=path , 
                                                          fmix_flag=True, cutmix_flag=False, mixup_flag=False)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=25, 
                                                   max_lr=CFG['lr'], epochs=CFG['epochs'], steps_per_epoch=len(train_loader))

            tr_loss = train_one_epoch(epoch, model, loss_tr, optimizer, train_loader, device, scheduler=scheduler, schd_batch_update=False)

            with torch.no_grad():
                val_loss = valid_one_epoch(epoch, model, loss_fn, val_loader, device, scheduler=None, schd_loss_update=False)
                #torch.save(model.state_dict(),'model/{}_fold_{}_{}'.format(CFG['model_arch'], fold, epoch))

                early_stopping(val_loss, model, fold, epoch,CFG['fmix_flag'])

                if early_stopping.early_stop:
                    print("Early stopping")
                    break 

            ############# cutmix training ###########
            print('Training epoch {} with fold {} cutmix started'.format(epoch, fold))

            print(len(trn_idx), len(val_idx))
            train_loader, val_loader = prepare_dataloader(train, trn_idx, val_idx, data_root=path , 
                                                          fmix_flag=False, cutmix_flag=True, mixup_flag=False)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=25, 
                                                   max_lr=CFG['lr'], epochs=CFG['epochs'], steps_per_epoch=len(train_loader))

            tr_loss = train_one_epoch(epoch, model, loss_tr, optimizer, train_loader, device, scheduler=scheduler, schd_batch_update=False)

            with torch.no_grad():
                val_loss = valid_one_epoch(epoch, model, loss_fn, val_loader, device, scheduler=None, schd_loss_update=False)
                #torch.save(model.state_dict(),'model/{}_fold_{}_{}'.format(CFG['model_arch'], fold, epoch))

                early_stopping(val_loss, model, fold, epoch,CFG['cutmix_flag'])

                if early_stopping.early_stop:
                    print("Early stopping")
                    break 

            ############# mixup training ###########
            print('Training epoch {} with fold {} mixup started'.format(epoch, fold))

            print(len(trn_idx), len(val_idx))
            train_loader, val_loader = prepare_dataloader(train, trn_idx, val_idx, data_root=path , 
                                                          fmix_flag=False, cutmix_flag=False, mixup_flag=True)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=25, 
                                                   max_lr=CFG['lr'], epochs=CFG['epochs'], steps_per_epoch=len(train_loader))

            tr_loss = train_one_epoch(epoch, model, loss_tr, optimizer, train_loader, device, scheduler=scheduler, schd_batch_update=False)

            with torch.no_grad():
                val_loss = valid_one_epoch(epoch, model, loss_fn, val_loader, device, scheduler=None, schd_loss_update=False)
                #torch.save(model.state_dict(),'model/{}_fold_{}_{}'.format(CFG['model_arch'], fold, epoch))

                early_stopping(val_loss, model, fold, epoch, CFG['mixup_flag'])

                if early_stopping.early_stop:
                    print("Early stopping")
                    break 
                    
                    
        if early_stopping.early_stop:
            print("Early stopping")
            break      

   
   
  
    torch.cuda.empty_cache()
    
    elapsed = time.time() - starttime
    print('Elapsed Time {}'.format(time.strftime("%H:%M:%S", time.gmtime(elapsed))))
    
    
    