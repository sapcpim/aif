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

#import timm
import time

import sklearn
import warnings
import joblib
from sklearn.metrics import roc_auc_score, log_loss
from sklearn import metrics
#import warnings
#import cv2
#import pydicom
#from efficientnet_pytorch import EfficientNet
from scipy.ndimage.interpolation import zoom

from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)

from albumentations.pytorch import ToTensorV2

CFG = {
    'fold_num': 5,
    'seed': 777,
    'model_arch': 'tf_efficientnet_b6_ns',
    'img_size': 512,  # original one
    # 'img_size': 256,    #changed by jw
    'num_classes': 5,
    'epochs': 36,
    'train_bs': 4,
    'valid_bs': 8,
    'test_bs': 16,
    'T_0': 10,
    'lr': 1e-4,
    'min_lr': 1e-6,
    'weight_decay': 1e-6,
    'num_workers': 4,
    'accum_iter': 2,  # suppoprt to do batch accumulation for backprop with effectively larger batch size
    'verbose_step': 1,
    # 'device': 'cuda:0',
    'patience': 10,  # check patience by jwp
    'alpha': 0.25,  # Focal Loss
    'gamma': 1.5,  # Focal Loss
    'device': 'cuda:0',

    ####### inference ########
    'tta': 3,
    # 'used_models': ['tf_efficientnet_b5_ns','tf_efficientnet_b6_ns','tf_efficientnet_b6_ns','tf_efficientnet_b6_ns','tf_efficientnet_b6_ns'],
    'used_epochs': [12, 12, 12],  # pick best epoch
    'used_folds': [3, 2, 1],  # pick best folds
    'weights': [3, 2, 1]

}

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def seed_everything(seed):
    random_seed = seed
    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def get_img(path):
    im_bgr = cv2.imread(path)
    im_rgb = im_bgr[:, :, ::-1]
    im_rgb = cv2.resize (im_rgb, (CFG['img_size'], CFG['img_size']))
    #print(im_rgb)
    return im_rgb


def get_img_rgb(path):
    im_bgr = cv2.imread(path)
    im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
    #im_rgb = cv2.resize (im_rgb, (512,512))
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
        self.transforms = transforms
        self.data_root = data_root
        self.do_fmix = do_fmix
        self.fmix_params = fmix_params
        self.do_cutmix = do_cutmix
        self.cutmix_params = cutmix_params
        self.do_mixup = do_cutmix
        self.mixup_params = mixup_params

        self.num_classes = CFG['num_classes']

        self.output_label = output_label
        self.one_hot_label = one_hot_label

        if output_label == True:
            self.labels = self.df['label'].values
            # print(self.labels)

            if one_hot_label is True:
                self.labels = np.eye(self.num_classes)[self.labels]
                # print(self.labels)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index: int):

        # get labels
        if self.output_label:
            target = self.labels[index]

        img = get_img("{}/{}".format(self.data_root, self.df.loc[index]['filename']))

        if self.transforms:
            img = self.transforms(image=img)['image']

        # do label smoothing
        # print(type(img), type(target))
        if self.output_label == True:
            return img, target
        else:
            return img

def get_inference_transforms():
    return Compose([
            RandomResizedCrop(CFG['img_size'], CFG['img_size']),
            Transpose(p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
            RandomBrightnessContrast(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5),
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


def do_main():
    # for training only, need nightly build pytorch
    # device = torch.device(CFG['device'])

    submission_list = []
    # images = glob.glob(os.path.join('./cassava/test_images', '*.jpg'))
    for dirname, _, filenames in os.walk('/kaggle/input/cassava-leaf-disease-classification/test_images/'):
        for filename in filenames:
            submission_list.append(filename)

        # with open(path + 'test_submission.csv', 'w') as f:
    #    for line in submission:
    #        f.write(line)

    col_name = ['image_id']
    submission = pd.DataFrame(submission_list, columns=col_name)

    path = '/kaggle/input/cassava-leaf-disease-classification/test_images'
    testfilename = '/kaggle/input/cassava-leaf-disease-classification/test_submission.csv'

    model = PlantImgClassifier(CFG['model_arch'], CFG['num_classes'], pretrained=False).to(device)
    seed_everything(CFG['seed'])

    print('Inference started')

    test_ds = PlantDataset(submission, path, transforms=get_inference_transforms(), output_label=False)

    tst_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=CFG['test_bs'],
        num_workers=CFG['num_workers'],
        shuffle=False,
        pin_memory=False,
    )

    tst_preds = []

    for i, epoch in enumerate(CFG['used_epochs']):

        model.load_state_dict(
            torch.load('../input/bestpt24/{}_epoch{}_fold{}'.format(CFG['model_arch'], epoch, CFG['used_folds'][i])))

        with torch.no_grad():
            for _ in range(CFG['tta']):
                tst_preds += [
                    CFG['weights'][i] / sum(CFG['weights']) / CFG['tta'] * inference_one_epoch(model, tst_loader,
                                                                                               device)]

    # print()
    # print('tst preds softmax values : {}'.format(np.mean(tst_preds, axis=0)))

    tst_preds = np.mean(tst_preds, axis=0)

    submission['label'] = np.argmax(tst_preds, axis=1)

    # print('tst preds argmax values : {}'.format(np.argmax(tst_preds, axis=1)))

    path_save = 'submission.csv'
    submission.to_csv(path_save, index=False)

    del model
    torch.cuda.empty_cache()

if __name__ == '__main__':
    do_main()
