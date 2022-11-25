import torch
import os
import glob
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from torchvision import transforms
from torch.optim import lr_scheduler
import torch.optim as optim
from data.triplet_loader import TestImageLoader, TestFileLoader
from model import TripletNet
from model import ComplexIrisNet, FeatNet
from model import ExtendedTripletLoss, TripletLoss
import numpy as np
import argparse
from util import plot_roc
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import scipy.io as sio
import logging
argparser = argparse.ArgumentParser(description="Set up the configuration for the testing.")
argparser.add_argument('--base_path', type=str,
                    default=None)
argparser.add_argument('--mask_path', type=str,
                    default=None)
argparser.add_argument('--checkpoints', type=str, default= 'snapshot/polyu')
argparser.add_argument('--save_path', type=str, default='match/')
argparser.add_argument('-d','--division', type=int, default=1)
argparser.add_argument('-f','--file', type=str, default=None)
argparser.add_argument('--two_session', action='store_true', default=False)
argparser.add_argument('-fg','--file_gallery', type=str, default=None)
argparser.add_argument('-fp','--file_probe', type=str, default=None)
argparser.add_argument('--base_path_gallery', type=str,
                    default=None)
argparser.add_argument('--mask_path_gallery', type=str,
                    default=None)
argparser.add_argument('--base_path_probe', type=str,
                    default=None)
argparser.add_argument('--mask_path_probe', type=str,
                    default=None)
argparser.add_argument('--phase', action='store_true', default=False)

args = argparser.parse_args()

@torch.no_grad()
def _compute_shift_loss(feat1, mask1, feat2, mask2, shift=8):
    feat1 = feat1.cuda()
    feat2 = feat2.cuda()
    if args.phase:
        feat1 = feat1.sign()
        feat2 = feat2.sign()
    mask1 = mask1.repeat(1, feat1.shape[1], 1, 1).clone().cuda()
    mask2 = mask2.repeat(1, feat2.shape[1], 1, 1).clone().cuda()
    dist = torch.zeros(2*shift+1, feat1.shape[0]).to(feat1.device)
    mr = torch.zeros(2*shift+1, feat1.shape[0]).to(feat1.device)
    for i in range(2*shift+1):
        featm2 = torch.roll(feat2, -shift+i, -1).clone()
        maskm2 = torch.roll(mask2, -shift+i, -1).clone()
        featm1 = feat1.clone()
        maskm1 = mask1.clone()
        
        mask = torch.bitwise_and(maskm1, maskm2)
        featm1[mask == False] = 0
        featm2[mask == False] = 0
        # Normalize by number of pixels
        dist[i] = torch.norm(
            (featm1-featm2), 2, dim=(1, 2, 3))/torch.sqrt(mask.sum(dim=(1, 2, 3))+1e-3)
        mr[i] = mask.sum(dim=(1, 2, 3))/(featm1.shape[1]*featm1.shape[2]*featm1.shape[3])
    # print(dist)
    loss, ind = dist.min(0, keepdim=True)
    mrate = torch.gather(mr, 0, ind)
    return loss.detach().cpu().numpy(), mrate.detach().cpu().numpy()

def compute_shift_loss(feat1, mask1, feat2, mask2, division, shift=16):
    if division == 1:
        return _compute_shift_loss(feat1, mask1, feat2, mask2, shift)
    else:
        slicing = int(feat1.shape[0]/division)
        slicing_points = [i*slicing for i in range(division)]
        slicing_points.append(feat1.shape[0])

        for i in range(division):
            loss, mrate = _compute_shift_loss(feat1[slicing_points[i]:slicing_points[i+1]], mask1=mask1[slicing_points[i]:slicing_points[i+1]], feat2=feat2[slicing_points[i]:slicing_points[i+1]], mask2=mask2[slicing_points[i]:slicing_points[i+1]], shift=shift)
            if i == 0:
                loss_all = loss
                mrate_all = mrate
            else:
                loss_all = np.concatenate((loss_all, loss), axis=1)
                mrate_all = np.concatenate((mrate_all, mrate), axis=1)
        return loss_all, mrate_all

@torch.no_grad()
def inference(base_path, mask_path, checkpoint, file=None, save_inter=True, mask_threshold = 0.06):
    print('Loading model from {}'.format(checkpoint))
    model = FeatNet()
    # model = torch.nn.DataParallel(model)
    model.cuda()
    statedict = torch.load(checkpoint)['state_dict']
    statedictNew = dict()
    for key, val in statedict.items():
        key = key.replace('module.embeddingnet.', '')
        statedictNew[key] = val
    model.load_state_dict(statedictNew)
    if file is None:
        testLoader = torch.utils.data.DataLoader(TestImageLoader(base_path, mask_path),
                                            batch_size=16, shuffle=False, num_workers=10, pin_memory=True)
    else:
        testLoader = torch.utils.data.DataLoader(TestFileLoader(base_path, mask_path, file),
                                            batch_size=16, shuffle=False, num_workers=10, pin_memory=True) 

    model.eval()
    featList = list()
    maskList = list()
    labelList = list()
    for batch_idx, (imgs, masks, labels) in tqdm(enumerate(testLoader)):
        imgs, masks = imgs.cuda(), masks.cuda()
        with torch.no_grad():
            feat = model(imgs)
            featList.append(feat.cpu())
            maskList.append(masks.cpu())
            labelList.append(labels)
    feats = torch.cat(featList, 0)
    print(f"feat shape is {feats.shape}")
    masks = torch.cat(maskList, 0)
    print(f"mask shape is {masks.shape}")
    label = np.concatenate(labelList)
    print(f"label shape is {label.shape}")
    for i, feat in enumerate(feats):
        masks[i][abs(feats[i]-feats[i].mean()) < mask_threshold] = False
        feats[i] = feats[i] > feats[i].mean()
    result_path = os.path.join(args.save_path, args.checkpoints.split(
        '/')[-1], os.path.basename(checkpoint).split('.')[0])
    np.save(os.path.join(result_path, 'feat.npy'), feats.cpu().numpy())
    np.save(os.path.join(result_path, 'mask.npy'), masks.cpu().numpy())      
    # binarized = False # TODO: change this to True if you want to binarize the features
    # if binarized:
    #     feat_bin = list()
    #     for ft in feat:
    #         feat_bin.append(ft > np.tile(ft.mean(axis=(1,2)), (8,32,1)).transpose(2,0,1))
    #     feat = np.stack(feat_bin, axis=0).astype(np.uint8)
    return feats, masks, label

def match_two_session():
    checkpoints = glob.glob(os.path.join(args.checkpoints, '*pth.tar'))
    checkpoints.sort(reverse=True)
    for checkpoint in checkpoints:
        feat_gallery, mask_gallery, label_gallery = inference(args.base_path_gallery, args.mask_path_gallery, checkpoint, args.file_gallery)
        feat_probe, mask_probe, label_probe = inference(args.base_path_probe, args.mask_path_probe, checkpoint, args.file_probe)
        shift = max(len(feat_gallery), len(feat_probe))
        crop = min(len(feat_gallery), len(feat_probe))
        scores = np.zeros((crop, shift))
        mrates = np.zeros(scores.shape)
        labelMat = np.zeros(scores.shape, dtype=np.int8)

        for i in tqdm(range(shift)):
            scores[:, i], mrates[:, i] = compute_shift_loss(
                feat_gallery[:crop], mask_gallery[:crop], torch.roll(feat_probe, -i, 0)[:crop], torch.roll(mask_probe, -i, 0)[:crop], args.division)
            labelMat[:, i] = (label_gallery[:crop] == np.roll(label_probe, -i)[:crop])

        for i in range(crop):
            scores[i] = np.roll(scores[i], i, 0)
            mrates[i] = np.roll(mrates[i], i, 0)
            labelMat[i] = np.roll(labelMat[i], i, 0)
        
        if len(scores) != len(feat_gallery):
            assert len(scores) == len(feat_probe)
            scores = scores.T
            mrates = mrates.T
            labelMat = labelMat.T
        result_path = os.path.join(args.save_path, args.checkpoints.split('/')[-1], os.path.basename(checkpoint).split('.')[0])
        if not os.path.isdir(result_path):
                os.makedirs(result_path)
        np.save(os.path.join(result_path, 'scores.npy'), scores)
        np.save(os.path.join(result_path, 'mrates.npy'), mrates)
        np.save(os.path.join(result_path, 'labelMat.npy'), labelMat)
        # gg = scores[np.bitwise_and(labelMat == 1, mrates > 0.5)]
        # ii = scores[np.bitwise_and(labelMat == 0, mrates > 0.5)]
        gg = scores[labelMat == 1]
        ii = scores[labelMat == 0]
        plot_roc(gg, ii, saveDir=os.path.join(args.save_path, args.checkpoints.split('/')[-1], os.path.basename(checkpoint).split('.')[0]+'_roc.png'))
        sio.savemat(os.path.join(result_path, 'scores.mat'), {'scores':scores, 'mrates':mrates, 'labelMat':labelMat})
            
def match_one_session():
    checkpoints = glob.glob(os.path.join(args.checkpoints, '*pth.tar'))
    checkpoints.sort()
    for checkpoint in checkpoints:
        feat, mask, label = inference(args.base_path, args.mask_path, checkpoint, args.file)
        scores = np.zeros((len(feat), len(feat)//2))
        mrates = np.zeros((len(feat), len(feat)//2))
        labelMat = np.zeros(scores.shape, dtype=np.int8)
        with torch.no_grad():
            for i in tqdm(range(1, len(feat)//2+1)):
                scores[:, i-1], mrates[:, i-1] = compute_shift_loss(
                    feat, mask, torch.roll(feat, -i, dims = 0), torch.roll(mask, -i, dims = 0), args.division)
                labelMat[:, i-1] = (label == np.roll(label, -i))
            result_path = os.path.join(args.save_path, args.checkpoints.split('/')[-1], os.path.basename(checkpoint).split('.')[0])
        if not os.path.isdir(result_path):
            os.makedirs(result_path)
        np.save(os.path.join(result_path, 'scores.npy'), scores)
        np.save(os.path.join(result_path, 'mrates.npy'), mrates)
        np.save(os.path.join(result_path, 'labelMat.npy'), labelMat)
        # gg = scores[np.bitwise_and(labelMat == 1, mrates > 0.5)]
        # ii = scores[np.bitwise_and(labelMat == 0, mrates > 0.5)]
        gg = scores[labelMat == 1]
        ii = scores[labelMat == 0]
        plot_roc(gg, ii, saveDir=os.path.join(args.save_path, args.checkpoints.split('/')[-1], os.path.basename(checkpoint).split('.')[0]+'_roc.png'))
        sio.savemat(os.path.join(result_path, 'scores.mat'), {'scores':scores, 'mrates':mrates, 'labelMat':labelMat})

if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    if args.two_session:
        match_two_session()
    else:
        match_one_session()
   
