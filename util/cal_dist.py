import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import cv2
import scipy.io as sio
from glob import glob
from tqdm import tqdm
import logging

def get_fm(featDir, maskDir, maskThreshold=0.3):
    imList = sorted(glob(os.path.join(featDir, '*.mat')))
    feats, masks = list(), list()
    for im in imList:
        feat = sio.loadmat(im)['output']
        if feat.shape[0] < feat.shape[1]:
            feat = feat.T
        mask = np.array(cv2.imread(os.path.join(maskDir, os.path.basename(im)[
                        :-4]+'.bmp'), cv2.IMREAD_GRAYSCALE).T, dtype=bool)
        assert feat.shape == mask.shape
        assert feat.shape[0] > feat.shape[1]
        mask[abs(feat-feat.mean()) < maskThreshold] = False
        feat = feat > feat.mean()
        feats.append(feat)
        masks.append(mask)
    return np.dstack(feats), np.dstack(masks)


def rotate_map(featmap: np.array, shift: int = 16):
    feats = list()
    for i in range(featmap.shape[2]):
        feattemp = list()
        for sf in range(-shift, shift+1):
            feattemp.append(np.roll(featmap[:, :, i], sf, axis=0).ravel())
        feats.append(np.vstack(feattemp))
    return np.dstack(feats)


def repeat_map(featmap: np.array, shift: int = 16):
    feats = list()
    for i in range(featmap.shape[2]):
        feattemp = list()
        for sf in range(-shift, shift+1):
            feattemp.append(featmap[:, :, i].ravel())
        feats.append(np.vstack(feattemp))
    return np.dstack(feats)


def get_label(gallery, probe, type, lr=-1):
    #TODO(KUO): add the left right difference.
    gList = sorted(glob(os.path.join(gallery, '*.'+type)))
    pList = sorted(glob(os.path.join(probe, '*.'+type)))
    labelG, labelP = list(), list()
    labelMat = np.zeros((len(gList), len(pList)))
    for g in gList:
        sign = 1 if os.path.basename(g).split('_')[1][0] == '1' else -1
        labelG.append(int(os.path.basename(g).split('_')[0])*sign)
    for p in pList:
        sign = 1 if os.path.basename(p).split('_')[1][0] == '1' else -1
        labelP.append(int(os.path.basename(p).split('_')[0])*sign)
    assert labelG == labelP
    for ig, gg in enumerate(labelG):
        for ip, pp in enumerate(labelP):
            labelMat[ig, ip] = 1 if gg == pp else -1
    return labelG, labelP, labelMat


def cal_hamming(featDir_G, featDir_P, maskDir_G, maskDir_P, batch=200, fold=1):
    assert torch.cuda.is_available()
    featGAll, maskGAll = get_fm(featDir_G, maskDir_G)
    featPAll, maskPAll = get_fm(featDir_P, maskDir_P)
    batchsize = featGAll.shape[2]//fold+1
    assert featGAll.shape == maskGAll.shape
    scoresAll = torch.zeros((featGAll.shape[2], featPAll.shape[2])).cuda()
    mratesAll = torch.zeros((featGAll.shape[2], featPAll.shape[2])).cuda()
    for folddiv in range(fold):
        featG = featGAll[..., folddiv *
                         batchsize:min(featGAll.shape[2], (folddiv+1)*batchsize)].copy()
        maskG = maskGAll[..., folddiv *
                         batchsize:min(maskGAll.shape[2], (folddiv+1)*batchsize)].copy()
        featG = rotate_map(featG)
        maskG = rotate_map(maskG)
        for folddivP in range(fold):
            featP = featPAll[..., folddivP *
                             batchsize:min(featPAll.shape[2], (folddivP+1)*batchsize)].copy()
            maskP = maskPAll[..., folddivP *
                             batchsize:min(maskPAll.shape[2], (folddivP+1)*batchsize)].copy()
            featP = repeat_map(featP)
            maskP = repeat_map(maskP)
            scores = torch.zeros((featG.shape[2], featP.shape[2])).cuda()
            mrates = torch.zeros((featG.shape[2], featP.shape[2])).cuda()

            for ga in tqdm(range(featG.shape[2]//batch+1)):
                for pr in range(featP.shape[2]//batch+1):
                    featGT = torch.from_numpy(
                        featG[..., ga*batch:min((ga+1)*batch, featG.shape[2])]).cuda()  # gallery feat tensor
                    featPT = torch.from_numpy(
                        featP[..., pr*batch:min((pr+1)*batch, featP.shape[2])]).cuda()  # probe feat tensor
                    maskGT = torch.from_numpy(
                        maskG[..., ga*batch:min((ga+1)*batch, maskG.shape[2])]).cuda()  # gallery mask tensor
                    maskPT = torch.from_numpy(
                        maskP[..., pr*batch:min((pr+1)*batch, maskP.shape[2])]).cuda()  # probe mask tensor

                    assert featGT.shape == maskGT.shape
                    assert featPT.shape == maskPT.shape

                    if featGT.shape[2] == 0 or featPT.shape[2] == 0:
                        continue
                    if featPT.shape[2] > featGT.shape[2]:
                        scoresTemp = torch.zeros(
                            (featGT.shape[2], featPT.shape[2])).cuda()
                        mratesTemp = torch.zeros(
                            (featGT.shape[2], featPT.shape[2])).cuda()
                        for i in range(featPT.shape[2]):
                            tempMask = torch.bitwise_and(torch.roll(
                                maskPT, -i, 2)[..., :maskGT.shape[2]], maskGT)
                            tempDist = torch.bitwise_and(torch.bitwise_xor(
                                featGT, torch.roll(featPT, -i, 2)[..., :featGT.shape[2]]), tempMask)
                            tempMsum = torch.sum(tempMask, 1)
                            tempScore = torch.min(
                                torch.sum(tempDist, 1)/tempMsum, 0, True)
                            scoresTemp[:, i] = tempScore.values.squeeze()
                            mratesTemp[:, i] = torch.gather(
                                tempMsum, 0, tempScore.indices).squeeze()
                        for j in range(len(scoresTemp)):
                            scoresTemp[j, :] = torch.roll(scoresTemp[j, :], j)
                            mratesTemp[j, :] = torch.roll(mratesTemp[j, :], j)
                        scores[ga*batch:min((ga+1)*batch, featG.shape[2]), pr *
                               batch:min((pr+1)*batch, featP.shape[2])] = scoresTemp
                        mrates[ga*batch:min((ga+1)*batch, featG.shape[2]), pr *
                               batch:min((pr+1)*batch, featP.shape[2])] = mratesTemp
                    else:
                        scoresTemp = torch.zeros(
                            (featPT.shape[2], featGT.shape[2])).cuda()
                        mratesTemp = torch.zeros(
                            (featPT.shape[2], featGT.shape[2])).cuda()
                        for i in range(maskGT.shape[2]):
                            tempMask = torch.bitwise_and(torch.roll(
                                maskGT, -i, 2)[..., :maskPT.shape[2]], maskPT)
                            tempDist = torch.bitwise_and(torch.bitwise_xor(
                                featPT, torch.roll(featGT, -i, 2)[..., :featPT.shape[2]]), tempMask)
                            tempMsum = torch.sum(tempMask, 1)
                            tempScore = torch.min(
                                torch.sum(tempDist, 1)/tempMsum, 0, True)
                            scoresTemp[:, i] = tempScore.values.squeeze()
                            mratesTemp[:, i] = torch.gather(
                                tempMsum, 0, tempScore.indices).squeeze()
                        for j in range(len(scoresTemp)):
                            scoresTemp[j, :] = torch.roll(scoresTemp[j, :], j)
                            mratesTemp[j, :] = torch.roll(mratesTemp[j, :], j)
                        scores[ga*batch:min((ga+1)*batch, featG.shape[2]), pr*batch:min(
                            (pr+1)*batch, featP.shape[2])] = scoresTemp.transpose(0, 1)
                        mrates[ga*batch:min((ga+1)*batch, featG.shape[2]), pr*batch:min(
                            (pr+1)*batch, featP.shape[2])] = mratesTemp.transpose(0, 1)
            scoresAll[folddiv*batchsize:min(featGAll.shape[2], (folddiv+1)*batchsize),
                      folddivP*batchsize:min(maskPAll.shape[2], (folddivP+1)*batchsize)] = scores
            mratesAll[folddiv*batchsize:min(featGAll.shape[2], (folddiv+1)*batchsize),
                      folddivP*batchsize:min(maskPAll.shape[2], (folddivP+1)*batchsize)] = mrates

    return scoresAll.cpu().numpy(), mratesAll.cpu().numpy()/featGAll.shape[1]



def cal_ws(featDir_G, featDir_P, maskDir_G, maskDir_P, w, batch=200, fold=1):
    assert torch.cuda.is_available()
    featGAll, maskGAll = get_fm(featDir_G, maskDir_G)
    featPAll, maskPAll = get_fm(featDir_P, maskDir_P)
    batchsize = featGAll.shape[2]//fold+1
    assert featGAll.shape == maskGAll.shape
    scoresAll = torch.zeros((featGAll.shape[2], featPAll.shape[2])).cuda()
    mratesAll = torch.zeros((featGAll.shape[2], featPAll.shape[2])).cuda()
    for folddiv in range(fold):
        featG = featGAll[..., folddiv *
                         batchsize:min(featGAll.shape[2], (folddiv+1)*batchsize)].copy()
        maskG = maskGAll[..., folddiv *
                         batchsize:min(maskGAll.shape[2], (folddiv+1)*batchsize)].copy()
        featG = rotate_map(featG)
        maskG = rotate_map(maskG)
        for folddivP in range(fold):
            featP = featPAll[..., folddivP *
                             batchsize:min(featPAll.shape[2], (folddivP+1)*batchsize)].copy()
            maskP = maskPAll[..., folddivP *
                             batchsize:min(maskPAll.shape[2], (folddivP+1)*batchsize)].copy()
            featP = repeat_map(featP)
            maskP = repeat_map(maskP)
            scores = torch.zeros((featG.shape[2], featP.shape[2])).cuda()
            mrates = torch.zeros((featG.shape[2], featP.shape[2])).cuda()

            for ga in tqdm(range(featG.shape[2]//batch+1)):
                for pr in range(featP.shape[2]//batch+1):
                    featGT = torch.from_numpy(
                        featG[..., ga*batch:min((ga+1)*batch, featG.shape[2])]).cuda()  # gallery feat tensor
                    featPT = torch.from_numpy(
                        featP[..., pr*batch:min((pr+1)*batch, featP.shape[2])]).cuda()  # probe feat tensor
                    maskGT = torch.from_numpy(
                        maskG[..., ga*batch:min((ga+1)*batch, maskG.shape[2])]).cuda()  # gallery mask tensor
                    maskPT = torch.from_numpy(
                        maskP[..., pr*batch:min((pr+1)*batch, maskP.shape[2])]).cuda()  # probe mask tensor

                    assert featGT.shape == maskGT.shape
                    assert featPT.shape == maskPT.shape

                    if featGT.shape[2] == 0 or featPT.shape[2] == 0:
                        continue
                    if featPT.shape[2] > featGT.shape[2]:
                        scoresTemp = torch.zeros(
                            (featGT.shape[2], featPT.shape[2])).cuda()
                        mratesTemp = torch.zeros(
                            (featGT.shape[2], featPT.shape[2])).cuda()
                        for i in range(featPT.shape[2]):
                            tempMask = torch.bitwise_and(torch.roll(
                                maskPT, -i, 2)[..., :maskGT.shape[2]], maskGT)
                            tempDist_0 = torch.bitwise_and(torch.bitwise_and(
                                featGT, torch.roll(featPT, -i, 2)[..., :featGT.shape[2]]), tempMask)
                            tempDist_1 = torch.bitwise_and(torch.bitwise_not(torch.bitwise_or(
                                featGT, torch.roll(featPT, -i, 2)[..., :featGT.shape[2]])), tempMask)
                            tempMsum = torch.sum(tempMask, 1)
                            tempScore = torch.max(
                                torch.sum(tempDist_0*w+(2-w)*tempDist_1, 1)/tempMsum, 0, True)
                            scoresTemp[:, i] = tempScore.values.squeeze()
                            mratesTemp[:, i] = torch.gather(
                                tempMsum, 0, tempScore.indices).squeeze()
                        for j in range(len(scoresTemp)):
                            scoresTemp[j, :] = torch.roll(scoresTemp[j, :], j)
                            mratesTemp[j, :] = torch.roll(mratesTemp[j, :], j)
                        scores[ga*batch:min((ga+1)*batch, featG.shape[2]), pr *
                               batch:min((pr+1)*batch, featP.shape[2])] = scoresTemp
                        mrates[ga*batch:min((ga+1)*batch, featG.shape[2]), pr *
                               batch:min((pr+1)*batch, featP.shape[2])] = mratesTemp
                    else:
                        scoresTemp = torch.zeros(
                            (featPT.shape[2], featGT.shape[2])).cuda()
                        mratesTemp = torch.zeros(
                            (featPT.shape[2], featGT.shape[2])).cuda()
                        for i in range(maskGT.shape[2]):
                            tempMask = torch.bitwise_and(torch.roll(
                                maskGT, -i, 2)[..., :maskPT.shape[2]], maskPT)
                            tempDist_0 = torch.bitwise_and(torch.bitwise_and(
                                featPT, torch.roll(featGT, -i, 2)[..., :featPT.shape[2]]), tempMask)
                            tempDist_1 = torch.bitwise_and(torch.bitwise_not(torch.bitwise_or(
                                featPT, torch.roll(featGT, -i, 2)[..., :featPT.shape[2]])), tempMask)
                            tempMsum = torch.sum(tempMask, 1)
                            tempScore = torch.max(
                                torch.sum(tempDist_0*w+(2-w)*tempDist_1, 1)/tempMsum, 0, True)
                            scoresTemp[:, i] = tempScore.values.squeeze()
                            mratesTemp[:, i] = torch.gather(
                                tempMsum, 0, tempScore.indices).squeeze()
                        for j in range(len(scoresTemp)):
                            scoresTemp[j, :] = torch.roll(scoresTemp[j, :], j)
                            mratesTemp[j, :] = torch.roll(mratesTemp[j, :], j)
                        scores[ga*batch:min((ga+1)*batch, featG.shape[2]), pr*batch:min(
                            (pr+1)*batch, featP.shape[2])] = scoresTemp.transpose(0, 1)
                        mrates[ga*batch:min((ga+1)*batch, featG.shape[2]), pr*batch:min(
                            (pr+1)*batch, featP.shape[2])] = mratesTemp.transpose(0, 1)
            scoresAll[folddiv*batchsize:min(featGAll.shape[2], (folddiv+1)*batchsize),
                      folddivP*batchsize:min(maskPAll.shape[2], (folddivP+1)*batchsize)] = scores
            mratesAll[folddiv*batchsize:min(featGAll.shape[2], (folddiv+1)*batchsize),
                      folddivP*batchsize:min(maskPAll.shape[2], (folddivP+1)*batchsize)] = mrates

    return scoresAll.cpu().numpy(), mratesAll.cpu().numpy()/featGAll.shape[1]