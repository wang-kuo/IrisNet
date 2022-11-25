from sre_constants import ANY_ALL
import torch
import torch.nn as nn
import torch.nn.functional as F


class ExtendedTripletLoss(nn.Module):
    '''
    Compute the extended triplet loss with bit shift and masks.
    '''
    def __init__(self, margin = 0.15):
        super(ExtendedTripletLoss, self).__init__()
        self.margin = margin 
    
    def forward(self, a, p, n, ma, mp, mn):
        def compute_shift_loss(feat1, mask1, feat2, mask2, shift=4):
            mask1 = mask1.repeat(1, feat1.shape[1], 1, 1).clone()
            mask2 = mask2.repeat(1, feat2.shape[1], 1, 1).clone()
            dist = torch.zeros(2*shift+1, feat1.shape[0]).to(feat1.device)
            for i in range(2*shift+1):
                featm2 = torch.roll(feat2, -shift+i, -1).clone()
                maskm2 = torch.roll(mask2, -shift+i, -1).clone()
                featm1 = feat1.clone()
                maskm1 = mask1.clone()
                mask = torch.bitwise_and(maskm1, maskm2)
                featm1[mask==False] = 0
                featm2[mask==False] = 0
                dist[i] = torch.norm(featm1-featm2,2, dim=(1,2,3))**2/(mask.sum(dim=(1,2,3))+1e-3) # Normalize by number of pixels
            # print(dist)
            loss, _ = dist.min(0)
            return loss
        ap_loss = compute_shift_loss(a, ma, p, mp)
        an_loss = compute_shift_loss(a, ma, n, mn)
        loss = F.relu(ap_loss-an_loss+self.margin)
        return loss.mean()

class ExtendedQuadletLoss(nn.Module):
    '''
    Compute the extended triplet loss with bit shift and masks.
    '''
    def __init__(self, margin = 0.15, margin2 = 0.2):
        super(ExtendedQuadletLoss, self).__init__()
        self.margin = margin
        self.margin2 = margin2
    
    def forward(self, a, p, n, n2, ma, mp, mn, mn2):
        def compute_shift_loss(feat1, mask1, feat2, mask2, shift=8):
            mask1 = mask1.repeat(1, feat1.shape[1], 1, 1).clone()
            mask2 = mask2.repeat(1, feat2.shape[1], 1, 1).clone()
            dist = torch.zeros(2*shift+1, feat1.shape[0]).to(feat1.device)
            for i in range(2*shift+1):
                featm2 = torch.roll(feat2, -shift+i, -1).clone()
                maskm2 = torch.roll(mask2, -shift+i, -1).clone()
                featm1 = feat1.clone()
                maskm1 = mask1.clone()
                mask = torch.bitwise_and(maskm1, maskm2)
                featm1[mask==False] = 0
                featm2[mask==False] = 0
                dist[i] = torch.norm(featm1-featm2,2, dim=(1,2,3))**2/(mask.sum(dim=(1,2,3))+1e-3) # Normalize by number of pixels
            # print(dist)
            loss, _ = dist.min(0)
            return loss
        ap_loss = compute_shift_loss(a, ma, p, mp)
        an_loss = compute_shift_loss(a, ma, n, mn)
        nn2_loss = compute_shift_loss(n, mn, n2, mn2)
        loss = F.relu(ap_loss-an_loss+self.margin) + F.relu(ap_loss-nn2_loss+self.margin2)
        return loss.mean()


class TripletLoss(nn.Module):
    '''
    Compute normal triplet loss or soft margin triplet loss given triplets
    '''
    def __init__(self, margin = None):
        super(TripletLoss, self).__init__()
        self.margin = margin
        if self.margin is None:  # use soft-margin
            self.Loss = nn.SoftMarginLoss()
        else:
            self.Loss = nn.TripletMarginLoss(margin = margin, p = 2)

    def forward(self, anchor, pos, neg):
        if self.margin is None:
            num_samples = anchor.shape[0]
            y = torch.ones((num_samples, 1)).view(-1)
            if anchor.is_cuda: y = y.cuda()
            ap_dist = torch.norm(anchor - pos, 2, dim = (1,2,3)).view(-1)
            an_dist = torch.norm(anchor - neg, 2, dim = (1,2,3)).view(-1)
            loss = self.Loss(an_dist - ap_dist, y)
        else:
            loss = self.Loss(anchor, pos, neg)

        return loss


if __name__ == '__main__':
    pass