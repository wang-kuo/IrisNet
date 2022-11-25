from PIL import Image
import os
import os.path
import torch.utils.data
import torchvision.transforms as transforms
import random
from collections import defaultdict

def default_image_loader(path):
    return Image.open(path).convert('L')

class QuadletFileLoader(torch.utils.data.Dataset):
    def __init__(self, base_path, mask_path, filename, transform=None, loader=default_image_loader, sampling = 1):
        self.base_path = base_path  
        self.mask_path = mask_path
        self.samples = defaultdict(list)
        self.sampling = sampling
        with open(filename, 'r') as f:
            lines = f.readlines()
        for line in lines:
            imname, maskname, label = line.strip().split()
            self.samples[label].append((os.path.join(base_path, imname),
                                    os.path.join(mask_path, maskname)))
        self.quadlets = list()
        self._quadlets = list()
        for sbj in self.samples.keys():
            for i, anchor in enumerate(self.samples[sbj]):
                for positive in self.samples[sbj][i+1:]:
                    if positive == None:
                        continue
                    while True:
                        negSbj = random.choice(list(self.samples.keys()))
                        if negSbj != sbj:
                            break
                    while True:
                        negSbj2 = random.choice(list(self.samples.keys()))
                        if negSbj2 != sbj and negSbj2 != negSbj:
                            break
                    negative = random.choice(self.samples[negSbj])
                    negative2 = random.choice(self.samples[negSbj2])
                    self._quadlets.append((*anchor, *positive, *negative, *negative2))
        self.quadlets = random.sample(self._quadlets, k=int(len(self._quadlets)*self.sampling))
        self.transform = transform
        self.transform2 = transforms.Compose([
                            transforms.ToTensor()
                        ])
        self.loader = loader

    def __getitem__(self, index):
        a, ma, p, mp, n, mn, n2, mn2 = self.quadlets[index]
        imgA = self.loader(a)
        imgP = self.loader(p)
        imgN = self.loader(n)
        imgN2 = self.loader(n2)
        maskA = self.transform2(self.loader(ma)).bool()
        maskP = self.transform2(self.loader(mp)).bool()
        maskN = self.transform2(self.loader(mn)).bool()
        maskN2 = self.transform2(self.loader(mn2)).bool()
        if self.transform is not None:
            imgA = self.transform(imgA)
            imgP = self.transform(imgP)
            imgN = self.transform(imgN)
            imgN2 = self.transform(imgN2)
        
        return imgA, imgP, imgN, imgN2, maskA, maskP, maskN, maskN2

    def __len__(self):
        return len(self.quadlets)
    
    def resample(self):
        self.quadlets = random.sample(self._quadlets, k=int(len(self._quadlets)*self.sampling))
        return self

class TestFileLoader(torch.utils.data.Dataset):
    def __init__(self, base_path, mask_path, filename, transform=None, loader=default_image_loader):
        self.samples = list()
        self.masks = list()
        self.labels =list()
        with open(filename, 'r') as f:
            lines = f.readlines()
        for line in lines: 
            imname, label = line.strip().split()
            imname = imname.replace('S1/Norm_En/', '')
            imname = imname.replace('S2/Norm_En/', '')
            self.samples.append(os.path.join(base_path, imname))
            self.masks.append(os.path.join(mask_path, imname))
            self.labels.append(label)      

        if transform is None:
            self.transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((64, 256))])
        else:
            self.transform = transform
        self.transform2 = transforms.Compose(
            [transforms.ToTensor()])
        self.loader = loader

    def __getitem__(self, index):
        img = self.transform(self.loader(self.samples[index]))   
        mask = self.transform2(self.loader(self.masks[index])).bool()
        label = self.labels[index]

        return img, mask, label

    def __len__(self):
        return len(self.samples)