
from torch import nn
import kornia as K
import kornia.feature as KF
import torch, os, cv2
from PIL import Image

    

def build_matcher(matcher_type=0, device='cuda', thr= 0.01, resolution=640):
    meta = {0: LoFTRWrapper0, 1: LoFTRWrapper, 2: ELoFTRWrapper} 
    return meta[matcher_type](thr= thr, resolution=resolution, device=device).eval().to(device)



class LoFTRWrapper0(nn.Module):
    def __init__(self, device='cuda', thr= 0.01, **kargs):
        super().__init__()


        self.matcher =  KF.LoFTR(pretrained="outdoor")
        self.matcher.coarse_matching.thr = thr

        self.device = device


    def to_tensor(self, image, device):
        if image.ndim > 2:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # image = K.image_to_tensor(image).float()[None, ...] / 255.0
        image = K.image_to_tensor(image, False).to(device).float() / 255.0

        return image


    def forward(self, image0, image1):

        image0 = self.to_tensor(image0, self.device)
        image1 = self.to_tensor(image1, self.device)

        batch = {"image0": image0, "image1": image1}

        with torch.inference_mode():
            correspondences = self.matcher(batch)


        return correspondences
    
class LoFTRWrapper(nn.Module):
    def __init__(self, device='cuda', resolution=512, weights=None, thr= 0.01, **kargs):
        super().__init__()


        self.matcher =  KF.LoFTR(pretrained="outdoor")
        self.matcher.coarse_matching.thr = thr

        self.device = device

        self.w_resized = resolution
        self.h_resized = resolution


    def forward(self, image0, image1):

        h1, w1  = image0.shape[:2]
        h2, w2 = image1.shape[:2]

        image0 = self.to_tensor(image0, self.device)
        image1 = self.to_tensor(image1, self.device)

        batch = {"image0": image0, "image1": image1}

        scale1 = torch.tensor(
                (w1*1.0 / self.w_resized, h1*1.0 / self.h_resized), device=self.device
            )[None]

        scale2 = torch.tensor(
                (w2*1.0 / self.w_resized, h2*1.0 / self.h_resized), device=self.device
            )[None]
        

        with torch.inference_mode():
            out = self.matcher(batch)

  
            correspondences = {
                "keypoints0": out["keypoints0"]*scale1,
                "keypoints1": out["keypoints1"]*scale2,
                "confidence": out["confidence"]
            }

        return correspondences


        
    def to_tensor(self, image, device):

        if image.ndim > 2:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # image = K.image_to_tensor(image).float()[None, ...] / 255.0
        image = K.image_to_tensor(image, False).to(device).float() / 255.0
        image = K.geometry.resize(image, (self.w_resized, self.h_resized), antialias=True)

        return image

from copy import deepcopy

import torch
import cv2
import numpy as np

class ELoFTRWrapper(nn.Module):
    def __init__(self, device='cuda', resolution=512, weights=None, thr= 0.01, **kargs):
        super().__init__()

        from src.loftr import LoFTR, full_default_cfg, opt_default_cfg, reparameter

        # You can choose model type in ['full', 'opt']
        model_type = 'full' # 'full' for best quality, 'opt' for best efficiency

        # You can choose numerical precision in ['fp32', 'mp', 'fp16']. 'fp16' for best efficiency
        precision = 'fp32' # Enjoy near-lossless precision with Mixed Precision (MP) / FP16 computation if you have a modern GPU (recommended NVIDIA architecture >= SM_70).

        # You can also change the default values like thr. and npe (based on input image size)

        if model_type == 'full':
            _default_cfg = deepcopy(full_default_cfg)
        elif model_type == 'opt':
            _default_cfg = deepcopy(opt_default_cfg)
            
        if precision == 'mp':
            _default_cfg['mp'] = True
        elif precision == 'fp16':
            _default_cfg['half'] = True
            
        print(_default_cfg)
        matcher = LoFTR(config=_default_cfg)

        matcher.coarse_matching.thr = thr

        matcher.load_state_dict(torch.load("weights/eloftr_outdoor.ckpt", weights_only=False)['state_dict'])
        matcher = reparameter(matcher) # no reparameterization will lead to low performance

        if precision == 'fp16':
            matcher = matcher.half()

        self.matcher = matcher.eval().cuda()


        self.device = device

        self.w_resized = resolution
        self.h_resized = resolution


    def forward(self, image0, image1):

        h1, w1  = image0.shape[:2]
        h2, w2 = image1.shape[:2]

        image0 = self.to_tensor(image0, self.device)
        image1 = self.to_tensor(image1, self.device)

        batch = {"image0": image0, "image1": image1}

        scale1 = torch.tensor(
                (w1*1.0 / self.w_resized, h1*1.0 / self.h_resized), device=self.device
            )[None]

        scale2 = torch.tensor(
                (w2*1.0 / self.w_resized, h2*1.0 / self.h_resized), device=self.device
            )[None]
        

        with torch.inference_mode():
            self.matcher(batch)
  
            correspondences = {
                "keypoints0": batch['mkpts0_f']*scale1,
                "keypoints1": batch['mkpts1_f']*scale2,
                "confidence": batch['mconf']
            }

        return correspondences


        
    def to_tensor(self, image, device):

        if image.ndim > 2:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # image = K.image_to_tensor(image).float()[None, ...] / 255.0
        image = K.image_to_tensor(image, False).to(device).float() / 255.0
        image = K.geometry.resize(image, (self.w_resized, self.h_resized), antialias=True)

        return image
