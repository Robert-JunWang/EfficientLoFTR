from pyslam.utilities.logging import Printer
from pyslam.utilities.system import import_from
from pyslam.utilities.data_management import AtomicCounter
from pyslam.utilities.serialization import SerializableEnum, register_class
# from pyslam.utilities.dust3r import Dust3rImagePreprocessor
from pyslam.config_parameters import Parameters
from pyslam.local_features.feature_matcher import kVerbose, FeatureMatcher, FeatureMatcherTypes, MatcherUtils, kDefaultRatioTest, FeatureDetectorTypes, FeatureDescriptorTypes, FeatureMatchingResult
import torch
import kornia.feature as KF

import cv2
import numpy as np

from matcher_wrapper import build_matcher

class DetectorFreeMatcher(FeatureMatcher):
    def __init__(
        self,
        norm_type=cv2.NORM_L2,
        cross_check=False,
        ratio_test=kDefaultRatioTest,
        matcher_type=FeatureMatcherTypes.LOFTR,
        detector_type=FeatureDetectorTypes.NONE,
        descriptor_type=FeatureDescriptorTypes.NONE,
        matcher_tid=0
    ):
        super().__init__(
            norm_type=norm_type,
            cross_check=cross_check,
            ratio_test=ratio_test,
            matcher_type=matcher_type,
            detector_type=detector_type,
            descriptor_type=descriptor_type,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        meta = {0:"LoFTRMatcher", 1: "LoFTRMatcher2", 2: "EDMMatcher", 3: "RoMaMatcher", 4: "RoMa8Matcher"}
        self.matcher_name = meta[matcher_tid]
        self.matcher_tid = matcher_tid


        # device = 'cpu' # force cpu mode
        if device.type == "cuda":
            print(f"{self.matcher_name}: Using CUDA")
        else:
            print(f"{self.matcher_name}: Using CPU")

        self.torch_device = device
        if self.torch_device == "cuda":
            torch.cuda.empty_cache()

        # if matcher_tid == 0:
        #     self.matcher = KF.LoFTR("outdoor").eval().to(device)
        # else:
        self.matcher = build_matcher(matcher_tid, resolution=640, thr=0.05).eval().to(device)

        print("device: ", self.torch_device)
        Printer.green(f"matcher: {self.matcher_name}")


    # input: des1 = queryDescriptors, des2= trainDescriptors
    # output: idxs1, idxs2  (vectors of corresponding indexes in des1 and des2, respectively)
    def match(
        self,
        img1,
        img2,
        des1,
        des2,
        kps1=None,
        kps2=None,
        ratio_test=None,
        row_matching=False,
        max_disparity=None,
        data=None,
    ):
        result = FeatureMatchingResult()

        result.des1 = des1
        result.des2 = des2
        result.kps1 = kps1
        result.kps2 = kps2

        # Optimize verbose checks: only do expensive operations if verbose is enabled
        if kVerbose:
            print(self.matcher_name, ", norm ", self.norm_type)
            print("matcher: ", self.matcher_type.name)
            if img1 is not None:
                print(f"img1.shape: {img1.shape}")
            print("des1.shape:", des1.shape, " des1.dtype:", des1.dtype)
            print("des2.shape:", des2.shape, " des2.dtype:", des2.dtype)
            if kps1 is not None and isinstance(kps1, np.ndarray):
                print("kps1.shape:", kps1.shape, " kps1.dtype:", kps1.dtype)
            if kps2 is not None and isinstance(kps2, np.ndarray):
                print("kps2.shape:", kps2.shape, " kps2.dtype:", kps2.dtype)

        if ratio_test is None:
            ratio_test = self.ratio_test
            # print(f'[FeatureMatcher.match]: ratio test: {ratio_test}')


        out_matching = self.matcher(img1, img2)
        kps1 = out_matching["keypoints0"].cpu().numpy()
        kps1 = np.array([cv2.KeyPoint(int(p[0]), int(p[1]), size=1, response=1) for p in kps1])
        kps2 = out_matching["keypoints1"].cpu().numpy()
        kps2 = np.array([cv2.KeyPoint(int(p[0]), int(p[1]), size=1, response=1) for p in kps2])
        # idxs = out_matching['batch_indexes'].cpu().numpy()
        # print(f'idxs.shape: {idxs.shape}, idxs.dtype: {idxs.dtype}')
        result.kps1 = kps1
        result.kps2 = kps2
        result.idxs1 = np.arange(len(kps1), dtype=np.int32)
        result.idxs2 = np.arange(len(kps2), dtype=np.int32)
        if row_matching:
            result.idxs1, result.idxs2 = MatcherUtils.filterNonRowMatches(
                kps1, result.idxs1, kps2, result.idxs2, max_disparity=max_disparity
            )
        return result
