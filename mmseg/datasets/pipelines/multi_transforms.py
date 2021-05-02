import mmcv
import numpy as np
from mmcv.utils import deprecated_api_warning, is_tuple_of
from numpy import random
from .transforms import PhotoMetricDistortion

from ..builder import PIPELINES


@PIPELINES.register_module()
class MultiNormalize(object):
    """Normalize the concated images.

    Added key is "img_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb: (bool): Whether to convert the image from BGR to RGB, 
            default is true
    """

    def __init__(self, mean,  std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns: 
            dict: Normalized rsults, 'igmg_norm_cfg' key is added into 
                result dict.
        """

        imgs = results['img']
        img_num = imgs.shape[-1] //3 # should be 6//3, 9//3, 12//3...

        for i in range(img_num):
            # Extract one image
            img = imgs[...,3*i:3*(i+1)]
            img = mmcv.imnormalize(img, self.mean, self.std,
                                    self.to_rgb)
            #concat img
            if i==0:
                img_concat = img 
            else:
                img_concat = np.concatenate([img_concat, img], axis=2)
        results['img'] = img_concat
        results['img_norm_cfg'] = dict(
            mean=self.mean,  std=self.std, to_rgb=self.to_rgb
        )
        return results


@PIPELINES.register_module()
class MultiPhotoMetricDistortion(object):
    """Apply photometric distortion to image sequentially every transformation
    is applied with a probabillity of 0.5. The position of random contrast is in 
    second or second to last.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)

    Args: 
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range ( tuple): range of saturation.
        heu_delta (int): delta of hue.
    """

    def __init__(self,
                brightness_delta=32,
                contrast_range=(0.5,1.5),
                saturation_range=(0.5, 1.5),
                hue_delta=18):

        self.converter = PhotoMetricDistortion(
            brightness_delta,
            contrast_range,
            saturation_range,
            hue_delta
        )

    def __call__(self, results):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """

        imgs = results['img']
        img_num = imgs.shape[-1] //3 # should be 6//3, 9//3, 12//3...

        for i in range(img_num):
            # Extract one image
            img = imgs[...,3*i:3*(i+1)]
            img = self.converter({'img': img})['img']
            #concat img
            if i==0:
                img_concat = img 
            else:
                img_concat = np.concatenate([img_concat, img], axis=2)
        results['img'] = img_concat
        
        return results



