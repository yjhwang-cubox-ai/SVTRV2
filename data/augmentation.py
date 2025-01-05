import cv2
import numpy as np
from typing import Dict, List, Tuple
import random

class TextRecogAugmentations:
    def __init__(self, is_train: bool, prob: float = 0.4):
        # 단일 resize 객체만 생성
        self.resize = Resize(scale=(256, 64))
        
        if is_train:
            # train 모드일 때만 리스트 생성
            self.train_augs = [
                (TextRecogGeneralAug(), prob),
                (CropHeight(), prob),
                # ... 다른 augmentations
            ]
        self.is_train = is_train

    def __call__(self, image):
        if self.is_train:
            # 벡터화된 연산으로 한 번에 마스크 생성
            apply_mask = np.random.random(len(self.train_augs)) <= [prob for _, prob in self.train_augs]
            
            # 불필요한 중간 객체 생성 없이 직접 순회
            for apply, (aug, _) in zip(apply_mask, self.train_augs):
                if apply:
                    image = aug(image)
        
        # 단일 resize 연산만 수행
        return self.resize(image)
    

class TextRecogGeneralAug:
    def __call__(self, results: Dict) -> Dict:
        """Call function to pad images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """
        h, w = results['img'].shape[:2]
        if h >= 20 and w >= 20:
            results['img'] = self.tia_distort(results['img'], random.randint(3, 6))
            results['img'] = self.tia_stretch(results['img'], random.randint(3, 6))
        h, w = results['img'].shape[:2]
        if h >= 5 and w >= 5:
            results['img'] = self.tia_perspective(results['img'])
        results['img_shape'] = results['img'].shape[:2]
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += '()'
        return repr_str
    
    def tia_distort(self, img: np.ndarray, segment: int = 4) -> np.ndarray:
        """Image distortion.

        Args:
            img (np.ndarray): The image.
            segment (int): The number of segments to divide the image along
                the width. Defaults to 4.
        """
        img_h, img_w = img.shape[:2]

        cut = img_w // segment
        thresh = cut // 3

        src_pts = list()
        dst_pts = list()

        src_pts.append([0, 0])
        src_pts.append([img_w, 0])
        src_pts.append([img_w, img_h])
        src_pts.append([0, img_h])

        dst_pts.append([np.random.randint(thresh), np.random.randint(thresh)])
        dst_pts.append(
            [img_w - np.random.randint(thresh),
             np.random.randint(thresh)])
        dst_pts.append([
            img_w - np.random.randint(thresh),
            img_h - np.random.randint(thresh)
        ])
        dst_pts.append(
            [np.random.randint(thresh), img_h - np.random.randint(thresh)])

        half_thresh = thresh * 0.5

        for cut_idx in np.arange(1, segment, 1):
            src_pts.append([cut * cut_idx, 0])
            src_pts.append([cut * cut_idx, img_h])
            dst_pts.append([
                cut * cut_idx + np.random.randint(thresh) - half_thresh,
                np.random.randint(thresh) - half_thresh
            ])
            dst_pts.append([
                cut * cut_idx + np.random.randint(thresh) - half_thresh,
                img_h + np.random.randint(thresh) - half_thresh
            ])

        dst = self.warp_mls(img, src_pts, dst_pts, img_w, img_h)

        return dst

    def tia_stretch(self, img: np.ndarray, segment: int = 4) -> np.ndarray:
        """Image stretching.

        Args:
            img (np.ndarray): The image.
            segment (int): The number of segments to divide the image along
                the width. Defaults to 4.
        """
        img_h, img_w = img.shape[:2]

        cut = img_w // segment
        thresh = cut * 4 // 5

        src_pts = list()
        dst_pts = list()

        src_pts.append([0, 0])
        src_pts.append([img_w, 0])
        src_pts.append([img_w, img_h])
        src_pts.append([0, img_h])

        dst_pts.append([0, 0])
        dst_pts.append([img_w, 0])
        dst_pts.append([img_w, img_h])
        dst_pts.append([0, img_h])

        half_thresh = thresh * 0.5

        for cut_idx in np.arange(1, segment, 1):
            move = np.random.randint(thresh) - half_thresh
            src_pts.append([cut * cut_idx, 0])
            src_pts.append([cut * cut_idx, img_h])
            dst_pts.append([cut * cut_idx + move, 0])
            dst_pts.append([cut * cut_idx + move, img_h])

        dst = self.warp_mls(img, src_pts, dst_pts, img_w, img_h)

        return dst

    def tia_perspective(self, img: np.ndarray) -> np.ndarray:
        """Image perspective transformation.

        Args:
            img (np.ndarray): The image.
            segment (int): The number of segments to divide the image along
                the width. Defaults to 4.
        """
        img_h, img_w = img.shape[:2]

        thresh = img_h // 2

        src_pts = list()
        dst_pts = list()

        src_pts.append([0, 0])
        src_pts.append([img_w, 0])
        src_pts.append([img_w, img_h])
        src_pts.append([0, img_h])

        dst_pts.append([0, np.random.randint(thresh)])
        dst_pts.append([img_w, np.random.randint(thresh)])
        dst_pts.append([img_w, img_h - np.random.randint(thresh)])
        dst_pts.append([0, img_h - np.random.randint(thresh)])

        dst = self.warp_mls(img, src_pts, dst_pts, img_w, img_h)

        return dst

    def warp_mls(self,
                 src: np.ndarray,
                 src_pts: List[int],
                 dst_pts: List[int],
                 dst_w: int,
                 dst_h: int,
                 trans_ratio: float = 1.) -> np.ndarray:
        """Warp the image."""
        rdx, rdy = self._calc_delta(dst_w, dst_h, src_pts, dst_pts, 100)
        return self._gen_img(src, rdx, rdy, dst_w, dst_h, 100, trans_ratio)

    def _calc_delta(self, dst_w: int, dst_h: int, src_pts: List[int],
                    dst_pts: List[int],
                    grid_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """Compute delta."""

        pt_count = len(dst_pts)
        rdx = np.zeros((dst_h, dst_w))
        rdy = np.zeros((dst_h, dst_w))
        w = np.zeros(pt_count, dtype=np.float32)

        if pt_count < 2:
            return

        i = 0
        while True:
            if dst_w <= i < dst_w + grid_size - 1:
                i = dst_w - 1
            elif i >= dst_w:
                break

            j = 0
            while True:
                if dst_h <= j < dst_h + grid_size - 1:
                    j = dst_h - 1
                elif j >= dst_h:
                    break

                sw = 0
                swp = np.zeros(2, dtype=np.float32)
                swq = np.zeros(2, dtype=np.float32)
                new_pt = np.zeros(2, dtype=np.float32)
                cur_pt = np.array([i, j], dtype=np.float32)

                k = 0
                for k in range(pt_count):
                    if i == dst_pts[k][0] and j == dst_pts[k][1]:
                        break

                    w[k] = 1. / ((i - dst_pts[k][0]) * (i - dst_pts[k][0]) +
                                 (j - dst_pts[k][1]) * (j - dst_pts[k][1]))

                    sw += w[k]
                    swp = swp + w[k] * np.array(dst_pts[k])
                    swq = swq + w[k] * np.array(src_pts[k])

                if k == pt_count - 1:
                    pstar = 1 / sw * swp
                    qstar = 1 / sw * swq

                    miu_s = 0
                    for k in range(pt_count):
                        if i == dst_pts[k][0] and j == dst_pts[k][1]:
                            continue
                        pt_i = dst_pts[k] - pstar
                        miu_s += w[k] * np.sum(pt_i * pt_i)

                    cur_pt -= pstar
                    cur_pt_j = np.array([-cur_pt[1], cur_pt[0]])

                    for k in range(pt_count):
                        if i == dst_pts[k][0] and j == dst_pts[k][1]:
                            continue

                        pt_i = dst_pts[k] - pstar
                        pt_j = np.array([-pt_i[1], pt_i[0]])

                        tmp_pt = np.zeros(2, dtype=np.float32)
                        tmp_pt[0] = (
                            np.sum(pt_i * cur_pt) * src_pts[k][0] -
                            np.sum(pt_j * cur_pt) * src_pts[k][1])
                        tmp_pt[1] = (-np.sum(pt_i * cur_pt_j) * src_pts[k][0] +
                                     np.sum(pt_j * cur_pt_j) * src_pts[k][1])
                        tmp_pt *= (w[k] / miu_s)
                        new_pt += tmp_pt

                    new_pt += qstar
                else:
                    new_pt = src_pts[k]

                rdx[j, i] = new_pt[0] - i
                rdy[j, i] = new_pt[1] - j

                j += grid_size
            i += grid_size
        return rdx, rdy

    def _gen_img(self, src: np.ndarray, rdx: np.ndarray, rdy: np.ndarray,
                 dst_w: int, dst_h: int, grid_size: int,
                 trans_ratio: float) -> np.ndarray:
        """Generate the image based on delta."""

        src_h, src_w = src.shape[:2]
        dst = np.zeros_like(src, dtype=np.float32)

        for i in np.arange(0, dst_h, grid_size):
            for j in np.arange(0, dst_w, grid_size):
                ni = i + grid_size
                nj = j + grid_size
                w = h = grid_size
                if ni >= dst_h:
                    ni = dst_h - 1
                    h = ni - i + 1
                if nj >= dst_w:
                    nj = dst_w - 1
                    w = nj - j + 1

                di = np.reshape(np.arange(h), (-1, 1))
                dj = np.reshape(np.arange(w), (1, -1))
                delta_x = self._bilinear_interp(di / h, dj / w, rdx[i, j],
                                                rdx[i, nj], rdx[ni, j],
                                                rdx[ni, nj])
                delta_y = self._bilinear_interp(di / h, dj / w, rdy[i, j],
                                                rdy[i, nj], rdy[ni, j],
                                                rdy[ni, nj])
                nx = j + dj + delta_x * trans_ratio
                ny = i + di + delta_y * trans_ratio
                nx = np.clip(nx, 0, src_w - 1)
                ny = np.clip(ny, 0, src_h - 1)
                nxi = np.array(np.floor(nx), dtype=np.int32)
                nyi = np.array(np.floor(ny), dtype=np.int32)
                nxi1 = np.array(np.ceil(nx), dtype=np.int32)
                nyi1 = np.array(np.ceil(ny), dtype=np.int32)

                if len(src.shape) == 3:
                    x = np.tile(np.expand_dims(ny - nyi, axis=-1), (1, 1, 3))
                    y = np.tile(np.expand_dims(nx - nxi, axis=-1), (1, 1, 3))
                else:
                    x = ny - nyi
                    y = nx - nxi
                dst[i:i + h,
                    j:j + w] = self._bilinear_interp(x, y, src[nyi, nxi],
                                                     src[nyi, nxi1],
                                                     src[nyi1, nxi], src[nyi1,
                                                                         nxi1])

        dst = np.clip(dst, 0, 255)
        dst = np.array(dst, dtype=np.uint8)

        return dst

    def _bilinear_interp(self, x, y, v11, v12, v21, v22):
        """Bilinear interpolation.

        TODO: Docs for args and put it into utils.
        """
        return (v11 * (1 - y) + v12 * y) * (1 - x) + (v21 *
                                                      (1 - y) + v22 * y) * x

class CropHeight:
    """Randomly crop the image's height, either from top or bottom.

    Adapted from
    https://github.com/PaddlePaddle/PaddleOCR/blob/release%2F2.6/ppocr/data/imaug/rec_img_aug.py  # noqa

    Required Keys:

    - img

    Modified Keys:

    - img
    - img_shape

    Args:
        crop_min (int): Minimum pixel(s) to crop. Defaults to 1.
        crop_max (int): Maximum pixel(s) to crop. Defaults to 8.
    """

    def __init__(
        self,
        min_pixels: int = 1,
        max_pixels: int = 8,
    ) -> None:
        super().__init__()
        assert max_pixels >= min_pixels
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

    def get_random_vars(self):
        """Get all the random values used in this transform."""
        crop_pixels = int(random.randint(self.min_pixels, self.max_pixels))
        crop_top = random.randint(0, 1)
        return crop_pixels, crop_top

    def __call__(self, results: Dict) -> Dict:
        """Transform function to crop images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Cropped results.
        """
        h = results['img'].shape[0]
        crop_pixels, crop_top = self.get_random_vars()
        crop_pixels = min(crop_pixels, h - 1)
        img = results['img'].copy()
        if crop_top:
            img = img[crop_pixels:h, :, :]
        else:
            img = img[0:h - crop_pixels, :, :]
        results['img_shape'] = img.shape[:2]
        results['img'] = img
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(min_pixels = {self.min_pixels}, '
        repr_str += f'max_pixels = {self.max_pixels})'
        return repr_str

class GaussianBlur:
    def __init__(self, kernel_size = 5, sigma = 1):
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.condition = 'min(results["img_shape"])>100'

    def __call__(self, results: Dict) -> Dict:
        if eval(self.condition):
            img = results['img'][..., ::-1]
            img = cv2.GaussianBlur(img, (self.kernel_size, self.kernel_size), self.sigma)
            img = img[..., ::-1]
            results['img'] = img
            results['img_shape'] = img.shape[:2]
            return results
        else:
            return results

class ColorJitter:
    def __init__(self, brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, results: Dict) -> Dict:
        img = results['img'][...,::-1]                
        if self.brightness > 0:
            img = self.adjust_brightness(img, self.brightness)
        if self.contrast > 0:
            img = self.adjust_contrast(img, self.contrast)
        if self.saturation > 0:
            img = self.adjust_saturation(img, self.saturation)
        if self.hue > 0:
            img = self.adjust_hue(img, self.hue)
        img = img[...,::-1]
        results['img'] = img
        results['img_shape'] = img.shape[:2]
        return results

    def adjust_brightness(self, img, factor):
        factor = random.uniform(max(0, 1 - factor), 1 + factor)
        return cv2.convertScaleAbs(img, alpha=factor, beta=0)

    def adjust_contrast(self, img, factor):
        factor = random.uniform(max(0, 1 - factor), 1 + factor)
        return cv2.convertScaleAbs(img, alpha=factor, beta=0)

    def adjust_saturation(self, img, factor):
        factor = random.uniform(max(0, 1 - factor), 1 + factor)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[..., 1] = cv2.convertScaleAbs(hsv[..., 1], alpha=factor, beta=0)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def adjust_hue(self, img, factor):
        factor = random.uniform(-factor, factor)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[..., 0] = (hsv[..., 0].astype(int) + int(factor * 180)) % 180
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

class ImageContentJitter:
    def __call__(self, results: Dict) -> Dict:
        h, w = results['img'].shape[:2]
        img = results['img'].copy()
        if h > 10 and w > 10:
            thres = min(h, w)
            jitter_range = int(random.random() * thres * 0.01)
            for i in range(jitter_range):
                img[i:, i:, :] = img[:h - i, :w - i, :]
        results['img'] = img
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += '()'
        return repr_str

class AdditiveGaussianNoise:
    def __init__(self, scale=0.1):
        self.scale = scale

    def __call__(self, results: Dict) -> Dict:
        img = results['img'].copy()

        noise = np.random.normal(loc=0, scale=eval(self.scale), size=img.shape)
        noisy_img = img + noise
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
        results['img'] = noisy_img
        results['img_shape'] = noisy_img.shape[:2]
        return results

class ReversePixels:
    def __call__(self, results: Dict) -> Dict:
        results['img'] = 255. - results['img'].copy()
        return results

class Resize:
    def __init__(self, scale=(256,64)):
        self.scale = scale

    def __call__(self, results: Dict) -> Dict:
        results['scale'] = self.scale

        results = self._resize_img(results)

        #ToDo
        #self._resize_bboxes(results)
        #self._resize_seg(results)
        #self._resize_keypoints(results)

        return results
    
    def _resize_img(self, results: Dict) -> Dict:
        h, w = results['img'].shape[:2]
        resized_img = cv2.resize(results['img'], results['scale'], interpolation=cv2.INTER_LINEAR)        
        w_scale = results['scale'][0] / w
        h_scale = results['scale'][1] / h

        results['img'] = resized_img
        results['img_shape'] = resized_img.shape[:2]
        results['scale_factor'] = (w_scale, h_scale)
        results['keep_ratio'] = False
        
        return results