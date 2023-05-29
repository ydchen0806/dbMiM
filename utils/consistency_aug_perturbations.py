import cv2
import torch
import random
import numpy as np
import torch.nn.functional as F
from skimage import filters
from scipy.ndimage.filters import gaussian_filter

from utils.augmentation import create_identity_transformation
from utils.augmentation import create_elastic_transformation
from utils.augmentation import apply_transformation
from utils.augmentation import misalign
from utils.flow_synthesis import gen_line, gen_flow
from utils.image_warp import image_warp

def simple_augment(data, rule):
    assert np.size(rule) == 4
    assert data.ndim == 3
    # z reflection
    if rule[0]:
        data = data[::-1, :, :]
    # x reflection
    if rule[1]:
        data = data[:, :, ::-1]
    # y reflection
    if rule[2]:
        data = data[:, ::-1, :]
    # transpose in xy
    if rule[3]:
        data = np.transpose(data, (0, 2, 1))
    return data

def simple_augment_torch(data, rule):
    assert np.size(rule) == 4
    assert len(data.shape) == 4
    # z reflection
    if rule[0]:
        data = torch.flip(data, [1])
    # x reflection
    if rule[1]:
        data = torch.flip(data, [3])
    # y reflection
    if rule[2]:
        data = torch.flip(data, [2])
    # transpose in xy
    if rule[3]:
        data = data.permute(0, 1, 3, 2)
    return data

def simple_augment_reverse(data, rule):
    assert np.size(rule) == 4
    assert len(data.shape) == 5
    # transpose in xy
    if rule[3]:
        # data = np.transpose(data, (0, 1, 2, 4, 3))
        data = data.permute(0, 1, 2, 4, 3)
    # y reflection
    if rule[2]:
        # data = data[:, :, :, ::-1, :]
        data = torch.flip(data, [3])
    # x reflection
    if rule[1]:
        # data = data[:, :, :, :, ::-1]
        data = torch.flip(data, [4])
    # z reflection
    if rule[0]:
        # data = data[:, :, ::-1, :, :]
        data = torch.flip(data, [2])
    return data

def order_aug(imgs, num_patch=4):
    assert imgs.shape[-1] % num_patch == 0
    patch_size = imgs.shape[-1] // num_patch
    new_imgs = np.zeros_like(imgs, dtype=np.float32)
    # ran_order = np.random.shuffle(np.arange(num_patch**2))
    ran_order = np.random.permutation(num_patch**2)
    for k in range(num_patch**2):
        xid_new = k // num_patch
        yid_new = k % num_patch
        order_id = ran_order[k]
        xid_old = order_id // num_patch
        yid_old = order_id % num_patch
        new_imgs[:, xid_new*patch_size:(xid_new+1)*patch_size, yid_new*patch_size:(yid_new+1)*patch_size] = \
            imgs[:, xid_old*patch_size:(xid_old+1)*patch_size, yid_old*patch_size:(yid_old+1)*patch_size]
    return new_imgs

def gen_mask(imgs, net_crop_size=[0,0,0], mask_counts=80, mask_size_z=8, mask_size_xy=15):
    crop_size = list(imgs.shape)
    mask = np.ones_like(imgs, dtype=np.float32)
    for k in range(mask_counts):
        mz = random.randint(net_crop_size[0], crop_size[0]-mask_size_z-net_crop_size[0])
        my = random.randint(net_crop_size[1], crop_size[1]-mask_size_xy-net_crop_size[1])
        mx = random.randint(net_crop_size[2], crop_size[2]-mask_size_xy-net_crop_size[2])
        mask[mz:mz+mask_size_z, my:my+mask_size_xy, mx:mx+mask_size_xy] = 0
    return mask

def resize_3d(imgs, det_size, mode='linear'):
    new_imgs = []
    for k in range(imgs.shape[0]):
        temp = imgs[k]
        if mode == 'linear':
            temp = cv2.resize(temp, (det_size, det_size), interpolation=cv2.INTER_LINEAR)
        elif mode == 'nearest':
            temp = cv2.resize(temp, (det_size, det_size), interpolation=cv2.INTER_NEAREST)
        else:
            raise AttributeError('No this interpolation mode!')
        new_imgs.append(temp)
    new_imgs = np.asarray(new_imgs)
    return new_imgs

def add_gauss_noise(imgs, std=0.01, norm_mode='norm'):
    gaussian = np.random.normal(0, std, (imgs.shape))
    imgs = imgs + gaussian
    if norm_mode == 'norm':
        imgs = (imgs-np.min(imgs)) / (np.max(imgs)-np.min(imgs))
    elif norm_mode == 'trunc':
        imgs[imgs<0] = 0
        imgs[imgs>1] = 1
    else:
        raise NotImplementedError
    return imgs

def add_gauss_blur(imgs, kernel_size=5, sigma=0):
    outs = []
    for k in range(imgs.shape[0]):
        temp = imgs[k]
        temp = cv2.GaussianBlur(temp, (kernel_size,kernel_size), sigma)
        outs.append(temp)
    outs = np.asarray(outs, dtype=np.float32)
    outs[outs < 0] = 0
    outs[outs > 1] = 1
    return outs

def add_sobel(imgs, if_mean=False):
    outs = []
    for k in range(imgs.shape[0]):
        temp = imgs[k]
        # sobelx = cv2.Sobel(temp, cv2.CV_32F, 1, 0)
        # sobely = cv2.Sobel(temp, cv2.CV_32F, 0, 1)
        # sobelx = filters.sobel_h(temp)
        # sobely = filters.sobel_v(temp)
        # dst = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
        # dst = sobelx * 0.5 + sobely * 0.5
        # dst = cv2.Sobel(temp, cv2.CV_32F, 1, 1)
        if if_mean:
            mean = np.mean(temp)
        else:
            mean = 0
        dst = filters.sobel(temp) + mean
        outs.append(dst)
    outs = np.asarray(outs, dtype=np.float32)
    outs[outs < 0] = 0
    outs[outs > 1] = 1
    return outs

def add_intensity(imgs, contrast_factor=0.1, brightness_factor=0.1):
    # imgs *= 1 + (np.random.rand() - 0.5) * contrast_factor
    # imgs += (np.random.rand() - 0.5) * brightness_factor
    # imgs = np.clip(imgs, 0, 1)
    # imgs **= 2.0**(np.random.rand()*2 - 1)
    imgs *= 1 + contrast_factor
    imgs += brightness_factor
    imgs = np.clip(imgs, 0, 1)
    return imgs

def interp_5d(data, det_size, mode='bilinear'):
    assert len(data.shape) == 5, "the dimension of data must be 5!"
    out = []
    depth = data.shape[2]
    for k in range(depth):
        temp = data[:,:,k,:,:]
        if mode == 'bilinear':
            temp = F.interpolate(temp, size=(det_size, det_size), mode='bilinear', align_corners=True)
        elif mode == 'nearest':
            temp = F.interpolate(temp, size=(det_size, det_size), mode='nearest')
        out.append(temp)
    out = torch.stack(out, dim=2)
    return out

def convert_consistency_scale(gt, det_size):
    B, C, D, H, W = gt.shape
    gt = gt.detach().clone()
    out_gt = []
    masks = []
    for k in range(B):
        gt_temp = gt[k]
        det_size_temp = det_size[k]
        if det_size_temp[0] == gt_temp.shape[-1]:
            mask = torch.ones_like(gt_temp)
            out_gt.append(gt_temp)
            masks.append(mask)
        elif det_size_temp[0] > gt_temp.shape[-1]:
            shift = int((det_size_temp[0] - gt_temp.shape[-1]) // 2)
            gt_padding = torch.zeros((1, C, D, int(det_size_temp[0]), int(det_size_temp[0]))).float().cuda()
            mask = torch.zeros_like(gt_padding)
            gt_padding[0,:,:,shift:-shift,shift:-shift] = gt_temp
            mask[0,:,:,shift:-shift,shift:-shift] = 1
            # gt_padding = F.interpolate(gt_padding, size=(D, int(gt_temp.shape[-1]), int(gt_temp.shape[-1])), mode='trilinear', align_corners=True)
            gt_padding = interp_5d(gt_padding, int(gt_temp.shape[-1]), mode='bilinear')
            mask = F.interpolate(mask, size=(D, int(gt_temp.shape[-1]), int(gt_temp.shape[-1])), mode='nearest')
            gt_padding = torch.squeeze(gt_padding, dim=0)
            mask = torch.squeeze(mask, dim=0)
            out_gt.append(gt_padding)
            masks.append(mask)
        else:
            shift = int((gt_temp.shape[-1] - det_size_temp[0]) // 2)
            mask = torch.zeros_like(gt_temp)
            mask[:,:,shift:-shift,shift:-shift] = 1
            gt_padding = gt_temp[:,:,shift:-shift,shift:-shift]
            gt_padding = gt_padding[None, ...]
            # gt_padding = F.interpolate(gt_padding, size=(D, int(gt_temp.shape[-1]), int(gt_temp.shape[-1])), mode='trilinear', align_corners=True)
            gt_padding = interp_5d(gt_padding, int(gt_temp.shape[-1]), mode='bilinear')
            gt_padding = torch.squeeze(gt_padding, dim=0)
            out_gt.append(gt_padding)
            masks.append(mask)
    out_gt = torch.stack(out_gt, dim=0)
    masks = torch.stack(masks, dim=0)
    return out_gt, masks

def convert_consistency_flip(gt, rules):
    B, C, D, H, W = gt.shape
    gt = gt.detach().clone()
    rules = rules.data.cpu().numpy()
    out_gt = []
    for k in range(B):
        gt_temp = gt[k]
        rule = rules[k]
        gt_temp = simple_augment_torch(gt_temp, rule)
        out_gt.append(gt_temp)
    out_gt = torch.stack(out_gt, dim=0)
    return out_gt


class Rescale(object):
    def __init__(self, scale_factor=2, det_shape=[18, 160, 160]):
        super(Rescale, self).__init__()
        self.scale_factor = scale_factor
        self.det_shape = det_shape

    def __call__(self, data):
        src_shape = data.shape
        assert src_shape[-1] >= self.det_shape[-1] * self.scale_factor, 'data shape must be 160*2'
        min_size = self.det_shape[-1] // self.scale_factor
        max_size = self.det_shape[-1] * self.scale_factor
        scale_size = random.randint(min_size // 2, max_size // 2)
        scale_size = scale_size * 2

        if scale_size < src_shape[-1]:
            shift = (src_shape[-1] - scale_size) // 2
            data = data[:, shift:-shift, shift:-shift]
        data = resize_3d(data, self.det_shape[-1], mode='linear')
        return data, scale_size


class Filp(object):
    def __init__(self):
        super(Filp, self).__init__()

    def __call__(self, data):
        rule = np.random.randint(2, size=4)
        data = simple_augment(data, rule)
        return data, rule


# class Intensity(object):
#     def __init__(self, contrast_factor=0.3, brightness_factor=0.3):
#         super(Intensity, self).__init__()
#         self.CONTRAST_FACTOR   = contrast_factor
#         self.BRIGHTNESS_FACTOR = brightness_factor

#     def __call__(self, data):
#         data = self._augment3D(data)
#         return data

#     def _augment3D(self, data, random_state=np.random):
#         """
#         Adapted from ELEKTRONN (http://elektronn.org/).
#         """
#         ran = random_state.rand(3)

#         transformedimgs = np.copy(data)
#         transformedimgs *= 1 + (ran[0] - 0.5)*self.CONTRAST_FACTOR
#         transformedimgs += (ran[1] - 0.5)*self.BRIGHTNESS_FACTOR
#         transformedimgs = np.clip(transformedimgs, 0, 1)
#         transformedimgs **= 2.0**(ran[2]*2 - 1)
        
#         return transformedimgs
class Intensity(object):
    def __init__(self, mode='mix',
                        skip_ratio=0.5,
                        CONTRAST_FACTOR=0.1,
                        BRIGHTNESS_FACTOR=0.1):
        '''Image intensity augmentation, including adjusting contrast and brightness
        Args:
            mode: '2D', '3D' or 'mix' (contains '2D' and '3D')
            skip_ratio: Probability of execution
            CONTRAST_FACTOR: Contrast factor
            BRIGHTNESS_FACTOR : Brightness factor
        '''
        super(Intensity, self).__init__()
        assert mode == '3D' or mode == '2D' or mode == 'mix'
        self.mode = mode
        self.ratio = skip_ratio
        self.CONTRAST_FACTOR = CONTRAST_FACTOR
        self.BRIGHTNESS_FACTOR = BRIGHTNESS_FACTOR
    
    def __call__(self, inputs):
        return self.forward(inputs)
    
    def forward(self, inputs):
        inputs = inputs.copy()
        skiprand = np.random.rand()
        if self.mode == 'mix':
            # The probability of '2D' is more than '3D'
            threshold = 1 - (1 - self.ratio) / 2
            mode_ = '3D' if skiprand > threshold else '2D'
        else:
            mode_ = self.mode
        if mode_ == '2D':
            inputs = self.augment2D(inputs)
        elif mode_ == '3D':
            inputs = self.augment3D(inputs)
        inputs[inputs<0] = 0
        inputs[inputs>1] = 1
        return inputs
    
    def augment2D(self, imgs):
        for z in range(imgs.shape[-3]):
            img = imgs[z, :, :]
            img *= 1 + (np.random.rand() - 0.5)*self.CONTRAST_FACTOR
            img += (np.random.rand() - 0.5)*self.BRIGHTNESS_FACTOR
            img = np.clip(img, 0, 1)
            img **= 2.0**(np.random.rand()*2 - 1)
            imgs[z, :, :] = img
        return imgs
    
    def augment3D(self, imgs):
        imgs *= 1 + (np.random.rand() - 0.5)*self.CONTRAST_FACTOR
        imgs += (np.random.rand() - 0.5)*self.BRIGHTNESS_FACTOR
        imgs = np.clip(imgs, 0, 1)
        imgs **= 2.0**(np.random.rand()*2 - 1)
        return imgs


class GaussBlur(object):
    def __init__(self, min_kernel=3, max_kernel=9, min_sigma=0, max_sigma=2):
        super(GaussBlur, self).__init__()
        self.min_kernel = min_kernel
        self.max_kernel = max_kernel
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma

    def __call__(self, data):
        kernel_size = random.randint(self.min_kernel // 2, self.max_kernel // 2)
        kernel_size = kernel_size * 2 + 1
        sigma = random.uniform(self.min_sigma, self.max_sigma)
        data = add_gauss_blur(data, kernel_size=kernel_size, sigma=sigma)
        return data


class GaussNoise(object):
    def __init__(self, min_std=0.01, max_std=0.2, norm_mode='trunc'):
        super(GaussNoise, self).__init__()
        self.min_std = min_std
        self.max_std = max_std
        self.norm_mode = norm_mode

    def __call__(self, data):
        std = random.uniform(self.min_std, self.max_std)
        data = add_gauss_noise(data, std=std, norm_mode=self.norm_mode)
        return data


class Cutout(object):
    def __init__(self, model_type='superhuman'):
        super(Cutout, self).__init__()
        self.model_type = model_type
        # mask size
        if self.model_type == 'mala':
            self.min_mask_size = [5, 5, 5]
            self.max_mask_size = [8, 12, 12]
            self.min_mask_counts = 40
            self.max_mask_counts = 60
            self.net_crop_size = [14, 106, 106]
        else:
            self.min_mask_size = [5, 10, 10]
            self.max_mask_size = [10, 20, 20]
            self.min_mask_counts = 60
            self.max_mask_counts = 100
            self.net_crop_size = [0, 0, 0]

    def __call__(self, data):
        mask_counts = random.randint(self.min_mask_counts, self.max_mask_counts)
        mask_size_z = random.randint(self.min_mask_size[0], self.max_mask_size[0])
        mask_size_xy = random.randint(self.min_mask_size[1], self.max_mask_size[1])
        mask = gen_mask(data, net_crop_size=self.net_crop_size, \
                        mask_counts=mask_counts, \
                        mask_size_z=mask_size_z, \
                        mask_size_xy=mask_size_xy)
        data = data * mask
        return data


class SobelFilter(object):
    def __init__(self, if_mean=False):
        super(SobelFilter, self).__init__()
        self.if_mean = if_mean

    def __call__(self, data):
        data = add_sobel(data, if_mean=self.if_mean)
        return data


class Mixup(object):
    def __init__(self, min_alpha=0.01, max_alpha=0.1):
        super(Mixup, self).__init__()
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha

    def __call__(self, data, auxi):
        alpha = random.uniform(self.min_alpha, self.max_alpha)
        data = auxi * alpha + data * (1 - alpha)
        data[data<0] = 0
        data[data>1] = 1
        return data


class Missing(object):
    '''Missing section augmentation
    Args:
        filling: the way of filling, 'zero' or 'random'
        mode: 'mix', 'fully' or 'partially'
        skip_ratio: Probability of execution
        miss_ratio: Probability of missing
    '''
    def __init__(self, filling='zero', mode='mix', skip_ratio=0.5, miss_fully_ratio=0.2, miss_part_ratio=0.5):
        super(Missing, self).__init__()
        self.filling = filling
        self.mode = mode
        self.ratio = skip_ratio
        self.miss_fully_ratio = miss_fully_ratio
        self.miss_part_ratio = miss_part_ratio
    
    def __call__(self, imgs):
        return self.forward(imgs)

    def forward(self, imgs):
        imgs = imgs.copy()
        if self.mode == 'mix':
            r = np.random.rand()
            mode_ = 'fully' if r < 0.5 else 'partially'
        else:
            mode_ = self.mode
        if mode_ == 'fully':
            imgs = self.augment_fully(imgs)
        elif mode_ == 'partially':
            imgs = self.augment_partially(imgs)
        return imgs
    
    def augment_fully(self, imgs):
        d, h, w = imgs.shape
        for i in range(d):
            if np.random.rand() < self.miss_fully_ratio:
                if self.filling == 'zero':
                    imgs[i] = 0
                elif self.filling == 'random':
                    imgs[i] = np.random.rand(h, w)
        return imgs
    
    def augment_partially(self, imgs, size_ratio=0.3):
        d, h, w = imgs.shape
        for i in range(d):
            if np.random.rand() < self.miss_part_ratio:
                # randomly generate an area
                sub_h = random.randint(int(h*size_ratio), int(h*(1-size_ratio)))
                sub_w = random.randint(int(w*size_ratio), int(w*(1-size_ratio)))
                start_h = random.randint(0, h - sub_h - 1)
                start_w = random.randint(0, w - sub_w - 1)
                if self.filling == 'zero':
                    imgs[i, start_h:start_h+sub_h, start_w:start_w+sub_w] = 0
                elif self.filling == 'random':
                    imgs[i, start_h:start_h+sub_h, start_w:start_w+sub_w] = np.random.rand(sub_h, sub_w)
        return imgs


class BlurEnhanced(object):
    '''Out-of-focus (Blur) section augmentation
    Args:
        mode: 'mix', 'fully' or 'partially'
        skip_ratio: Probability of execution
        blur_ratio: Probability of blur
    '''
    def __init__(self, mode='mix', skip_ratio=0.5, blur_fully_ratio=0.5, blur_part_ratio=0.7):
        super(BlurEnhanced, self).__init__()
        self.mode = mode
        self.ratio = skip_ratio
        self.blur_fully_ratio = blur_fully_ratio
        self.blur_part_ratio = blur_part_ratio
    
    def __call__(self, imgs):
        return self.forward(imgs)

    def forward(self, imgs):
        imgs = imgs.copy()
        if self.mode == 'mix':
            r = np.random.rand()
            mode_ = 'fully' if r < 0.5 else 'partially'
        else:
            mode_ = self.mode
        if mode_ == 'fully':
            imgs = self.augment_fully(imgs)
        elif mode_ == 'partially':
            imgs = self.augment_partially(imgs)
        return imgs
    
    def augment_fully(self, imgs):
        d, h, w = imgs.shape
        for i in range(d):
            if np.random.rand() < self.blur_fully_ratio:
                sigma = np.random.uniform(0, 5)
                imgs[i] = gaussian_filter(imgs[i], sigma)
        return imgs
    
    def augment_partially(self, imgs, size_ratio=0.3):
        d, h, w = imgs.shape
        for i in range(d):
            if np.random.rand() < self.blur_part_ratio:
                # randomly generate an area
                sub_h = random.randint(int(h*size_ratio), int(h*(1-size_ratio)))
                sub_w = random.randint(int(w*size_ratio), int(w*(1-size_ratio)))
                start_h = random.randint(0, h - sub_h - 1)
                start_w = random.randint(0, w - sub_w - 1)
                sigma = np.random.uniform(0, 5)
                imgs[i, start_h:start_h+sub_h, start_w:start_w+sub_w] = \
                    gaussian_filter(imgs[i, start_h:start_h+sub_h, start_w:start_w+sub_w], sigma)
        return imgs


class Elastic(object):
    '''Elasticly deform a batch. Requests larger batches upstream to avoid data 
    loss due to rotation and jitter.
    Args:
        control_point_spacing (``tuple`` of ``int``):
            Distance between control points for the elastic deformation, in
            voxels per dimension.
        jitter_sigma (``tuple`` of ``float``):
            Standard deviation of control point jitter distribution, in voxels
            per dimension.
        rotation_interval (``tuple`` of two ``floats``):
            Interval to randomly sample rotation angles from (0, 2PI).
        prob_slip (``float``):
            Probability of a section to "slip", i.e., be independently moved in
            x-y.
        prob_shift (``float``):
            Probability of a section and all following sections to move in x-y.
        max_misalign (``int``):
            Maximal voxels to shift in x and y. Samples will be drawn
            uniformly. Used if ``prob_slip + prob_shift`` > 0.
        subsample (``int``):
            Instead of creating an elastic transformation on the full
            resolution, create one subsampled by the given factor, and linearly
            interpolate to obtain the full resolution transformation. This can
            significantly speed up this node, at the expense of having visible
            piecewise linear deformations for large factors. Usually, a factor
            of 4 can savely by used without noticable changes. However, the
            default is 1 (i.e., no subsampling).
    '''
    def __init__(
            self,
            control_point_spacing=[4, 40, 40],
            jitter_sigma=[0,0,0],   # recommend: [0, 2, 2]
            rotation_interval=[0,0],
            prob_slip=0,   # recommend: 0.05
            prob_shift=0,   # recommend: 0.05
            max_misalign=0,   # 17 in superhuman
            subsample=1,
            padding=None,
            skip_ratio=0.5):   # recommend: 10
        super(Elastic, self).__init__()

        self.control_point_spacing = control_point_spacing
        self.jitter_sigma = jitter_sigma
        self.rotation_start = rotation_interval[0]
        self.rotation_max_amount = rotation_interval[1] - rotation_interval[0]
        self.prob_slip = prob_slip
        self.prob_shift = prob_shift
        self.max_misalign = max_misalign
        self.subsample = subsample
        self.padding = padding
        self.ratio = skip_ratio

    def create_transformation(self, target_shape):
        transformation = create_identity_transformation(
            target_shape,
            subsample=self.subsample)
        # shape: channel,d,w,h

        # elastic  ##cost time##
        if sum(self.jitter_sigma) > 0:
            transformation += create_elastic_transformation(
                target_shape,
                self.control_point_spacing,
                self.jitter_sigma,
                subsample=self.subsample)

        # rotation = random.random()*self.rotation_max_amount + self.rotation_start
        # if rotation != 0:
        #     transformation += create_rotation_transformation(
        #         target_shape,
        #         rotation,
        #         subsample=self.subsample)

        # if self.subsample > 1:
        #     transformation = upscale_transformation(
        #         transformation,
        #         tuple(target_shape))

        if self.prob_slip + self.prob_shift > 0:
            misalign(transformation, self.prob_slip,
                     self.prob_shift, self.max_misalign)

        return transformation

    def __call__(self, imgs):
        return self.forward(imgs)

    def forward(self, imgs):
        '''Args:
            imgs: numpy array, [Z, Y, Z], it always is float and 0~1
            mask: numpy array, [Z, Y, Z], it always is uint16
        '''
        imgs = imgs.copy()
        if self.padding is not None:
            imgs = np.pad(imgs, ((0,0), \
                                (self.padding,self.padding), \
                                (self.padding,self.padding)), mode='reflect')
        transform = self.create_transformation(imgs.shape)
        img_transform = apply_transformation(imgs,
                                         transform,
                                         interpolate=False,
                                         outside_value=0,  # imgs.dtype.type(-1)
                                         output=np.zeros(imgs.shape, dtype=np.float32))
        # seg_transform[seg_transform < 0] = 0
        # seg_transform[seg_transform > 60000] = 0
        if self.padding is not None and self.padding != 0:
            img_transform = img_transform[:, self.padding:-self.padding, self.padding:-self.padding]
        return img_transform


class Artifact(object):
    def __init__(self, min_sec=1, max_sec=5):
        super(Artifact, self).__init__()
        self.min_sec = min_sec
        self.max_sec = max_sec
        self.offset = 40

    def __call__(self, data):
        data = data.copy()
        num_sec = random.randint(self.min_sec, self.max_sec)
        num_imgs = data.shape[0]
        rand_sample = random.sample(range(num_imgs), num_sec)
        for k in rand_sample:
            tmp = data[k].copy()
            tmp = (tmp * 255).astype(np.uint8)
            tmp = self.degradation(tmp)
            data[k] = tmp.astype(np.float32) / 255.0
        return data

    def degradation(self, img):
        img = np.pad(img, ((self.offset,self.offset),(self.offset,self.offset)), mode='reflect')
        height, width = img.shape
        line_width = random.randint(5, 10)
        fold_width = random.randint(line_width+1, 40)

        # two end points
        # 1 --> top line (0, x)
        # 2 --> right line (x, width)
        # 3 --> bottom line (height, x)
        # 4 --> left line (x, 0)
        k1 = random.randint(1, 4)
        k2 = random.randint(1, 4)
        while k1 == k2:
            k2 = random.randint(1, 4)
        
        if k1 == 1:
            x = random.randint(1, width-1)
            p1 = [0, x]
        elif k1 == 2:
            x = random.randint(1, height-1)
            p1 = [x, width]
        elif k1 == 3:
            x = random.randint(1, width-1)
            p1 = [height, x]
        else:
            x = random.randint(1, height-1)
            p1 = [x, 0]
        
        if k2 == 1:
            x = random.randint(1, width-1)
            p2 = [0, x]
        elif k2 == 2:
            x = random.randint(1, height-1)
            p2 = [x, width]
        elif k2 == 3:
            x = random.randint(1, width-1)
            p2 = [height, x]
        else:
            x = random.randint(1, height-1)
            p2 = [x, 0]
        
        dis_k = random.uniform(0.00001, 0.1)
        k, b = gen_line(p1, p2)
        flow, flow2, mask = gen_flow(height, width, k, b, line_width, fold_width, dis_k)

        deformed = image_warp(img, flow, mode='bilinear')  # nearest or bilinear
        deformed = (deformed * mask).astype(np.uint8)
        deformed = deformed[self.offset:-self.offset, self.offset:-self.offset]

        return deformed
