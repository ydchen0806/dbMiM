import torch
import random
import numpy as np
import subprocess

def center_crop(image, det_shape=[18, 160, 160]):
    # To prevent overflow
    image = np.pad(image, ((2,2),(20,20),(20,20)), mode='reflect')
    src_shape = image.shape
    shift0 = (src_shape[0] - det_shape[0]) // 2
    shift1 = (src_shape[1] - det_shape[1]) // 2
    shift2 = (src_shape[2] - det_shape[2]) // 2
    assert shift0 > 0 or shift1 > 0 or shift2 > 0, "overflow in center-crop"
    image = image[shift0:shift0+det_shape[0], shift1:shift1+det_shape[1], shift2:shift2+det_shape[2]]
    return image

def setup_seed(seed):
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def execute(cmd):
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)

def compute_num_single(size, stride):
    # 计算大概需要滑窗的个数
    num_window = size // stride
    # 判断是否能整除，如果不可以就+1
    # 即滑窗个数加1
    if size % stride != 0:
        num_window += 1
    # 计算padding的大小
    padding_2times = num_window * stride - size + (stride * 2)
    # 如果padding的大小不能被2整除
    # 就继续增加滑窗的个数，已使得padding_2times能被2整除
    while padding_2times % 2 != 0:
        num_window += 1
        padding_2times = num_window * stride - size + (stride * 2)
    # 除以2是为了对称padding
    padding = padding_2times // 2
    # 加1是为了补上最后一个完整的滑窗
    num_window += 1
    return num_window, padding

def compute_num(raw_shape, stride):
    size_z = raw_shape[0]
    size_xy = raw_shape[1]
    stride_z = stride[0]
    stride_xy = stride[1]
    num_z, padding_z = compute_num_single(size_z, stride_z)
    num_xy, padding_xy = compute_num_single(size_xy, stride_xy)
    return [num_z, num_xy, num_xy], [padding_z, padding_xy, padding_xy]

if __name__ == "__main__":
    raw = [500, 4096, 4096]
    stride = [18, 128, 128]
    num, padding = compute_num(raw, stride)
    print(num)
    print(padding)
