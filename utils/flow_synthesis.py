import os
import math
import random
import numpy as np 
from PIL import Image
import matplotlib.pyplot as plt 
from utils.flow_display import dense_flow, sparse_flow
from utils.image_warp import image_warp

mina = 0.000000001
def gen_line(p1, p2):
    denominator = p2[1] - p1[1]
    if denominator == 0:
        denominator = mina
    k = (p2[0] - p1[0]) / denominator
    b = p1[0] - (k * p1[1])
    return k, b

def func_line(x, k, b):
    y = k * x + b
    return y

def gen_flow(height, width, k, b, line_width=5, fold_width=10, dis_k=0.1):
    grid_x = np.tile(np.expand_dims(np.arange(width), 0), [height, 1])
    grid_y = np.tile(np.expand_dims(np.arange(height), 1), [1, width])
    pos_x = grid_x.flatten()
    pos_y = grid_y.flatten()
    dis = (k * pos_x - pos_y + b) / (math.sqrt(k ** 2 + 1))
    # dis = 1 / dis
    dis = dis.reshape((height, width))
    sign = np.zeros_like(dis)
    mask = np.zeros_like(dis)
    sign[dis > 0] = 1
    sign[dis < 0] = -1

    dis_abs = np.abs(dis)
    mask[dis_abs <= line_width] = 0
    mask[dis_abs > line_width] = 1

    # dis = dis * 10
    # max_dis = np.max(dis)
    # min_dis = np.min(dis)
    # print(max_dis, min_dis)
    # dis = (max_dis - dis) * sign + (min_dis - dis) * (1 - sign)

    mask_dis = np.ones_like(dis)
    mask_dis2 = np.ones_like(dis)
    dis_width = fold_width - line_width
    mask_dis[dis_abs < line_width] = 0
    mask_dis[dis_abs >= line_width] = 1
    mask_dis2[dis_abs < fold_width] = 0
    mask_dis2[dis_abs >= fold_width] = 1
    
    # dis_abs[dis_abs >= dis_width] = fold_width
    dis_abs_s = np.zeros_like(dis_abs)
    dis_abs_s2 = np.zeros_like(dis_abs)
    dis_k = -dis_k
    dis_b = dis_width - dis_k * line_width
    dis_abs_s = dis_k * dis_abs + dis_b
    dis_abs_s[dis_abs_s < 0] = 0
    dis_abs_s2 = dis_abs_s * mask_dis2 + dis_abs * (1 - mask_dis2)
    dis_abs_s = dis_abs_s * mask_dis + dis_abs * (1 - mask_dis)

    dis = dis_abs_s * sign
    dis2 = dis_abs_s2 * (-sign)

    if k == 0:
        k_T = 1 / mina
    else:
        k_T = 1 / k

    angle = angle = math.atan(k_T)
    sin_p = math.sin(angle)
    cos_p = math.cos(angle)

    flow = np.zeros((height, width, 2), dtype=np.float32)
    flow2 = np.zeros((height, width, 2), dtype=np.float32)
    if k > 0:
        flow[:, :, 0] = (dis * cos_p)
        flow[:, :, 1] = -(dis * sin_p)
        flow2[:, :, 0] = (dis2 * cos_p)
        flow2[:, :, 1] = -(dis2 * sin_p)
    else:
        flow[:, :, 0] =  -(dis * cos_p)
        flow[:, :, 1] = (dis * sin_p)
        flow2[:, :, 0] =  -(dis2 * cos_p)
        flow2[:, :, 1] = (dis2 * sin_p)

    # print(np.max(flow), np.min(flow))
    return flow, flow2, mask

if __name__ == "__main__":
    height = 256
    width = 256
    # line_width = 10
    # fold_width = 20
    for kkk in range(100):
        line_width = random.randint(5, 20)
        fold_width = random.randint(line_width+1, 80)
        
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

        # p1 = [0, 128]
        # p2 = [128, 256]
        
        # dis_k = random.uniform(0.001, 0.1)
        dis_k = random.uniform(0.00001, 0.1)
        k, b = gen_line(p1, p2)
        
        flow, flow2, mask = gen_flow(height, width, k, b, line_width, fold_width, dis_k)
        # flow = flow * 10
        # print(flow[:10,-10:,0])
        # print(flow[:10,-10:,1])
        flow_show1 = dense_flow(flow)
        flow_show2 = dense_flow(flow2)
        flow_show = np.concatenate([flow_show1, flow_show2], axis=1)
        Image.fromarray(flow_show).save('./temp/flow_'+str(kkk).zfill(4)+'.png')
        # sparse_flow(flow2, stride=10)
        

        # img = np.asarray(Image.open('./0000.png'))
        # img = img[:256, :256]

        # deformed = image_warp(img, flow, mode='bilinear')  # nearest or bilinear
        # deformed = (deformed * mask).astype(np.uint8)
        # Image.fromarray(img).save('./deformed1.png')
        # Image.fromarray(deformed).save('./deformed2.png')
