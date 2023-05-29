import os
import math
import numpy as np 
from PIL import Image
from matplotlib import pyplot as plt

def show(img3d):
    # only used for image with shape [18, 160, 160, 3]
    num = img3d.shape[0]
    column = 5
    row = math.ceil(num / float(column))
    size = img3d.shape[1]
    img_all = np.zeros((size*row, size*column, 3), dtype=np.uint8)
    for i in range(row):
        for j in range(column):
            index = i*column + j
            if index >= num:
                img = np.zeros_like(img3d[0], dtype=np.uint8)
            else:
                if np.max(img3d[index]) > 1:
                    img = (img3d[index]).astype(np.uint8)
                else:
                    img = (img3d[index] * 255).astype(np.uint8)
            img_all[i*size:(i+1)*size, j*size:(j+1)*size] = img
    return img_all

def show_one(img3d):
    # only used for image with shape [18, 160, 160]
    num = img3d.shape[0]
    column = 5
    row = math.ceil(num / float(column))
    size = img3d.shape[1]
    img_all = np.zeros((size*row, size*column), dtype=np.uint8)
    for i in range(row):
        for j in range(column):
            index = i*column + j
            if index >= num:
                img = np.zeros_like(img3d[0], dtype=np.uint8)
            else:
                img = (img3d[index] * 255).astype(np.uint8)
            img_all[i*size:(i+1)*size, j*size:(j+1)*size] = img
    return img_all

def show_CE(img3d):
    # only used for image with shape [18, 160, 160]
    num = img3d.shape[0]
    column = 5
    row = math.ceil(num / float(column))
    size = img3d.shape[1]
    img_all = np.zeros((size*row, size*column), dtype=np.uint8)
    for i in range(row):
        for j in range(column):
            index = i*column + j
            if index >= num:
                img = np.zeros_like(img3d[0], dtype=np.uint8)
            else:
                img = (img3d[index]).astype(np.uint8)
            img_all[i*size:(i+1)*size, j*size:(j+1)*size] = img
    return img_all

def training_show(iters, inputs, label, pred_bound, cache_path, if_skele=None, skele=None, pred_skele=None):
    img_input = np.repeat(inputs[0].data.cpu().numpy(), 3, 0)
    img_input = np.transpose(img_input, (1,2,3,0))
    img_input = show(img_input)
    input_placehplder = np.zeros_like(img_input, dtype=np.uint8)
    im_cat1 = np.concatenate([img_input, input_placehplder], axis=1)

    img_label = label[0][0:3].data.cpu().numpy()
    img_label = np.transpose(img_label, (1,2,3,0))
    img_label = show(img_label)

    img_pred_bound = pred_bound[0][0:3].data.cpu().numpy()
    img_pred_bound = np.transpose(img_pred_bound, (1,2,3,0))
    img_pred_bound = show(img_pred_bound)
    im_cat2 = np.concatenate([img_pred_bound, img_label], axis=1)

    if if_skele is not None:
        img_skele = np.repeat(skele[0, 0:1].data.cpu().numpy(), 3, 0)
        img_skele = np.transpose(img_skele, (1,2,3,0))
        img_skele = show(img_skele)

        img_pred_skele = np.repeat(pred_skele[0, 0:1].data.cpu().numpy(), 3, 0)
        img_pred_skele = np.transpose(img_pred_skele, (1,2,3,0))
        img_pred_skele = show(img_pred_skele)
        im_cat3 = np.concatenate([img_pred_skele, img_skele], axis=1)

        im_cat = np.concatenate([im_cat1, im_cat2, im_cat3], axis=0)
    else:
        im_cat = np.concatenate([im_cat1, im_cat2], axis=0)
    Image.fromarray(im_cat).save(os.path.join(cache_path, '%06d.png' % iters))

def training_show_pretrain(iters, pred, label, cache_path, loss_mode='CrossEntropy'):
    img_input = pred[0].data.cpu().numpy()
    if loss_mode == 'CrossEntropy':
        img_input = show_CE(img_input)
    else:
        img_input[img_input < 0] = 0
        img_input[img_input > 1] = 1
        img_input = show_one(img_input)
    img_label = label[0].data.cpu().numpy()
    img_label = show_one(img_label)
    im_cat = np.concatenate([img_input, img_label], axis=1)
    Image.fromarray(im_cat).save(os.path.join(cache_path, '%06d.png' % iters))


def show_inpaining(iters, pred, label, mask, cache_path):
    pred = pred[0].data.cpu().numpy()
    label = label[0].data.cpu().numpy()
    mask = mask[0].data.cpu().numpy()
    inputs = label * mask
    inputs = np.squeeze(inputs)
    pred = np.squeeze(pred)
    inputs = inputs[14:-14, 106:-106, 106:-106]
    pred[pred < 0] = 0; pred[pred > 1] =1
    pred_img = show_one(pred)
    inputs_img = show_one(inputs)
    im_cat = np.concatenate([inputs_img, pred_img], axis=1)
    Image.fromarray(im_cat).save(os.path.join(cache_path, '%06d.png' % iters))


def show_affs(iters, inputs, pred, target, cache_path, model_type='mala'):
    pred = pred[0].data.cpu().numpy()
    inputs = inputs[0].data.cpu().numpy()
    target = target[0].data.cpu().numpy()
    inputs = np.squeeze(inputs)
    if model_type == 'mala':
        inputs = inputs[14:-14, 106:-106, 106:-106]
    inputs = inputs[:,:,:,np.newaxis]
    inputs = np.repeat(inputs, 3, 3)
    pred = np.transpose(pred, (1,2,3,0))
    target = np.transpose(target, (1,2,3,0))
    inputs[inputs<0]=0; inputs[inputs>1]=1
    pred[pred<0]=0; pred[pred>1]=1
    target[target<0]=0; target[target>1]=1
    inputs_img = show(inputs)
    pred_img = show(pred)
    target_img = show(target)
    im_cat = np.concatenate([inputs_img, pred_img, target_img], axis=1)
    Image.fromarray(im_cat).save(os.path.join(cache_path, '%06d.png' % iters))

def save_rec(iters, gt, input1, input2, out1, out2, cache_path, model_type='mala'):
    b,_,z,_,_ = gt.shape
    
    index1 = np.random.randint(0, b)
    if model_type == 'mala':
        input1 = input1[...,14:-14,106:-106,106:-106]
        input2 = input2[...,14:-14,106:-106,106:-106]
    gt = gt[index1].data.cpu().numpy()
    out1 = out1[index1].data.cpu().numpy()
    out2 = out2[index1].data.cpu().numpy()
    input1 = input1[index1].data.cpu().numpy()
    input2 = input2[index1].data.cpu().numpy()
    gt = np.squeeze(gt)
    out1 = np.squeeze(out1)
    out2 = np.squeeze(out2)
    input1 = np.squeeze(input1)
    input2 = np.squeeze(input2)

    index2 = np.random.randint(0, z)
    index3 = np.random.randint(0, z)
    index4 = np.random.randint(0, z)
    index5 = np.random.randint(0, z)
    # print('gt shape: ', gt.shape,'input1 shape: ', input1.shape, 'input2 shape: ', input2.shape, 'out1 shape: ', out1.shape, 'out2 shape: ', out2.shape)
    img1 = np.concatenate([gt[index2], input1[index2], input2[index2], out1[index2], out2[index2]], axis=1)
    img2 = np.concatenate([gt[index3],  input1[index3], input2[index3], out1[index3], out2[index3]], axis=1)
    img3 = np.concatenate([gt[index4],  input1[index4], input2[index4], out1[index4], out2[index4]], axis=1)
    img4 = np.concatenate([gt[index5],  input1[index5], input2[index5], out1[index5], out2[index5]], axis=1)
    im_cat = np.concatenate([img1,img2,img3,img4], axis=0)
    plt.imshow(im_cat, cmap='gray')
    plt.axis('off')
    plt.savefig(os.path.join(cache_path, '%06d.png' % iters), bbox_inches='tight', pad_inches=0)

def class_color(lb):
    d, h, w = lb.shape
    lb_color1 = np.zeros((d, h, w), dtype=np.uint8)
    lb_color2 = np.zeros((d, h, w), dtype=np.uint8)
    lb_color3 = np.zeros((d, h, w), dtype=np.uint8)
    ids_0 = lb == 0
    ids_1 = lb == 1
    lb_color1[ids_0] = 0; lb_color2[ids_0] = 0; lb_color3[ids_0] = 255
    lb_color1[ids_1] = 0; lb_color2[ids_1] = 255; lb_color3[ids_1] = 0
    lb_color = np.concatenate([lb_color1[:,:,:,np.newaxis], lb_color2[:,:,:,np.newaxis], lb_color3[:,:,:,np.newaxis]], axis=3)
    return lb_color


def show_affs_pseudo(iters, inputs, pred, target, mask, cache_path, model_type='mala'):
    pred = pred[0].data.cpu().numpy()
    inputs = inputs[0].data.cpu().numpy()
    target = target[0].data.cpu().numpy()
    mask = mask[0].data.cpu().numpy()
    inputs = np.squeeze(inputs)
    if model_type == 'mala':
        inputs = inputs[14:-14, 106:-106, 106:-106]
    inputs = inputs[:,:,:,np.newaxis]
    inputs = np.repeat(inputs, 3, 3)
    pred = np.transpose(pred, (1,2,3,0))
    target = np.transpose(target, (1,2,3,0))
    affs_z = class_color(target[:, :, :, 0]) * mask[0][:,:,:,np.newaxis]
    affs_y = class_color(target[:, :, :, 1]) * mask[1][:,:,:,np.newaxis]
    affs_x = class_color(target[:, :, :, 2]) * mask[2][:,:,:,np.newaxis]
    inputs_img = show(inputs)
    pred_img = show(pred)
    # target_img = show(target)
    mask = np.transpose(mask, (1,2,3,0))
    mask_img = show(mask)
    affs_z_img = show(affs_z)
    affs_y_img = show(affs_y)
    affs_x_img = show(affs_x)
    # im_cat = np.concatenate([inputs_img, pred_img, target_img], axis=1)
    im_cat1 = np.concatenate([inputs_img, pred_img], axis=1)
    im_cat2 = np.concatenate([mask_img, affs_z_img], axis=1)
    im_cat3 = np.concatenate([affs_y_img, affs_x_img], axis=1)
    im_cat = np.concatenate([im_cat1, im_cat2, im_cat3], axis=0)
    Image.fromarray(im_cat).save(os.path.join(cache_path, '%06d.png' % iters))

def show_affs_whole(iters, out_affs, gt_affs, cache_path):
    out_affs = out_affs[:, -1, ...]
    gt_affs = gt_affs[:, -1, ...]
    out_affs = (out_affs * 255).astype(np.uint8)
    out_affs = np.transpose(out_affs, (1,2,0))
    gt_affs = (gt_affs * 255).astype(np.uint8)
    gt_affs = np.transpose(gt_affs, (1,2,0))
    im_cat = np.concatenate([out_affs, gt_affs], axis=1)
    Image.fromarray(im_cat).save(os.path.join(cache_path, '%06d.png' % iters))

def show_affs_consistency(iters, inputs, pred, target, inputs_u, out_u1, out_u2, cache_path):
    pred = pred[0].data.cpu().numpy()
    inputs = inputs[0].data.cpu().numpy()
    target = target[0].data.cpu().numpy()
    inputs = np.squeeze(inputs)
    inputs = inputs[14:-14, 106:-106, 106:-106]
    inputs = inputs[:,:,:,np.newaxis]
    inputs = np.repeat(inputs, 3, 3)
    pred = np.transpose(pred, (1,2,3,0))
    target = np.transpose(target, (1,2,3,0))
    inputs_img = show(inputs)
    pred_img = show(pred)
    target_img = show(target)
    im_cat1 = np.concatenate([inputs_img, pred_img, target_img], axis=1)

    out_u1 = out_u1[0].data.cpu().numpy()
    inputs_u = inputs_u[0].data.cpu().numpy()
    out_u2 = out_u2[0].data.cpu().numpy()
    inputs_u = np.squeeze(inputs_u)
    inputs_u = inputs_u[14:-14, 106:-106, 106:-106]
    inputs_u = inputs_u[:,:,:,np.newaxis]
    inputs_u = np.repeat(inputs_u, 3, 3)
    out_u1 = np.transpose(out_u1, (1,2,3,0))
    out_u2 = np.transpose(out_u2, (1,2,3,0))
    inputs_u_img = show(inputs_u)
    out_u1_img = show(out_u1)
    out_u2_img = show(out_u2)
    im_cat2 = np.concatenate([inputs_u_img, out_u1_img, out_u2_img], axis=1)
    im_cat = np.concatenate([im_cat1, im_cat2], axis=0)
    Image.fromarray(im_cat).save(os.path.join(cache_path, '%06d.png' % iters))