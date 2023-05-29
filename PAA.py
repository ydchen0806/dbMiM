import kornia
from kornia.augmentation import RandomErasing, RandomSharpness
from kornia.augmentation import RandomPosterize, RandomEqualize, RandomAffine
from kornia.augmentation import RandomRotation,RandomGrayscale, ColorJitter
from kornia.augmentation import Normalize
import numpy as np
import random
import torch
from torch.distributions import Categorical

import sys

from torchvision import transforms
import time

# Augmentation action

normalize_mean = torch.tensor((125.3, 123.0, 113.9))/255.0
normalize_std = torch.tensor((63.0, 62.1, 66.7))/255.0

# randomerasing = RandomErasing(p=1., scale=(0.09, 0.36), ratio=(0.5, 1/0.5), same_on_batch=False)
# sharpness = RandomSharpness(sharpness=0.5, same_on_batch=False)
# # randomerasing = RandomErasing(p=1., scale=(.2, .4), ratio=(.1, 1/.3), same_on_batch=False)
# # sharpness = RandomSharpness(sharpness=1., same_on_batch=False)
# posterize = RandomPosterize(bits=3, same_on_batch=False)
# equalize = RandomEqualize()
# shear = RandomAffine(degrees=0., shear=(10, 20))
# translate = RandomAffine(degrees=0.,translate=(0.3, 0.4), same_on_batch = False)
# brightness = ColorJitter(brightness=(0.5, 0.9))
# contrast = ColorJitter(contrast=(0.5, 0.9))
# color = ColorJitter(hue=(-0.3, 0.3)) # time consuming but feel useful
# rotation= RandomRotation(degrees=60.0)
# gray = RandomGrayscale(p=1.0)

# normal
randomerasing = RandomErasing(p=1., scale=(0.09, 0.36), ratio=(0.5, 1/0.5), same_on_batch=False)
# randomerasing = RandomErasing(p=1., scale=(.2, .4), ratio=(.1, 1/.3), same_on_batch=False)
posterize = RandomPosterize(bits=3, same_on_batch=False)
shear = RandomAffine(degrees=0., shear=(30, 30), same_on_batch=False)
translate = RandomAffine(degrees=0., translate=(0.4, 0.4), same_on_batch=False)
brightness = ColorJitter(brightness=(0.5, 0.95), same_on_batch=False)
contrast = ColorJitter(contrast=(0.5, 0.95), same_on_batch=False)
rotation= RandomRotation(degrees=30.0, same_on_batch=False)
gray = RandomGrayscale(p=1.0, same_on_batch=False)
color = ColorJitter(hue=(-0.3, 0.3)) # time consuming but feel useful
equalize = RandomEqualize(p=1.0, same_on_batch=False)
sharpness = RandomSharpness(sharpness=.5, same_on_batch=False)

# subtle change
# randomerasing = RandomErasing(p=1., scale=(0.09, 0.36), ratio=(0.5, 1/0.5), same_on_batch=False)
# # randomerasing = RandomErasing(p=1., scale=(.2, .4), ratio=(.1, 1/.3), same_on_batch=False)
# posterize = RandomPosterize(bits=6, same_on_batch=False)
# shear = RandomAffine(degrees=0., shear=(15, 15), same_on_batch=False)
# translate = RandomAffine(degrees=0., translate=(0.15, 0.15), same_on_batch=False)
# brightness = ColorJitter(brightness=(0.75, 0.99), same_on_batch=False)
# contrast = ColorJitter(contrast=(0.75, 0.99), same_on_batch=False)
# rotation= RandomRotation(degrees=15.0, same_on_batch=False)
# gray = RandomGrayscale(p=1.0, same_on_batch=False)
# color = ColorJitter(hue=(-0.10, 0.10)) # time consuming but feel useful
# equalize = RandomEqualize(p=1.0, same_on_batch=False)
# sharpness = RandomSharpness(sharpness=.5, same_on_batch=False)

def invert(imagetensor):
    imagetensor = 1-imagetensor
    return imagetensor
def dropout(imagetensor):
    mean=imagetensor.mean()
    imagetensor[:,:,:]=mean
    return imagetensor
def mixup(imagetensor):
    lower_bound = 0.5
    lam = (1-lower_bound) * random.random() + lower_bound   #! ADD by ry
    for i in range(imagetensor.size(0)-1):
        imagetensor[i]=lam*imagetensor[i]+(1-lam)*imagetensor[i+1]
    imagetensor[-1]=lam*imagetensor[-1]+(1-lam)*imagetensor[0]
    return imagetensor
def cutmix(imagetensor):
    for i in range(imagetensor.size(0) - 1):
        imagetensor[i] = imagetensor[i + 1]
    imagetensor[-1] = imagetensor[0]
    return imagetensor
def cropimage(image, i, j, k, patch_size=16):
    region = image[i,:, j * patch_size:j * patch_size + (patch_size-1), k * patch_size:k * patch_size + patch_size-1]
    return region
def cutout_tensor(image, length=16):
    mean = 0
    top = np.random.randint(0 - length // 2, 32 - length)
    left = np.random.randint(0 - length // 2, 32 - length)
    bottom = top + length
    right = left + length
    top = 0 if top < 0 else top
    left = 0 if left < 0 else left
    image[:, :, top:bottom, left:right] = mean
    return image

ops = [
        randomerasing, 
        posterize, 
        dropout, 
        cutmix, 
        shear, 
        mixup, 
        translate,
        brightness, 
        contrast, 
        rotation, 
        gray,
        invert,
        color,  
        # equalize,   
        # sharpness,   
        ]
num_ops = len(ops)

# def patch_auto_augment(image, operations, batch_size, size=1, patch_size=32):
def patch_auto_augment(image, operations, batch_size, size=2, patch_size=16, epochs=None, epoch=None, probabilities=None):
    tensorlist = [ [] for _ in range(num_ops)] 
    location = [ [] for _ in range(num_ops)]
    sum_num = []

    # if random operations
    # operations = torch.randint(num_ops, (batch_size, size, size)).cuda()

    if probabilities != None:
        '''
            only used for RLRL
        '''
        for i in range(batch_size): 
            for j in range(size):
                for k in range(size):
                    '''First,randomly select probability.
                    Then if act , RL select operation
                    '''
                    #probability = 1.0
                    probability = probabilities[i,j,k].item()/10.0
                    #print(probability)
                    operation = operations[i,j,k].item() #0-13 include 13 ,14 in total
                    if random.random()<probability:
                        tensorlist[operation].append(cropimage(image,i,j,k, patch_size=patch_size).squeeze())
                        location[operation].append((i,j,k))
    else:
        for i in range(batch_size): 
            for j in range(size):
                for k in range(size):
                    '''
                        First,randomly select probability.
                        Then if act , RL select operation
                    '''
                    # probability = 1.0
                    probability = random.random()
                    # thres_prob = 0
                    # thres_prob = 0.5
                    # thres_prob = 0.7
                    # thres_prob = 0.75
                    # thres_prob = 0.8
                    # thres_prob = 0.9
                    # thres_prob = 0.95
                    # thres_prob = 1.0
                    # thres_prob = linear_change(0.99, 0.5, epochs, epoch)
                    thres_prob = linear_change(0.5, 0.99, epochs, epoch)
                    # thres_prob = linear_change(0, 0.99, epochs, epoch)
                    # thres_prob = linear_change(0.25, 0.99, epochs, epoch)

                    # thres_prob = random.random()

                    operation = operations[i,j,k].item() #0-13 include 13 ,14 in total
                    
                    if probability > thres_prob:
                        tensorlist[operation].append(cropimage(image,i,j,k, patch_size=patch_size).squeeze())
                        location[operation].append((i,j,k))

    output = image.clone().detach()

    thres = 0.5
    # thres = linear_change(0.75, 0.25, epochs, epoch)
    # thres = linear_change(1, 0, epochs, epoch)
    # thres = linear_change(0.25, 0.75, epochs, epoch)
    # thres = linear_change(0, 1, epochs, epoch)
    if random.random() > thres:
        output = Normalize(normalize_mean, normalize_std)(output)
        return output

    for i in range(num_ops): # 13 without equalize, 12 without sharpness 
        if len(tensorlist[i])==0 :
            continue
        try:
            if ops[i] in [sharpness]:
                aftertensor = ops[i](torch.stack(tensorlist[i]).cpu()).cuda()
            else:
                aftertensor = ops[i](torch.stack(tensorlist[i]).cuda()) # time: 0.2 - 0.3
        except:
            aftertensor = torch.stack(tensorlist[i]).cuda()
            print('an error accured in ops[i]: {}'.format(ops[i]))
            # print("Unexpected error:", sys.exc_info()[0])
            pass

        for j in range(aftertensor.size(0)):
            t= location[i][j]
            output[t[0], :, t[1]* patch_size:t[1] * patch_size + (patch_size-1), 
                t[2]* patch_size:t[2]* patch_size + patch_size - 1] = aftertensor[j] # 0.01

    output = Normalize(normalize_mean, normalize_std)(output)

    return output

# choose action
def rl_choose_action(images, a2cmodel):
    result = a2cmodel(images)
    if len(result) == 2:
        dist, state_value = result
        dist = dist.permute(0,2,3,1).contiguous()# N 4 4 14
        operation_dist = Categorical(dist)
        operations = operation_dist.sample() # N 4 4
        operation_logprob = operation_dist.log_prob(operations).mean()
        operation_entropy = operation_dist.entropy().mean()
        return operations, operation_logprob, operation_entropy, state_value
    else:
        dist, dist_prob, state_value = result
        dist = dist.permute(0,2,3,1).contiguous()# N 4 4 14
        dist_prob = dist_prob.permute(0,2,3,1).contiguous()
        operation_dist = Categorical(dist)
        prob_dist = Categorical(dist_prob)
        operations = operation_dist.sample() # N 4 4
        probabilities = prob_dist.sample()
        operation_logprob = operation_dist.log_prob(operations).mean()
        operation_entropy = operation_dist.entropy().mean()
        prob_logprob = prob_dist.log_prob(probabilities).mean()
        prob_entropy = prob_dist.entropy().mean()
        return operations, probabilities, operation_logprob, operation_entropy, prob_logprob, prob_entropy, state_value

# def rl_choose_action(images, a2cmodel):
#     dist, state_value = a2cmodel(images) # dist N 14 4 4
#     dist = dist.permute(0,2,3,1).contiguous()# N 4 4 14
#     operation_dist = Categorical(dist)
#     operations = operation_dist.sample() # N 4 4
#     operation_logprob = operation_dist.log_prob(operations).mean()
#     operation_entropy = operation_dist.entropy().mean()
#     return operations, operation_logprob, operation_entropy, state_value

# loss
def a2closs(operation_logprob , operation_entropy, state_value, training_loss, prob_logprob=None, prob_entropy=None):
    if prob_logprob != None:
        advantage = (training_loss - state_value).mean()
        actor_loss = -(operation_logprob * advantage)
        prob_loss = -(prob_logprob * advantage)
        critic_loss = advantage.pow(2)
        a2closs = 0.5*actor_loss + 0.5*prob_loss + 0.5 * critic_loss - 0.0005 * operation_entropy - 0.0005 * prob_entropy
    else:
        advantage = (training_loss - state_value).mean()
        actor_loss = -(operation_logprob * advantage)
        critic_loss = advantage.pow(2)
        a2closs = actor_loss + 0.5 * critic_loss - 0.001 * operation_entropy
    return a2closs

# visualize
def saveimage(tensorlist, str):
    savepath = './visualize'
    import os
    if not os.path.exists(savepath):
        os.makedirs(savepath, exist_ok=True)
    to_pil_image = transforms.ToPILImage()
    for i in range(tensorlist.size(0)): #100 tensorlist.size(0)
        image=tensorlist[i].squeeze(0).cpu()
        imageresult=to_pil_image(image)
        imageresult.save(savepath + "/%s%d.jpg"%(str,i))

# linear function
def linear_change(start, end, total, x):
    k = (end - start) / total
    b = start
    y = k * x + b
    return y 

# time_recoder
class TimeRecoder():
    '''
    Usage:
        time_recoder = TimeRecoder()
        for ...
            time_recoder.insert()
            ...code...
            time_recoder.insert()
            ...code...
            time_recoder.insert()
            time_recoder.fresh()
        time_recoder.display()
    '''
    def __init__(self):
        self.last_time = 0
        self.time_2d_list = [[]]

    def insert(self):
        if self.last_time == 0:
            self.last_time = time.time()
        else:
            time_gap = time.time() - self.last_time
            self.last_time = time.time()
            self.time_2d_list[-1].append(time_gap)

    def fresh(self):
        self.last_time = 0
        self.time_2d_list.append([])

    def display(self, is_sum=False):
        self.time_2d_list.pop()
        self.time_2d_list_T = map(list, zip(*self.time_2d_list))
        if is_sum:
            results = [sum(each) for each in self.time_2d_list_T]
        else:
            results = [sum(each)/len(each) for each in self.time_2d_list_T]
        for i in range(len(results)):
            print('time cost in step {} to {} is {}'.format(i, i+1, results[i]))

if __name__ == '__main__':
    if args.aug == 'PAA':
    losses_a2c = AverageMeter()
    model_a2c.train()

    
    for i, (input, target) in enumerate(train_loader):
        target = target.cuda(non_blocking=True)
        input = input.cuda(non_blocking=True)

        if args.aug == 'PAA':
            # patch data augmentation
            operations ,operation_logprob , operation_entropy , state_value = \
                rl_choose_action(input, model_a2c)
            input_aug = patch_auto_augment(input, operations, args.batch_size, epochs=hyper_params['epochs'], epoch=epoch)

            output = model(input_aug.detach())
            loss = criterion(output, target)

            # PAA loss
            reward = loss.detach()
            loss_a2c = a2closs(operation_logprob, operation_entropy, state_value, reward)

            # update PAA model
            optimizer_a2c.zero_grad()
            loss_a2c.backward()
            optimizer_a2c.step()
            scheduler_a2c.step()
