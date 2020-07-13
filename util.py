import torch
from torch.autograd import Variable
from torchvision import models
import cv2
import sys
import numpy as np
import matplotlib as mpl
#mpl.use('Agg')
import matplotlib.pyplot as plt
from skimage import filters

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor



def topmax(HattMap, thre_num):
    Hatt_sort = np.argsort(HattMap, axis=None)

    Hatt_valid = np.sort(HattMap, axis=None)
    #print('Hatt_valid:', Hatt_valid)
    #print('Hatt_sort:', Hatt_sort)
    thre_index = Hatt_sort[thre_num]
    #print('thre_index:', thre_index)
    shape1 = HattMap.shape[1]
    #print('shape1:', shape1)

    thre = HattMap[int(thre_index / shape1)][int(thre_index % shape1)]
    #print('thre:', thre)
    TestHattMap = HattMap.copy()
    TestHattMap[TestHattMap < thre] = 0
    TestHattMap[TestHattMap >= thre] = 1
    #print('sum sum sum', np.sum(HattMap))
    #print('sum sum sum', np.sum(TestHattMap))

    #print(type(HattMap), HattMap.dtype)
    #HattMap = transform.resize(HattMap, oldsize, order=3, mode='edge')

    HattMap[HattMap < thre] = 0
    HattMap[HattMap >= thre] = 1
    #Hmask = 1 - HattMap

    img_ratio = np.sum(HattMap) / HattMap.size
    #print('img ratio:', img_ratio)
    return HattMap, img_ratio


def topmax_insertion(HattMap, thre_num):
    Hatt_sort = np.argsort(HattMap, axis=None)

    Hatt_valid = np.sort(HattMap, axis=None)
    # print('Hatt_valid:', Hatt_valid)
    # print('Hatt_sort:', Hatt_sort)
    thre_index = Hatt_sort[thre_num]
    # print('thre_index:', thre_index)
    shape1 = HattMap.shape[1]
    # print('shape1:', shape1)

    thre = HattMap[int(thre_index / shape1)][int(thre_index % shape1)]
    # print('thre:', thre)
    TestHattMap = HattMap.copy()
    TestHattMap[TestHattMap <= thre] = 0
    TestHattMap[TestHattMap > thre] = 1
    # print('sum sum sum', np.sum(HattMap))
    # print('sum sum sum', np.sum(TestHattMap))

    # print(type(HattMap), HattMap.dtype)
    # HattMap = transform.resize(HattMap, oldsize, order=3, mode='edge')

    HattMap[HattMap <= thre] = 0
    HattMap[HattMap > thre] = 1
    # Hmask = 1 - HattMap
    HattMap = 1 - HattMap

    img_ratio = np.sum(HattMap) / HattMap.size
    #print('img ratio:', img_ratio)
    return HattMap, img_ratio




def topmaxPixel(HattMap, thre_num):
    ii = np.unravel_index(np.argsort(HattMap.ravel())[: thre_num], HattMap.shape)
    #print(ii)
    OutHattMap = HattMap*0
    OutHattMap[ii] = 1

    img_ratio = np.sum(OutHattMap) / OutHattMap.size
    #print(OutHattMap.size)
    OutHattMap = 1 - OutHattMap


    return OutHattMap, img_ratio


def topmaxPixel_insertion(HattMap, thre_num):
    ii = np.unravel_index(np.argsort(HattMap.ravel())[: thre_num], HattMap.shape)
    # print(ii)
    OutHattMap = HattMap * 0
    OutHattMap[ii] = 1

    img_ratio = np.sum(OutHattMap) / OutHattMap.size

    return OutHattMap, img_ratio




def tv_norm(input, tv_beta):
    img = input[0, 0, :]
    row_grad = torch.mean(torch.abs((img[:-1, :] - img[1:, :])).pow(tv_beta))
    col_grad = torch.mean(torch.abs((img[:, :-1] - img[:, 1:])).pow(tv_beta))
    return row_grad + col_grad

def tv_norm_all(input, tv_beta):
    img = input[0, 0, :]
    row_grad = torch.mean(torch.abs((img[:-1, :] - img[1:, :])).pow(tv_beta))
    col_grad = torch.mean(torch.abs((img[:, :-1] - img[:, 1:])).pow(tv_beta))
    slash_grad = torch.mean(torch.abs((img[:-1, :-1] - img[1:, 1:])).pow(tv_beta))
    slash_grad2 = torch.mean(torch.abs((img[1:, :-1] - img[:-1, 1:])).pow(tv_beta))
    return row_grad + col_grad + slash_grad + slash_grad2



def tv(x, beta = 1):
    d1 = np.zeros(x.shape)
    d2 = np.zeros(x.shape)
    d1[:-1,:] = np.diff(x, axis=0)
    d2[:,:-1] = np.diff(x, axis=1)
    v = np.sqrt(d1*d1 + d2*d2)**beta
    e = v.sum()
    d1_ = (np.maximum(v, 1e-5)**(2*(beta/float(2)-1)/float(beta)))*d1
    d2_ = (np.maximum(v, 1e-5)**(2*(beta/float(2)-1)/float(beta)))*d2
    d11 = -d1_
    d22 = -d2_
    d11[1:,:] = -np.diff(d1_, axis=0)
    d22[:,1:] = -np.diff(d2_, axis=1)
    dx = beta*(d11 + d22)
    return (e,dx)


def preprocess_image(img, use_cuda=1, require_grad = False):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))

    if use_cuda:
        preprocessed_img_tensor = torch.from_numpy(preprocessed_img).cuda()
    else:
        preprocessed_img_tensor = torch.from_numpy(preprocessed_img)

    preprocessed_img_tensor.unsqueeze_(0)
    return Variable(preprocessed_img_tensor, requires_grad=require_grad)


def save(mask, img, blurred):
    mask = mask.cpu().data.numpy()[0]

    mask = np.transpose(mask, (1, 2, 0))

    print('mask.shape', type(mask), mask.shape, mask)

    mask = (mask - np.min(mask)) / np.max(mask)
    mask = 1 - mask




    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)

    print('heatmap', type(heatmap), heatmap.shape, heatmap)

    heatmap = np.float32(heatmap) / 255
    cam = 1.0 * heatmap + np.float32(img) / 255
    cam = cam / np.max(cam)

    img = np.float32(img) / 255
    perturbated = np.multiply(1 - mask, img) + np.multiply(mask, blurred)

    cv2.imwrite("Integ_perturbated.png", np.uint8(255 * perturbated))
    cv2.imwrite("Integ_heatmap.png", np.uint8(255 * heatmap))
    cv2.imwrite("Integ_mask.png", np.uint8(255 * mask))
    cv2.imwrite("Integ_cam.png", np.uint8(255 * cam))



def save_new_masked(output_path, mask, img, blurred, blur_mask=1):
    mask = mask.cpu().detach().numpy()[0]
    #print('mask.shape', type(mask), mask.shape)
    mask = np.transpose(mask, (1, 2, 0))
    #print('mask.shape', type(mask), mask.shape, mask)
    mask = (mask - np.min(mask)) / (np.max(mask)-np.min(mask))
    #mask = (mask - np.min(mask)) / np.max(mask)
    #print('mask.shape', type(mask), mask.shape, mask)
    mask = 1 - mask
    #print('mask.shape', type(mask), mask.shape, mask)
    #mask[mask < 0.2] = 0
    if blur_mask:
        mask = cv2.GaussianBlur(mask, (11, 11), 10)
        mask = np.expand_dims(mask, axis=2)
    #print('mask.shape', type(mask), mask.shape, mask)

    '''maskV = np.squeeze(mask)
    cmap = plt.get_cmap('jet')
    attMapV = cmap(maskV)
    print('attMapV:', attMapV.shape, attMapV)
    attMapV = np.delete(attMapV, 3, 2)
    print('attMapV:', attMapV.shape, attMapV)
    attMap = 1 * (1 - maskV ** 0.8).reshape(maskV.shape + (1,)) * img + (maskV ** 0.8).reshape(
            maskV.shape + (1,)) * attMapV;
    print('attMap:', attMap.shape)
    cv2.imwrite("Integ_color.png", np.uint8(attMap))'''




    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    #print('heatmap', type(heatmap), heatmap.shape, heatmap)
    heatmap = np.float32(heatmap) / 255



    #cam = 1.0 * heatmap + np.float32(img) / 255
    #cam = cam / np.max(cam)

    img = np.float32(img) / 255
    print(img.shape)
    print(mask.shape)
    print(blurred.shape)
    perturbated = np.multiply(1 - mask, img) + np.multiply(mask, blurred)

    #cam = np.multiply(1 - mask, img) + np.multiply(mask, heatmap)
    #cam = 1 * (1 - mask ** 0.8).reshape(mask.shape + (1,)) * img + (mask ** 0.8).reshape(
        #mask.shape + (1,)) * heatmap;
    cam = 1 * (1 - mask ** 0.8) * img + (mask ** 0.8)* heatmap;



    cv2.imwrite(output_path + "_perturbated.png", np.uint8(255 * perturbated))
    cv2.imwrite(output_path + "_heatmap.png", np.uint8(255 * heatmap))
    cv2.imwrite(output_path + "_mask.png", np.uint8(255 * mask))
    cv2.imwrite(output_path + "_cam.png", np.uint8(255 * cam))
    cv2.imwrite(output_path + "_blurred.png", np.uint8(255 * blurred))

    return output_path + "_cam.png"

def save_top(output_path, mask, img, blurred):
    mask = mask.cpu().data.numpy()[0]
    #print('mask.shape', type(mask), mask.shape)
    mask = np.transpose(mask, (1, 2, 0))
    #print('mask.shape', type(mask), mask.shape, mask)
    mask = (mask - np.min(mask)) / (np.max(mask)-np.min(mask))
    #mask = (mask - np.min(mask)) / np.max(mask)
    #print('mask.shape', type(mask), mask.shape, mask)
    mask = 1 - mask
    #print('mask.shape', type(mask), mask.shape, mask)

    '''maskV = np.squeeze(mask)
    cmap = plt.get_cmap('jet')
    attMapV = cmap(maskV)
    print('attMapV:', attMapV.shape, attMapV)
    attMapV = np.delete(attMapV, 3, 2)
    print('attMapV:', attMapV.shape, attMapV)
    attMap = 1 * (1 - maskV ** 0.8).reshape(maskV.shape + (1,)) * img + (maskV ** 0.8).reshape(
            maskV.shape + (1,)) * attMapV;
    print('attMap:', attMap.shape)
    cv2.imwrite("Integ_color.png", np.uint8(attMap))'''




    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    #print('heatmap', type(heatmap), heatmap.shape, heatmap)
    heatmap = np.float32(heatmap) / 255



    #cam = 1.0 * heatmap + np.float32(img) / 255
    #cam = cam / np.max(cam)

    img = np.float32(img) / 255
    perturbated = np.multiply(1 - mask, img) + np.multiply(mask, blurred)

    #cam = np.multiply(1 - mask, img) + np.multiply(mask, heatmap)
    #cam = 1 * (1 - mask ** 0.8).reshape(mask.shape + (1,)) * img + (mask ** 0.8).reshape(
        #mask.shape + (1,)) * heatmap;
    cam = 1 * (1 - mask ** 0.8) * img + (mask ** 0.8)* heatmap;



    cv2.imwrite(output_path + "perturbated.png", np.uint8(255 * perturbated))
    cv2.imwrite(output_path + "heatmap.png", np.uint8(255 * heatmap))
    cv2.imwrite(output_path + "mask.png", np.uint8(255 * mask))
    cv2.imwrite(output_path + "cam.png", np.uint8(255 * cam))
    cv2.imwrite(output_path + "blurred.png", np.uint8(255 * blurred))




def numpy_to_torch(img, use_cuda=1, requires_grad=False):
    if len(img.shape) < 3:
        output = np.float32([img])
    else:
        output = np.transpose(img, (2, 0, 1))

    output = torch.from_numpy(output)
    if use_cuda:
        output = output.cuda()

    output.unsqueeze_(0)
    v = Variable(output, requires_grad=requires_grad)
    return v


def load_model(use_cuda = 1):
    model = models.vgg19(pretrained=True)
    #model = models.resnet101(pretrained=True)
    model.eval()
    if use_cuda:
        model.cuda()

    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = False


    return model

def load_model_new(use_cuda = 1, model_name = 'resnet50'):

    if model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
    elif model_name == 'vgg19':
        model = models.vgg19(pretrained=True)

    #print(model)
    model.eval()
    if use_cuda:
        model.cuda()

    for p in model.parameters():
        p.requires_grad = False

    return model

def save_curve(curve1, curve2,curvetop, output_file):
    plt.figure()
#    plt.hold(True)
    plt.plot(curve1, 'r*-')
    plt.xlabel('iterations')
    plt.ylabel('objective function value')
    # hold(True)
    plt.plot(curve2, 'bo-')
    plt.plot(curvetop, 'g+-')
    plt.legend(['loss1', 'loss2', 'losstop'])
    plt.savefig(output_file + "loss.png")


