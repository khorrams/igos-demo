from util import *
import os
import time
import scipy.io as scio
import datetime
import re
import matplotlib.pyplot as plt
import numpy as np
import pylab
import os
import csv
from skimage import transform, filters
from textwrap import wrap
import cv2
import shutil
import dummy




def Get_blurred_img(input_img, img_label, model, resize_shape=(224, 224), Gaussian_param = [51, 50], Median_param = 11, blur_type= 'Gaussian', use_cuda=use_cuda):

    original_img = cv2.imread(input_img, 1)
    #print('original_img:', original_img)
    original_img = cv2.resize(original_img, resize_shape)
    img = np.float32(original_img) / 255

    if blur_type =='Gaussian':
        Kernelsize = Gaussian_param[0]
        SigmaX = Gaussian_param[1]
        blurred_img = cv2.GaussianBlur(img, (Kernelsize, Kernelsize), SigmaX)
        # blurred_img *= 0.0
    elif blur_type == 'Median':
        Kernelsize_M = Median_param
        blurred_img = np.float32(cv2.medianBlur(original_img, Kernelsize_M)) / 255

    elif blur_type == 'Mixed':
        Kernelsize = Gaussian_param[0]
        SigmaX = Gaussian_param[1]
        blurred_img1 = cv2.GaussianBlur(img, (Kernelsize, Kernelsize), SigmaX)

        Kernelsize_M = Median_param
        blurred_img2 = np.float32(cv2.medianBlur(original_img, Kernelsize_M)) / 255

        blurred_img = (blurred_img1 + blurred_img2) / 2

    img_torch = preprocess_image(img, use_cuda, require_grad = False)
    blurred_img_torch = preprocess_image(blurred_img, use_cuda, require_grad = False)

    ori_output = model(img_torch)
    blurred_output = model(blurred_img_torch)

    if use_cuda:
        logitori = ori_output.data.cpu().numpy().copy().squeeze()
        logitblur = blurred_output.data.cpu().numpy().copy().squeeze()
    else:
        logitori = ori_output.data.numpy().copy().squeeze()
        logitblur = blurred_output.data.cpu().numpy().copy().squeeze()


    top_5_idx = np.argsort(logitori)[-5:]
    top_5_values = [logitori[i] for i in top_5_idx]
    print('top_5_idx:', top_5_idx, top_5_values)

    rew = np.where(logitori == np.max(logitori))
    print(rew)
    output_label = rew[0][0]

    if img_label == -1:
        img_label = output_label

    rew_blur = np.where(logitblur == np.max(logitblur))
    output_label_blur = rew_blur[0][0]


    #print('ori_output:', ori_output)
    print('ori_output:', ori_output[0, img_label], output_label)
    print('blurred_output:', blurred_output[0, img_label], output_label_blur)
    blur_ratio = blurred_output[0, img_label] / ori_output[0, img_label]


    #return img, blurred_img, blur_ratio, output_label, output_label_blur, ori_output, blurred_output
    return img, blurred_img, logitori


def Integrated_Mask(img, blurred_img, model, category, max_iterations = 15, integ_iter = 20,
                    tv_beta=2, l1_coeff = 0.01*300, tv_coeff = 0.2*300, size_init = 112, use_cuda =1):


    img = preprocess_image(img, use_cuda, require_grad=False)
    blurred_img = preprocess_image(blurred_img, use_cuda, require_grad=False)

    resize_size = img.data.shape
    resize_wh = (img.data.shape[2], img.data.shape[3])
    #print('resize_size:', resize_size)
    #print('resize_wh:', resize_wh)
    if use_cuda:
        zero_img = Variable(torch.zeros(resize_size).cuda(), requires_grad=False)
    else:
        zero_img = Variable(torch.zeros(resize_size), requires_grad=False)
    #print('zero_img:', type(zero_img))
    mask_init = np.ones((size_init, size_init), dtype=np.float32)
    mask = numpy_to_torch(mask_init, use_cuda, requires_grad=True)

    #mask_base = np.zeros((size_init, size_init), dtype=np.float32)
    #mask_base = numpy_to_torch(mask_base, use_cuda, requires_grad=False)


    if use_cuda:
        upsample = torch.nn.UpsamplingBilinear2d(size=resize_wh).cuda()
    else:
        upsample = torch.nn.UpsamplingBilinear2d(size=resize_wh)

    # You can choose any optimizer
    # The optimizer doesn't matter, because we don't need optimizer.step(), we just to compute the gradient
    optimizer = torch.optim.Adam([mask], lr=0.1)

    target = torch.nn.Softmax()(model(img))
    if use_cuda:
        category_out = np.argmax(target.cpu().data.numpy())
    else:
        category_out = np.argmax(target.data.numpy())

    if category ==-1:
        category = category_out

    print("Category with highest probability", category_out)
    print("Category want to generate mask", category)
    print("Optimizing.. ")




    curve1 = np.array([])
    curve2 = np.array([])
    curvetop = np.array([])
    #curve_total = np.array([])

    #alpha = 0.25
    alpha = 0.0001
    #alpha = 0.00005
    #beta = 0.9
    beta = 0.2

    for i in range(max_iterations):
        upsampled_mask = upsample(mask)
        # The single channel mask is used with an RGB image,
        # so the mask is duplicated to have 3 channel,
        upsampled_mask = \
            upsampled_mask.expand(1, 3, upsampled_mask.size(2), \
                                  upsampled_mask.size(3))


        loss1 = l1_coeff * torch.mean(torch.abs(1 - mask)) + \
                tv_coeff * tv_norm(mask, tv_beta)
        loss_all = loss1.clone()

        perturbated_input_base = img.mul(upsampled_mask) + \
                                 blurred_img.mul(1 - upsampled_mask)


        for inte_i in range(integ_iter):


            ############################### Use the mask to perturbated the input image.
            integ_mask = 0.0 + ((inte_i + 1.0) / integ_iter) * upsampled_mask


            perturbated_input_integ = img.mul(integ_mask) + \
                                     blurred_img.mul(1 - integ_mask)

            noise = np.zeros((resize_wh[0], resize_wh[1], 3), dtype=np.float32)
            noise = noise + cv2.randn(noise, 0, 0.2)
            #noise = noise + cv2.randn(noise, (0, 0, 0), (0.1, 0.1, 0.1))
            noise = numpy_to_torch(noise, use_cuda, requires_grad=False)

            perturbated_input = perturbated_input_integ + noise
            ####################################################################################

            #new_image = 0.5* blurred_img + (inte_i / 20.0) * perturbated_input
            new_image = perturbated_input
            outputs = torch.nn.Softmax()(model(new_image))
            loss2 = outputs[0, category]

            loss_all = loss_all + loss2/20.0



        optimizer.zero_grad()
        loss_all.backward()
        whole_grad = mask.grad.data.clone()

        loss2_ori = torch.nn.Softmax()(model(perturbated_input_base))[0, category]



        loss_ori = loss1 + loss2_ori
        if i==0:
            if use_cuda:
                curve1 = np.append(curve1, loss1.data.cpu().numpy())
                curve2 = np.append(curve2, loss2_ori.data.cpu().numpy())
                curvetop = np.append(curvetop, loss2_ori.data.cpu().numpy())
                #curve_total = np.append(curve_total, loss_ori.data.cpu().numpy())
            else:
                curve1 = np.append(curve1, loss1.data.numpy())
                curve2 = np.append(curve2, loss2_ori.data.numpy())
                curvetop = np.append(curvetop, loss2_ori.data.numpy())
                #curve_total = np.append(curve_total, loss_ori.data.numpy())


        if use_cuda:
            loss_oridata = loss_ori.data.cpu().numpy()
        else:
            loss_oridata = loss_ori.data.numpy()

        ######################################################LINE SEARCH
        step = 200.0
        #directionLS = torch.div(whole_grad, np.abs(whole_grad.sum()))
        MaskClone = mask.data.clone()
        MaskClone -= step * whole_grad
        #MaskClone -= step * directionLS
        MaskClone = Variable(MaskClone, requires_grad=False)
        MaskClone.data.clamp_(0, 1)


        mask_LS = upsample(MaskClone)   # Here the direction is the whole_grad
        Img_LS = img.mul(mask_LS) + \
                 blurred_img.mul(1 - mask_LS)
        outputsLS = torch.nn.Softmax()(model(Img_LS))
        loss_LS = l1_coeff * torch.mean(torch.abs(1 - MaskClone)) + \
                  tv_coeff * tv_norm(MaskClone, tv_beta) + outputsLS[0, category]

        if use_cuda:
            loss_LSdata = loss_LS.data.cpu().numpy()
        else:
            loss_LSdata = loss_LS.data.numpy()


        new_condition = whole_grad ** 2  # Here the direction is the whole_grad
        #new_condition = torch.mul(whole_grad, directionLS)
        new_condition = new_condition.sum()
        new_condition = alpha * step * new_condition




        #while loss_LS.data.cpu().numpy() > loss_ori.data.cpu().numpy() - new_condition:
        while loss_LSdata > loss_oridata - new_condition:
            step *= beta

            #directionLS = torch.div(whole_grad, np.abs(whole_grad.sum()))
            MaskClone = mask.data.clone()
            MaskClone -= step * whole_grad
            #MaskClone -= step * directionLS
            MaskClone = Variable(MaskClone, requires_grad=False)
            MaskClone.data.clamp_(0, 1)
            mask_LS = upsample(MaskClone)
            Img_LS = img.mul(mask_LS) + \
                     blurred_img.mul(1 - mask_LS)
            outputsLS = torch.nn.Softmax()(model(Img_LS))
            loss_LS = l1_coeff * torch.mean(torch.abs(1 - MaskClone)) + \
                      tv_coeff * tv_norm(MaskClone, tv_beta) + outputsLS[0, category]

            if use_cuda:
                loss_LSdata = loss_LS.data.cpu().numpy()
            else:
                loss_LSdata = loss_LS.data.numpy()


            new_condition = whole_grad ** 2  # Here the direction is the whole_grad
            #new_condition = torch.mul(whole_grad, directionLS)
            new_condition = new_condition.sum()
            new_condition = alpha * step * new_condition

            if step<0.00001:
                break

        #print('loss_LSdata:', loss_LSdata)
        print('step:', step)
        mask.data -= step * whole_grad

        #######################################################################################################



        #mask.data -= learning_rate * whole_grad
        #mask.data -= 50*whole_grad


        if use_cuda:
            curve1 = np.append(curve1, loss1.data.cpu().numpy())
            curve2 = np.append(curve2, loss2_ori.data.cpu().numpy())
        else:
            curve1 = np.append(curve1, loss1.data.numpy())
            curve2 = np.append(curve2, loss2_ori.data.numpy())

        # Optional: clamping seems to give better results
        mask.data.clamp_(0, 1)
        if use_cuda:
            maskdata = mask.data.cpu().numpy()
        else:
            maskdata = mask.data.numpy()
        #print('mask:', mask)
        #print('mask_min:', np.min(maskdata))
        #print('mask_min_index:', np.unravel_index(maskdata.argmin(), maskdata.shape))
        #print(maskdata.shape)
        maskdata = np.squeeze(maskdata)
        maskdata, imgratio = topmaxPixel(maskdata, 40)
        maskdata = np.expand_dims(maskdata, axis=0)
        maskdata = np.expand_dims(maskdata, axis=0)

        ###############################################
        # Masktop = maskdata
        if use_cuda:
            Masktop = torch.from_numpy(maskdata).cuda()
        else:
            Masktop = torch.from_numpy(maskdata)
        # Use the mask to perturbated the input image.
        Masktop = Variable(Masktop, requires_grad=False)
        MasktopLS = upsample(Masktop)
        # MasktopLS = \
        #    MasktopLS.expand(1, 3, MasktopLS.size(2), \
        #                     MasktopLS.size(3))
        Img_topLS = img.mul(MasktopLS) + \
                    blurred_img.mul(1 - MasktopLS)
        outputstopLS = torch.nn.Softmax()(model(Img_topLS))
        loss_top1 = l1_coeff * torch.mean(torch.abs(1 - Masktop)) + \
                    tv_coeff * tv_norm(Masktop, tv_beta)
        loss_top2 = outputstopLS[0, category]

        if use_cuda:
            curvetop = np.append(curvetop, loss_top2.data.cpu().numpy())
        else:
            curvetop = np.append(curvetop, loss_top2.data.numpy())


        if max_iterations >3:

            if i == int(max_iterations / 2):
                if np.abs(curve2[0] - curve2[i]) <= 0.001:
                    print('Adjust Parameter l1_coeff at iteration:', int(max_iterations / 2))
                    l1_coeff = l1_coeff / 10

            elif i == int(max_iterations / 1.25):
                if np.abs(curve2[0] - curve2[i]) <= 0.01:
                    print('Adjust Parameters l1_coeff again at iteration:', int(max_iterations / 1.25))
                    l1_coeff = l1_coeff / 5


            #######################################################################################

    upsampled_mask = upsample(mask)

    if use_cuda:
        mask = mask.data.cpu().numpy().copy()
    else:
        mask = mask.data.numpy().copy()

    return mask, upsampled_mask, imgratio, curvetop, curve1, curve2, category



def Integrated_Mask_ori(img, blurred_img, model, category, max_iterations = 15, integ_iter = 20,
                    tv_beta=2, l1_coeff = 0.01*300, tv_coeff = 0.2*300, size_init = 112, use_cuda =1):


    img = preprocess_image(img, use_cuda, require_grad=False)
    blurred_img = preprocess_image(blurred_img, use_cuda, require_grad=False)

    resize_size = img.data.shape
    resize_wh = (img.data.shape[2], img.data.shape[3])
    #print('resize_size:', resize_size)
    #print('resize_wh:', resize_wh)
    if use_cuda:
        zero_img = Variable(torch.zeros(resize_size).cuda(), requires_grad=False)
    else:
        zero_img = Variable(torch.zeros(resize_size), requires_grad=False)
    #print('zero_img:', type(zero_img))
    mask_init = np.ones((size_init, size_init), dtype=np.float32)
    mask = numpy_to_torch(mask_init, use_cuda, requires_grad=True)

    #mask_base = np.zeros((size_init, size_init), dtype=np.float32)
    #mask_base = numpy_to_torch(mask_base, use_cuda, requires_grad=False)


    if use_cuda:
        upsample = torch.nn.UpsamplingBilinear2d(size=resize_wh).cuda()
    else:
        upsample = torch.nn.UpsamplingBilinear2d(size=resize_wh)

    optimizer = torch.optim.Adam([mask], lr=0.1)

    target = torch.nn.Softmax()(model(img))
    if use_cuda:
        category_out = np.argmax(target.cpu().data.numpy())
    else:
        category_out = np.argmax(target.data.numpy())

    if category ==-1:
        category = category_out

    print("Category with highest probability", category_out)
    print("Category want to generate mask", category)
    print("Optimizing.. ")




    curve1 = np.array([])
    curve2 = np.array([])
    curvetop = np.array([])

    #alpha = 0.25
    alpha = 0.0001
    #alpha = 0.00005
    #beta = 0.9
    beta = 0.1

    for i in range(max_iterations):
        upsampled_mask = upsample(mask)
        # The single channel mask is used with an RGB image,
        # so the mask is duplicated to have 3 channel,
        upsampled_mask = \
            upsampled_mask.expand(1, 3, upsampled_mask.size(2), \
                                  upsampled_mask.size(3))


        loss1 = l1_coeff * torch.mean(torch.abs(1 - mask)) + \
                tv_coeff * tv_norm(mask, tv_beta)
        loss_all = loss1.clone()

        perturbated_input_base = img.mul(upsampled_mask) + \
                                 blurred_img.mul(1 - upsampled_mask)


        for inte_i in range(integ_iter):


            ############################### Use the mask to perturbated the input image.
            integ_mask = 0 + ((inte_i + 1.0) / integ_iter) * upsampled_mask


            perturbated_input_integ = img.mul(integ_mask) + \
                                     blurred_img.mul(1 - integ_mask)

            noise = np.zeros((resize_wh[0], resize_wh[1], 3), dtype=np.float32)
            noise = noise + cv2.randn(noise, 0, 0.2)
            #noise = noise + cv2.randn(noise, (0, 0, 0), (0.1, 0.1, 0.1))
            noise = numpy_to_torch(noise, use_cuda, requires_grad=False)

            perturbated_input = perturbated_input_integ + noise
            ####################################################################################

            #new_image = 0.5* blurred_img + (inte_i / 20.0) * perturbated_input
            new_image = perturbated_input
            outputs = torch.nn.Softmax()(model(new_image))
            loss2 = outputs[0, category]

            loss_all = loss_all + loss2/20.0



        optimizer.zero_grad()
        loss_all.backward()
        whole_grad = mask.grad.data.clone()

        loss2_ori = torch.nn.Softmax()(model(perturbated_input_base))[0, category]


        ######################################################LINE SEARCH
        loss_ori = loss1 + loss2_ori
        if i==0:
            if use_cuda:
                curve1 = np.append(curve1, loss1.data.cpu().numpy())
                curve2 = np.append(curve2, loss2_ori.data.cpu().numpy())
                curvetop = np.append(curvetop, loss2_ori.data.cpu().numpy())
            else:
                curve1 = np.append(curve1, loss1.data.numpy())
                curve2 = np.append(curve2, loss2_ori.data.numpy())
                curvetop = np.append(curvetop, loss2_ori.data.numpy())


        if use_cuda:
            loss_oridata = loss_ori.data.cpu().numpy()
        else:
            loss_oridata = loss_ori.data.numpy()


        step = 100.0
        directionLS = torch.div(whole_grad, np.abs(whole_grad.sum()))
        MaskClone = mask.data.clone()
        #MaskClone -= step * whole_grad
        MaskClone -= step * directionLS
        MaskClone = Variable(MaskClone, requires_grad=False)


        mask_LS = upsample(MaskClone)   # Here the direction is the whole_grad
        Img_LS = img.mul(mask_LS) + \
                 blurred_img.mul(1 - mask_LS)
        outputsLS = torch.nn.Softmax()(model(Img_LS))
        loss_LS = l1_coeff * torch.mean(torch.abs(1 - MaskClone)) + \
                  tv_coeff * tv_norm(MaskClone, tv_beta) + outputsLS[0, category]

        if use_cuda:
            loss_LSdata = loss_LS.data.cpu().numpy()
        else:
            loss_LSdata = loss_LS.data.numpy()


        #new_condition = whole_grad ** 2  # Here the direction is the whole_grad
        new_condition = torch.mul(whole_grad, directionLS)
        new_condition = new_condition.sum()
        new_condition = alpha * step * new_condition




        #while loss_LS.data.cpu().numpy() > loss_ori.data.cpu().numpy() - new_condition:
        while loss_LSdata > loss_oridata - new_condition:
            step *= beta

            directionLS = torch.div(whole_grad, np.abs(whole_grad.sum()))
            MaskClone = mask.data.clone()
            #MaskClone -= step * whole_grad
            MaskClone -= step * directionLS
            MaskClone = Variable(MaskClone, requires_grad=False)

            mask_LS = upsample(MaskClone)
            Img_LS = img.mul(mask_LS) + \
                     blurred_img.mul(1 - mask_LS)
            outputsLS = torch.nn.Softmax()(model(Img_LS))
            loss_LS = l1_coeff * torch.mean(torch.abs(1 - MaskClone)) + \
                      tv_coeff * tv_norm(MaskClone, tv_beta) + outputsLS[0, category]

            if use_cuda:
                loss_LSdata = loss_LS.data.cpu().numpy()
            else:
                loss_LSdata = loss_LS.data.numpy()


            #new_condition = whole_grad ** 2  # Here the direction is the whole_grad
            new_condition = torch.mul(whole_grad, directionLS)
            new_condition = new_condition.sum()
            new_condition = alpha * step * new_condition


        mask.data -= step * whole_grad

        #######################################################################################################


        # mask.data -= learning_rate * whole_grad


        if use_cuda:
            curve1 = np.append(curve1, loss1.data.cpu().numpy())
            curve2 = np.append(curve2, loss2_ori.data.cpu().numpy())
        else:
            curve1 = np.append(curve1, loss1.data.numpy())
            curve2 = np.append(curve2, loss2_ori.data.numpy())

        # Optional: clamping seems to give better results
        mask.data.clamp_(0, 1)
        if use_cuda:
            maskdata = mask.data.cpu().numpy()
        else:
            maskdata = mask.data.numpy()
        #print('mask:', mask)
        #print('mask_min:', np.min(maskdata))
        #print('mask_min_index:', np.unravel_index(maskdata.argmin(), maskdata.shape))
        #print(maskdata.shape)
        maskdata = np.squeeze(maskdata)
        maskdata, imgratio = topmaxPixel(maskdata, 40)
        maskdata = np.expand_dims(maskdata, axis=0)
        maskdata = np.expand_dims(maskdata, axis=0)

        ###############################################
        # Masktop = maskdata
        if use_cuda:
            Masktop = torch.from_numpy(maskdata).cuda()
        else:
            Masktop = torch.from_numpy(maskdata)
        # Use the mask to perturbated the input image.
        Masktop = Variable(Masktop, requires_grad=False)
        MasktopLS = upsample(Masktop)
        # MasktopLS = \
        #    MasktopLS.expand(1, 3, MasktopLS.size(2), \
        #                     MasktopLS.size(3))
        Img_topLS = img.mul(MasktopLS) + \
                    blurred_img.mul(1 - MasktopLS)
        outputstopLS = torch.nn.Softmax()(model(Img_topLS))
        loss_top1 = l1_coeff * torch.mean(torch.abs(1 - Masktop)) + \
                    tv_coeff * tv_norm(Masktop, tv_beta)
        loss_top2 = outputstopLS[0, category]

        if use_cuda:
            curvetop = np.append(curvetop, loss_top2.data.cpu().numpy())
        else:
            curvetop = np.append(curvetop, loss_top2.data.numpy())


        if max_iterations >3:

            if i == int(max_iterations / 2):
                if np.abs(curve2[0] - curve2[i]) <= 0.001:
                    print('Adjust Parameter l1_coeff at iteration:', int(max_iterations / 2))
                    l1_coeff = l1_coeff / 10

            elif i == int(max_iterations / 1.25):
                if np.abs(curve2[0] - curve2[i]) <= 0.01:
                    print('Adjust Parameters l1_coeff again at iteration:', int(max_iterations / 1.25))
                    l1_coeff = l1_coeff / 5


            #######################################################################################

    upsampled_mask = upsample(mask)

    if use_cuda:
        mask = mask.data.cpu().numpy().copy()
    else:
        mask = mask.data.numpy().copy()

    return mask, upsampled_mask, imgratio, curvetop, curve1, curve2, category







def save_new(mask, img, blurred, type, i, outpath):
    mask = mask.cpu().data.numpy()[0]
    #print('mask:', mask.shape)
    mask = np.transpose(mask, (1, 2, 0))
    #print('mask:', mask.shape)
    img = np.float32(img) / 255
    perturbated = np.multiply(mask, img) + np.multiply(1-mask, blurred)
    # perturbated = cv2.cvtColor(perturbated, cv2.COLOR_BGR2RGB)
    if not os.path.isdir(outpath):
        os.makedirs(outpath)
    file_path = '{}{}_{}.jpg'.format(outpath, type, i)
    print(file_path)
    # perturbated = np.transpose(perturbated, (2, 0, 1))
    cv2.imwrite(file_path, np.int32(perturbated * 255))
    return perturbated



def showimage(del_img, insert_img, del_curve, insert_curve, target_path, sizeM, xtick, title):
    ########################

    ####################################################
    pylab.rcParams['figure.figsize'] = (10, 10)
    f, ax = plt.subplots(2,2)
    ax[0,0].set_title('Category: ' + title, fontsize=13)
    f.tight_layout()  # 调整整体空白
    plt.subplots_adjust(left=0.005, bottom=0.1, right=0.98, top=0.93,
                        wspace=0.05, hspace=0.25)  # 调整子图间距



    ax[0,0].imshow(del_img)
    ax[0,0].set_xticks([])
    ax[0,0].set_yticks([])
    ax[0,0].set_title("Deletion", fontsize=14)

    ax[1,0].imshow(insert_img)
    ax[1,0].set_xticks([])
    ax[1,0].set_yticks([])
    ax[1,0].set_title("Insertion", fontsize=14)




    ax[0,1].plot(del_curve,'r*-')
    #ax[0,1].set_xlabel('number of pixels (*' + str(int(sizeM / 50)) + ')')
    ax[0,1].set_xlabel('number of blocks', fontsize=12)
    ax[0,1].set_ylabel('classification confidence', fontsize=12)
    ax[0,1].legend(['Deletion'])
    ax[0,1].set_xticks(range(0, xtick, 10))
    ax[0,1].set_yticks(np.arange(0, 1.1, 0.1))


    ax[1,1].plot(insert_curve, 'b*-')
    #ax[1,1].set_xlabel('number of pixels (*' + str(int(sizeM / 50)) + ')')
    ax[1,1].set_xlabel('number of blocks', fontsize=12)
    ax[1,1].set_ylabel('classification confidence', fontsize=12)
    ax[1,1].legend(['Insertion'])
    ax[1,1].set_xticks(range(0, xtick, 10))
    ax[1,1].set_yticks(np.arange(0, 1.1, 0.1))
    print(insert_curve.shape[0])

    plt.savefig(target_path + 'video'+ str(insert_curve.shape[0])+ '.jpg')
    #plt.show()



def predict(model, img_ori, use_cuda):

    img = preprocess_image(img_ori, use_cuda, require_grad=False)

    target = torch.nn.Softmax()(model(img))

    if use_cuda:
        category_out = np.argmax(target.cpu().data.numpy())
    else:
        category_out = np.argmax(target.data.numpy())

    f_groundtruth = open('./GroundTruth1000.txt')
    line_i = f_groundtruth.readlines()[category_out]
    f_groundtruth.close()

    category_name = line_i.split(': ')[1]

    return category_name

def Deletion_Insertion(mask, model, output_path, img_ori, blurred_img_ori, logitori, category, pixelnum = 200, use_cuda =1, blur_mask=0, outputfig = 0):

    print(output_path)
    #print('mask.shape', type(mask), mask.shape, mask)
    #print(np.min(mask), np.max(mask))

    sizeM = mask.shape[2] * mask.shape[3]



    if blur_mask:
        mask = (mask - np.min(mask)) / np.max(mask)
        mask = 1 - mask
        mask = cv2.GaussianBlur(mask, (51, 51), 50)
        mask = 1-mask


    #blurred_insert = cv2.GaussianBlur(img, (11, 11), 10)
    blurred_insert = blurred_img_ori.copy()                             #todo
    blurred_insert = preprocess_image(blurred_insert, use_cuda, require_grad=False)

    img = preprocess_image(img_ori, use_cuda, require_grad=False)
    blurred_img = preprocess_image(blurred_img_ori, use_cuda, require_grad=False)
    resize_wh = (img.data.shape[2], img.data.shape[3])
    if use_cuda:
        upsample = torch.nn.UpsamplingBilinear2d(size=resize_wh).cuda()
    else:
        upsample = torch.nn.UpsamplingBilinear2d(size=resize_wh)

    target = torch.nn.Softmax()(model(img))
    if use_cuda:
        category_out = np.argmax(target.cpu().data.numpy())
    else:
        category_out = np.argmax(target.data.numpy())

    if category == -1:
        category = category_out

    print("Category with highest probability", category_out)
    print("Category want to generate mask", category)

    outmax = target[0, category].cpu().data.numpy()
    #print('logitori:', logitori)
    logitori = logitori[category]

    #print('sizeM:', sizeM, sizeM/50)
    del_curve = np.array([])
    insert_curve = np.array([])




    if sizeM<pixelnum:
        intM = 4
    else:
        intM = int(sizeM/pixelnum)

    print(sizeM, intM)

    xtick = np.arange(0, int(sizeM+1), intM)
    xnum = xtick.shape[0]
    print('xnum:', xnum)

    xtick = xtick.shape[0] + 10

    f_groundtruth = open('./GroundTruth1000.txt')
    line_i = f_groundtruth.readlines()[category]
    f_groundtruth.close()

    category_name = line_i.split(': ')[1]
    print('line_i:', line_i)
    #title = re.findall('\'([^"]*)\'', line_i)
    #title = title[0]
    #print('title:', title)


    for pix_num in range(0, int(sizeM+1), intM):
        maskdata = mask.copy()

        maskdata = np.squeeze(maskdata)
        #print('pix_num:', pix_num)
        #print('maskdata.shape', maskdata.shape)
        maskdata, imgratio = topmaxPixel(maskdata, pix_num)
        #maskdata, imgratio = topmax(maskdata, pix_num)
        maskdata = np.expand_dims(maskdata, axis=0)
        maskdata = np.expand_dims(maskdata, axis=0)

        ###############################################
        if use_cuda:
            Masktop = torch.from_numpy(maskdata).cuda()
        else:
            Masktop = torch.from_numpy(maskdata)
        # Use the mask to perturbated the input image.
        Masktop = Variable(Masktop, requires_grad=False)
        #print('Masktop Del:', Masktop)
        MasktopLS = upsample(Masktop)
        # MasktopLS = \
        #    MasktopLS.expand(1, 3, MasktopLS.size(2), \
        #                     MasktopLS.size(3))
        Img_topLS = img.mul(MasktopLS) + \
                    blurred_img.mul(1 - MasktopLS)

        #print('MasktopLS Del:', MasktopLS)

        outputstopLS_ori = model(Img_topLS)[0, category].data.cpu().numpy().copy()
        outputstopLS = torch.nn.Softmax()(model(Img_topLS))
        delloss_top2 = outputstopLS[0, category].data.cpu().numpy().copy()
        del_mask = MasktopLS.clone()

        #del_ratio = outputstopLS_ori / logitori
        delimg_ratio = imgratio.copy()
        del_ratio = delloss_top2 / outmax
        del_curve = np.append(del_curve, delloss_top2)
        #print('del_ratio:', del_ratio)



        maskdata = mask.copy()

        maskdata = np.squeeze(maskdata)
        # print('pix_num:', pix_num)
        # print('maskdata.shape', maskdata.shape)
        maskdata, imgratio = topmaxPixel_insertion(maskdata, pix_num)
        # maskdata, imgratio = topmax_insertion(maskdata, pix_num)

        # print('maskdata:', maskdata)
        maskdata = np.expand_dims(maskdata, axis=0)
        maskdata = np.expand_dims(maskdata, axis=0)

        ###############################################
        if use_cuda:
            Masktop = torch.from_numpy(maskdata).cuda()
        else:
            Masktop = torch.from_numpy(maskdata)
        # Use the mask to perturbated the input image.
        Masktop = Variable(Masktop, requires_grad=False)
        # print('Masktop Ins:', Masktop)
        MasktopLS = upsample(Masktop)
        # MasktopLS = \
        #    MasktopLS.expand(1, 3, MasktopLS.size(2), \
        #                     MasktopLS.size(3))
        Img_topLS = img.mul(MasktopLS) + \
                    blurred_insert.mul(1 - MasktopLS)
        # print('MasktopLS Ins:', MasktopLS)

        outputstopLS_ori = model(Img_topLS)[0, category].data.cpu().numpy().copy()
        outputstopLS = torch.nn.Softmax()(model(Img_topLS))
        insloss_top2 = outputstopLS[0, category].data.cpu().numpy().copy()
        ins_mask = MasktopLS.clone()

        # ins_ratio = outputstopLS_ori / logitori
        insimg_ratio = imgratio.copy()
        ins_ratio = insloss_top2 / outmax
        insert_curve = np.append(insert_curve, insloss_top2)

        if outputfig == 1:
            deletion_img = save_new(del_mask, img_ori * 255, blurred_img_ori, 'del', pix_num, output_path)

            '''cv2.imwrite(output_path + str(outmax) + '_' + str(delloss_top2) + '_' + str(delimg_ratio) + '_deletion__'+str(category)+'.png',
                        np.uint8(255 * perturbated))
                        '''

            insertion_img = save_new(ins_mask, img_ori * 255, blurred_img_ori, 'ins', pix_num, output_path)

            '''cv2.imwrite(output_path + str(outmax) + '_' + str(insloss_top2) + '_' + str(insimg_ratio) + '_insertion__'+str(category)+'.png',
                        np.uint8(255 * perturbated))'''
            # showimage(deletion_img, insertion_img, del_curve, insert_curve, output_path, sizeM, xtick, category_name)




    '''
    for pix_num in range(0, sizeM, intM):
        maskdata = mask.copy()

        maskdata = np.squeeze(maskdata)
        #print('pix_num:', pix_num)
        #print('maskdata.shape', maskdata.shape)
        maskdata, imgratio = topmaxPixel(maskdata, pix_num)
        #maskdata, imgratio = topmax(maskdata, pix_num)
        maskdata = np.expand_dims(maskdata, axis=0)
        maskdata = np.expand_dims(maskdata, axis=0)

        ###############################################
        if use_cuda:
            Masktop = torch.from_numpy(maskdata).cuda()
        else:
            Masktop = torch.from_numpy(maskdata)
        # Use the mask to perturbated the input image.
        Masktop = Variable(Masktop, requires_grad=False)
        #print('Masktop Del:', Masktop)
        MasktopLS = upsample(Masktop)
        # MasktopLS = \
        #    MasktopLS.expand(1, 3, MasktopLS.size(2), \
        #                     MasktopLS.size(3))
        Img_topLS = img.mul(MasktopLS) + \
                    blurred_img.mul(1 - MasktopLS)

        #print('MasktopLS Del:', MasktopLS)

        outputstopLS_ori = model(Img_topLS)[0, category].data.cpu().numpy().copy()
        outputstopLS = torch.nn.Softmax()(model(Img_topLS))
        delloss_top2 = outputstopLS[0, category].data.cpu().numpy().copy()
        del_mask = MasktopLS.clone()

        #del_ratio = outputstopLS_ori / logitori
        delimg_ratio = imgratio.copy()
        del_ratio = delloss_top2 / outmax
        del_curve = np.append(del_curve, delloss_top2)
        #print('del_ratio:', del_ratio)
        if del_ratio<=0.2:
            print('del_ratio:', del_ratio)
            break





    for pix_num in range(0, sizeM, intM):
        maskdata = mask.copy()

        maskdata = np.squeeze(maskdata)
        #print('pix_num:', pix_num)
        #print('maskdata.shape', maskdata.shape)
        maskdata, imgratio = topmaxPixel_insertion(maskdata, pix_num)
        #maskdata, imgratio = topmax_insertion(maskdata, pix_num)

        #print('maskdata:', maskdata)
        maskdata = np.expand_dims(maskdata, axis=0)
        maskdata = np.expand_dims(maskdata, axis=0)

        ###############################################
        if use_cuda:
            Masktop = torch.from_numpy(maskdata).cuda()
        else:
            Masktop = torch.from_numpy(maskdata)
        # Use the mask to perturbated the input image.
        Masktop = Variable(Masktop, requires_grad=False)
        #print('Masktop Ins:', Masktop)
        MasktopLS = upsample(Masktop)
        # MasktopLS = \
        #    MasktopLS.expand(1, 3, MasktopLS.size(2), \
        #                     MasktopLS.size(3))
        Img_topLS = img.mul(MasktopLS) + \
                    blurred_insert.mul(1 - MasktopLS)
        #print('MasktopLS Ins:', MasktopLS)

        outputstopLS_ori = model(Img_topLS)[0, category].data.cpu().numpy().copy()
        outputstopLS = torch.nn.Softmax()(model(Img_topLS))
        insloss_top2 = outputstopLS[0, category].data.cpu().numpy().copy()
        ins_mask = MasktopLS.clone()

        #ins_ratio = outputstopLS_ori / logitori
        insimg_ratio = imgratio.copy()
        ins_ratio = insloss_top2/outmax
        insert_curve = np.append(insert_curve, insloss_top2)
        #print('ins_ratio:', ins_ratio)
        if ins_ratio >= 0.8:
            print('ins_ratio:', ins_ratio)
            break
    '''


    #cv2.imwrite(output_path +str(outmax)+'_'+str(delloss_top2)+'_'+str(del_ratio)+ "deletion.png", np.uint8(255 * del_img))
    #cv2.imwrite(output_path +str(outmax)+'_'+str(insloss_top2)+'_'+str(ins_ratio)+  "insertion.png", np.uint8(255 * ins_img))
    outmax = np.around(outmax, decimals=3)
    delloss_top2 = np.around(delloss_top2, decimals=3)
    del_ratio = np.around(del_ratio, decimals=3)
    delimg_ratio = np.around(delimg_ratio, decimals=3)

    insloss_top2 = np.around(insloss_top2, decimals=3)
    ins_ratio = np.around(ins_ratio, decimals=3)
    insimg_ratio = np.around(insimg_ratio, decimals=3)



    np.save(os.path.join(output_path, 'data.npy'), (insert_curve, del_curve, category_name))
    return del_mask, ins_mask, delloss_top2, insloss_top2, del_ratio, ins_ratio, outmax, category, xnum, category_name, \
           1 * insert_curve, 1 * del_curve



def write_video(inputpath, outputname, img_num, fps = 10):
    #fps = 10   #视频帧率
    #fourcc = cv2.cv.CV_FOURCC('M','J','P','G')
    # fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    #videoWriter = cv2.VideoWriter('D:/testResults/match/flower2.avi', fourcc, fps, (1360,480))   #(1360,480)为视频大小
    videoWriter = cv2.VideoWriter(outputname, fourcc, fps, (1000, 1000))  # (1360,480)为视频大小
    for i in range(img_num):

        img_no = i+1
        #img12 = cv2.imread('D:/testResults/img_'+str(p1)+'_'+str(p2)+'.jpg')
        print(inputpath+'video'+str(img_no) +'.jpg')
        img12 = cv2.imread(inputpath+'video'+str(img_no) +'.jpg',1)
        #cv2.imshow('img', img12)
    #    cv2.waitKey(1000/int(fps))
        videoWriter.write(img12)
    videoWriter.release()


def predict_category(input_img, resize_shape=(224, 224)):
    # original_img = cv2.imread("." + input_img, 1)
    print(input_img)
    original_img = cv2.imread(input_img, 1)
    # original_img = input_im
    # print('original_img:', original_img)
    original_img = cv2.resize(original_img, resize_shape)
    img = np.float32(original_img) / 255
    model = load_model_new(use_cuda=use_cuda, model_name='vgg19')  #

    img_pr = preprocess_image(img, use_cuda, require_grad=False)
    target = torch.nn.Softmax()(model(img_pr))
    if use_cuda:
        category = np.argmax(target.cpu().data.numpy())
    else:
        category = np.argmax(target.data.numpy())

    f_groundtruth = open('./GroundTruth1000.txt')
    line_i = f_groundtruth.readlines()[category]
    f_groundtruth.close()
    category_name = line_i.split(': ')[1]

    return category_name[1:-3]


def gen_video(input_img_path, mask_gcam=None, output_path_gcam=None):

    mask_gcam = mask_gcam[np.newaxis, np.newaxis, ...]

    # from grad_cam import gcam
    num_images = 72

    input_img = input_img_path
    imgname = os.path.basename(os.path.normpath(input_img))
    print('imgname:', imgname)

    file_id = imgname.split('.')[0]
    if file_id in [str(x) for x in range(1, num_images + 1)]:
        output_path = './static/_igos/{}/'.format(file_id)
    else:
        output_path = './static/_igos/downloaded/{}/'.format(file_id)

    print(output_path)


    if os.path.exists(output_path) and os.path.exists(os.path.join(output_path, 'data.npy')):
        print("igos calculated!")
        # load insertion and deletion
        # TODO
        # if os.path.exists(os.path.join(output_path_gcam, 'data.npy')):

        ins_curve, del_curve, category_name = np.load(os.path.join(output_path, 'data.npy'))
        # ins_curv_gcam, del_curv_gcam, _= np.load(os.path.join(output_path_gcam, 'data.npy'))

        # return ins_curv, del_curv, ins_curv_gcam, del_curv_gcam, output_path[1:], category_name[1:-3]

    else:
        os.makedirs(output_path, exist_ok=True)

        # if not os.path.exists(output_path):
        #     os.makedirs(output_path)

        model = load_model_new(use_cuda=use_cuda, model_name='vgg19')  #

        img_label = -1
        img, blurred_img, logitori = Get_blurred_img(input_img, img_label, model, resize_shape=(224, 224),
                                                     Gaussian_param=[51, 50], Median_param=11, blur_type='Gaussian',
                                                     use_cuda=use_cuda)

        # saliency directory
        # saliency_path = output_path + 'saliency/'
        # if not os.path.isdir(saliency_path):
        #     os.makedirs(saliency_path)

        # saliency_dir = os.path.join(output_path, "_cam.png")





        mask, upsampled_mask, imgratio, curvetop, curve1, curve2, category = Integrated_Mask(img, blurred_img, model,
                                                                                                 img_label,
                                                                                                 max_iterations=15,
                                                                                                 integ_iter=20,
                                                                                                 tv_beta=2,
                                                                                                 l1_coeff=0.01 * 100,
                                                                                                 tv_coeff=0.2 * 100,
                                                                                                 size_init=14,
                                                                                                 use_cuda=1)  #

        #

        save_new_masked(output_path, upsampled_mask , 255 * img, blurred_img)


        # output_file = output_path + 'tmp/'
        # if not os.path.isdir(output_file):
        #     os.makedirs(output_file)


        #scio.savemat(outvideo_path + imgname[:-5] + 'Mask' + '.mat',
        #             mdict={'mask': mask},
        #             oned_as='column')

        #data_all = scio.loadmat(outvideo_path + imgname[:-5] + 'Mask' + '.mat')
        #mask = data_all['mask']

        del_img, ins_img, delloss_top2, insloss_top2, del_ratio, ins_ratio, \
        outmax, cateout, xnum, category_name, ins_curve, del_curve = Deletion_Insertion(mask,
                                                                                       model,
                                                                                       output_path,
                                                                                       img,
                                                                                       blurred_img,
                                                                                       logitori,
                                                                                       category=-1,
                                                                                       pixelnum=200,
                                                                                       use_cuda=1,
                                                                                       blur_mask=0,
                                                                                       outputfig=1)


    if os.path.exists(output_path_gcam) and os.path.exists(os.path.join(output_path_gcam, 'data.npy')):
        print("gcam calculated!")
        # load insertion and deletion
        # TODO
        # if os.path.exists(os.path.join(output_path_gcam, 'data.npy')):

        # ins_curv, del_curv, category_name = np.load(os.path.join(output_path, 'data.npy'))
        ins_curv_gcam, del_curv_gcam, category_name_= np.load(os.path.join(output_path_gcam, 'data.npy'))
    else:

        os.makedirs(output_path_gcam, exist_ok=True)

        model = load_model_new(use_cuda=use_cuda, model_name='vgg19')  #

        img_label = -1
        img, blurred_img, logitori = Get_blurred_img(input_img, img_label, model, resize_shape=(224, 224),
                                                     Gaussian_param=[51, 50], Median_param=11, blur_type='Gaussian',
                                                     use_cuda=use_cuda)

        _, _, _, _, _, _, _, _, _, _, ins_curv_gcam, del_curv_gcam = Deletion_Insertion(mask_gcam,
                                                                                           model,
                                                                                           output_path_gcam,
                                                                                           img,
                                                                                           blurred_img,
                                                                                           logitori,
                                                                                           category=-1,
                                                                                           pixelnum=200,
                                                                                           use_cuda=1,
                                                                                           blur_mask=0,
                                                                                           outputfig=1)


    # # mask_gcam = gcam(input_img_path, category,  1`)
    #
    # del_img, ins_img, delloss_top2, insloss_top2, del_ratio, ins_ratio, outmax, cateout, xnum, category_name, ins_curve_gcam, del_curve_gcam = Deletion_Insertion(mask_gcam,
    #                                                                                                                model,
    #                                                                                                                output_path,
    #                                                                                                                img,
    #                                                                                                                blurred_img,
    #                                                                                                                logitori,
    #                                                                                                                category=-1,
    #                                                                                                                pixelnum=200,
    #                                                                                                                use_cuda=1,
    #                                                                                                                blur_mask=0,
    #                                                                                                                outputfig=1)

    #
    # # xnum =75
    # print(video_dir)
    # write_video(output_file, video_dir, xnum, fps=3)
    #
    # if os.path.isdir(output_file):
    #     shutil.rmtree(output_file)
    #
    # print(video_dir[1:])
    # print(saliency_dir[1:])

    def int_2_str(input_list):
        return ','.join([str(x) for x in input_list])


    return int_2_str(ins_curve), int_2_str(del_curve), int_2_str(ins_curv_gcam), int_2_str(del_curv_gcam),\
           output_path[1:], category_name[1:-3]
    # return ins_curve, del_curve, output_path[1:], category_name[1:-3]


    # You can choose any optimizer
    # The optimizer doesn't matter, because we don't need optimizer.step(), we just to compute the gradient


# ins_del_curve(mask, out_path, )

#
#
if __name__ == '__main__':
    video_name, image_path = dummy.video_image_name('http://am1380.ca/wp-content/uploads/Kitty.jpg')
    gen_video(image_path, video_name)


