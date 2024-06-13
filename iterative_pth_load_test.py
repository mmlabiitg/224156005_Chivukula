from models import CC_Module
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import os
import numpy as np
import time
from options import opt
import math
from measure_ssim_psnr import *
import shutil
from tqdm import tqdm

CHECKPOINTS_DIR = opt.checkpoints_dir
INP_DIR = opt.testing_dir_inp
CLEAN_DIR = opt.testing_dir_gt
DEPTH_DIR = opt.testing_dir_inp

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'        

ch = 3

network = CC_Module()


result_dir = '../River-LSUI/output-rest/'
if not os.path.exists(result_dir):
    os.makedirs(result_dir)



'''
files = os.listdir(folder_path)

# Count number of files
num_files = len(files)
'''

def fun():
    with torch.no_grad():
        total_files = os.listdir(INP_DIR)
        st = time.time()
        with tqdm(total=len(total_files)) as t:

            for m in total_files:
            
                img = cv2.resize(cv2.imread(INP_DIR + str(m)), (256,256), cv2.INTER_CUBIC)
                img = img[:, :, ::-1]   
                img = np.float32(img) / 255.0
                h,w,c=img.shape

                train_x = np.zeros((1, ch, h, w)).astype(np.float32)

                train_x[0,0,:,:] = img[:,:,0]
                train_x[0,1,:,:] = img[:,:,1]
                train_x[0,2,:,:] = img[:,:,2]
                dataset_torchx = torch.from_numpy(train_x)
                dataset_torchx=dataset_torchx.to(device)

                img = cv2.resize(cv2.imread(DEPTH_DIR + str(m)), (256,256), cv2.INTER_CUBIC)
                img = img[:, :, ::-1]   
                img = np.float32(img) / 255.0
                h,w,c=img.shape

                train_x = np.zeros((1, ch, h, w)).astype(np.float32)

                train_x[0,0,:,:] = img[:,:,0]
                train_x[0,1,:,:] = img[:,:,1]
                train_x[0,2,:,:] = img[:,:,2]
                depth_torchx = torch.from_numpy(train_x)
                depth_torchx=depth_torchx.to(device)

                output=network(dataset_torchx, depth_torchx)
                output = (output.clamp_(0.0, 1.0)[0].detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
                output = output[:, :, ::-1]
                cv2.imwrite(os.path.join(result_dir + str(m)), output)

                t.set_postfix_str("name: {} | old [hw]: {}/{} | new [hw]: {}/{}".format(str(m), h,w, output.shape[0], output.shape[1]))
                t.update(1)
                
        end = time.time()
        print('Total time taken in secs : '+str(end-st))
        print('Per image (avg): '+ str(float((end-st)/len(total_files))))

        ### compute SSIM and PSNR
        SSIM_measures, PSNR_measures = SSIMs_PSNRs(CLEAN_DIR, result_dir)
        print("SSIM on {0} samples".format(len(SSIM_measures))+"\n")
        print("Mean: {0} std: {1}".format(np.mean(SSIM_measures), np.std(SSIM_measures))+"\n")
        print("PSNR on {0} samples".format(len(PSNR_measures))+"\n")
        print("Mean: {0} std: {1}".format(np.mean(PSNR_measures), np.std(PSNR_measures))+"\n")
        return np.mean(SSIM_measures), np.std(SSIM_measures),np.mean(PSNR_measures), np.std(PSNR_measures)
        #inp_uqims = measure_UIQMs(result_dir)
        #print ("Input UIQMs >> Mean: {0} std: {1}".format(np.mean(inp_uqims), np.std(inp_uqims)))
    # shutil.rmtree(result_dir)

if __name__ =='__main__':

    start = 20
    file_path = 'log3.txt'

    BEST_PSNR = [0.0, 0.0]
    BEST_SSIM = [0.0, 0.0]
    BEST_PSNR_EPOCH = -1
    BEST_SSIM_EPOCH = -1
    if os.path.exists(file_path):
        # If file exists, delete it
        os.remove(file_path)

    for i in range(252, 254):
        print('EPOCH : ', i)
        print()
        checkpoint = torch.load(os.path.join(CHECKPOINTS_DIR,"netG_{}.pt".format(i)))
        network.load_state_dict(checkpoint['model_state_dict'])
        network.eval()
        network.to(device)

        a, b, c, d = fun()

        if c > BEST_PSNR[0] :
            BEST_PSNR = [c, d]
            BEST_PSNR_EPOCH = i 
        if a > BEST_SSIM[0]:
            BEST_SSIM = [a, b]
            BEST_SSIM_EPOCH = i

        with open(file_path, mode='a') as file:
            file.write('EPOCH {} : PSNR {} + {}, SSIM : {} + {}'.format(i, c, d, a, b))
            file.write('\n')


    with open(file_path, mode='a') as file:
        file.write('BEST PSNR EPOCH {} & BEST SSIM EPOCH {}: PSNR {} + {}, SSIM : {} + {}'.format(BEST_PSNR_EPOCH, BEST_SSIM_EPOCH, BEST_PSNR[0], BEST_PSNR[1], BEST_SSIM[0], BEST_SSIM[1]))
        file.write('\n')



    
