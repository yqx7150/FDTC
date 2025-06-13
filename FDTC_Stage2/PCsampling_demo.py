#@title Autoload all modules


from dataclasses import dataclass, field
import matplotlib.pyplot as plt
import io
import csv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import importlib
import os
import functools
import itertools
import torch
from losses import get_optimizer
from models.ema import ExponentialMovingAverage

import torch.nn as nn
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_gan as tfgan
import tqdm
import io
import likelihood
import controllable_generation
from utils import restore_checkpoint
sns.set(font_scale=2)
sns.set(style="whitegrid")

import models
from models import utils as mutils
from models import ncsnv2
from models import ncsnpp
from models import ddpm as ddpm_model
from models import layerspp
from models import layers
from models import normalization
import sampling2
from likelihood import get_likelihood_fn
from sde_lib import VESDE, VPSDE, subVPSDE
from sampling2 import (ReverseDiffusionPredictor, 
                      LangevinCorrector, 
                      EulerMaruyamaPredictor, 
                      AncestralSamplingPredictor, 
                      NoneCorrector, 
                      NonePredictor,
                      AnnealedLangevinDynamics)
import datasets
import os.path as osp
import time
import scipy.io as sio
import imageio
import cv2
from packages.ffdnet.models import FFDNet
import scipy.io as sio
from packages.ffdnet.test_ffdnet_ipol import ffdnet_vdenoiser
from tvdenoise import tvdenoise

net = FFDNet(num_input_channels=1).cuda()
model_fn = 'packages/ffdnet/models/net_gray.pth'
state_dict = torch.load(model_fn)
net = torch.nn.DataParallel(net, device_ids=[0]).cuda()
net.load_state_dict(state_dict)


# @title Load the score-based model
sde = 'VESDE' #@param ['VESDE', 'VPSDE', 'subVPSDE'] {"type": "string"}
if sde.lower() == 'vesde':
  from configs.ve import SIAT_kdata_ncsnpp as configs  # 修改config
  #from configs.ve import bedroom_ncsnpp_continuous as configs  # 修改config
  #ckpt_filename = "exp/ve/cifar10_ncsnpp_continuous/checkpoint_24.pth"
  model_num = 'checkpoint.pth'
  #ckpt_filename ='./exp1/checkpoints/checkpoint_40.pth'#41 54 65
  ckpt_filename ='./exp/checkpoints/checkpoint_28.pth'#41 54 65
  #ckpt_filename ='./exp/checkpoints/checkpoint_20.pth'#41 54 65
  config = configs.get_config()  
  sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales) ###################################  sde
  #sde = VESDE(sigma_min=0.01, sigma_max=5000, N=600) ###################################  sde
  sampling_eps = 1e-5


batch_size = 1 #@param {"type":"integer"}
config.training.batch_size = batch_size
config.eval.batch_size = batch_size

random_seed = 0 #@param {"type": "integer"}

sigmas = mutils.get_sigmas(config)
scaler = datasets.get_data_scaler(config)
inverse_scaler = datasets.get_data_inverse_scaler(config)
score_model = mutils.create_model(config)

optimizer = get_optimizer(config, score_model.parameters())
ema = ExponentialMovingAverage(score_model.parameters(),
                               decay=config.model.ema_rate)
state = dict(step=0, optimizer=optimizer,
             model=score_model, ema=ema)

state = restore_checkpoint(ckpt_filename, state, config.device)
ema.copy_to(score_model.parameters())

#@title PC sampling
img_size = config.data.image_size
channels = config.data.num_channels
shape = (batch_size, channels, img_size, img_size)
predictor = ReverseDiffusionPredictor #@param ["EulerMaruyamaPredictor", "AncestralSamplingPredictor", "ReverseDiffusionPredictor", "None"] {"type": "raw"}
corrector = LangevinCorrector #@param ["LangevinCorrector", "AnnealedLangevinDynamics", "None"] {"type": "raw"}
snr = 0.075#0.16 #@param {"type": "number"}
n_steps =  1#@param {"type": "integer"}
probability_flow = False #@param {"type": "boolean"}

'''
sampling_fn = sampling2.get_pc_sampler(sde, shape, predictor, corrector,
                                      inverse_scaler, snr, n_steps=n_steps,
                                      probability_flow=probability_flow,
                                      continuous=config.training.continuous,
                                      eps=sampling_eps, device=config.device)

x, n = sampling_fn(score_model)
'''

magnitudes_oversampled =imageio.imread(r'./measurement/result_0.png')#[0:800,:800]
#magnitudes_oversampled = cv2.resize(magnitudes_oversampled,(256,256))     ###########
#magnitudes_oversampled=cv2.medianBlur(magnitudes_oversampled,3)#median filter (optional)
#magnitudes_oversampled=np.pad(magnitudes_oversampled, 200, 'constant')

trial=1
percent_list=[0.001,0.002,0.003,0.004,0.005,0.006]#tunable list for different input
#percent_list=[0.005,0.006]
for k in range(trial):
  steps=600
  #steps=60
  for percent in percent_list:
    start=time.time()
    sampling_fn = sampling2.get_pc_sampler(sde, shape, predictor, corrector,
                                      inverse_scaler, snr, n_steps=n_steps,
                                      probability_flow=probability_flow,
                                      continuous=config.training.continuous,
                                      eps=sampling_eps, device=config.device)
    Reconstruct_Field,a,b,c=sampling_fn(score_model,np.fft.ifftshift(magnitudes_oversampled),40,-4,0,steps,np.random.rand(*magnitudes_oversampled.shape)+1j*np.random.rand(*magnitudes_oversampled.shape),percent,net)
    Reconstruct_Image_Inten = np.power(np.abs(Reconstruct_Field),2)
    Reconstruct_Image_Amp = np.abs(Reconstruct_Field)
    end=time.time()
    print("Saving_{}_time:{}s".format(k,end-start))
    sio.savemat(r'./result_0/New_result_num={}_percent={}.mat'.format(k, percent),{'result': Reconstruct_Field})
    cv2.imwrite(r'./result_0/New_FFD_ML_result_percent={}_k={}.png'.format(percent,k),Reconstruct_Image_Amp/np.max(Reconstruct_Image_Amp)*255)
'''
x = x.detach().cpu().numpy() # (1,3,256,256)

for ii in range(batch_size):      
  kw_real = (x[ii,0,:,:]+x[ii,2,:,:])/2
  kw_imag = x[ii,1,:,:]

  k_w = kw_real+1j*kw_imag
  image = np.fft.ifft2(k_w)
  max_ = np.max(np.abs(k_w))
  min_ = np.min(np.abs(k_w))
  save_kImage(k_w,'./result/sample/',"sample_"+"max="+str(max_)+"min="+str(min_)+".png")
  save_kImage(image,'./result/sample/',"sample_ifft2"+str(ii)+".png") 
'''
