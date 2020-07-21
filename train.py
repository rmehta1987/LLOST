from __future__ import print_function
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import itertools
from tqdm import tqdm
import json
from latent_align_modules import GlowLatent, GaussianDiag

''' Get parameters from params.json'''

config = json.loads(open('params.json', 'r').read())
data_path = config['pathToData']
vocab_path = config['vocab_path']
noise_im = int(config['noise_im'])
noise_txt = int(config['noise_txt'])
embed_size = int(config['embed_size'])
max_length = int(config['max_length'])
num_encoder_tokens = int(config['num_encoder_tokens'])
num_decoder_tokens= int(config['num_decoder_tokens'])
word_dim = int(config['word_dim'])
batch_size = int(config['batch_size'])
num_gpus = int(config['num_gpus'])
img_dim = int(config['img_dim'])
epochs = int(config['epochs'])
lambda_1 = config['lambda_1']
lambda_2 = config['lambda_2']
lambda_3 = config['lambda_3']
lambda_4 = config['lambda_4']
lambda_5 = config['lambda_5']
device = torch.device("cuda:0")


def get_labels_NLL_loss(seq,logp):  
  target = seq[:,1:]
  target = target[:, :torch.max(seq_lengths).item()].contiguous().view(-1)
  preds = logp.clone().detach().cpu().numpy()
  logp = logp.view(-1, logp.size(2))
  return NLL(logp, target), preds

def adjust_learning_rate(optimizer, epoch,lr):
  """Sets the learning rate to the initial LR decayed by 0.5 every 30 epochs"""
  lr = lr * (0.5 ** (epoch // 25))
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr


def autoencode_features(encoder, decoder, glow_latent_image, im_batch_gan, im_batch_vgg, z_im_text2img ):
  '''Build image pipeline '''
  encoded = encoder(im_batch_vgg)
  _encoded = encoded[:,:] + torch.randn(encoded.size(),device=device)*0.05
  encoded[:,img_dim:] = encoded[:,img_dim:] + torch.randn(batch_size,noise_im,device=device)*0.05 
  decoded = decoder(t2i = _encoded[:,:img_dim], z = encoded[:,img_dim:])

  rec_loss = torch.sum(torch.abs( im_batch_gan - decoded ), dim=(1,2,3)) / batch_size

  _, nll, _ = glow_latent_image(encoded[:,img_dim:].to(device), cond=z_im_text2img, reverse=False)

  return encoded, rec_loss, decoded ,nll

'''define models'''
feature_encoder = FeatEncoder().cuda()
feature_decoder = FeatDecoder().cuda()
feature_encoder =  nn.DataParallel(feature_encoder).cuda()
feature_decoder =  nn.DataParallel(feature_decoder).cuda()

labelEncoder = LabelEncoder()
labelEncoder = nn.DataParallel(labelEncoder).cuda()
labelDecoder = LabelDecoder().cuda()


glow_feat_cond = GlowLatent(batch_size=batch_size,input_dim=noise_txt,hidden_channels=1024,K=16,gaussian_dims=noise_txt,gaussian_var=0.25,cond_dim=img_dim,coupling='full')
glow_feat_cond = glow_text_cond.to(device)

glow_latent_align = GlowLatent(batch_size//num_gpus,img_dim,hidden_channels=1024,K=12,gaussian_dims=(0),gaussian_var=0,coupling='linear').cuda()
glow_latent_align= nn.DataParallel(glow_latent_align).cuda()
glow_latent_image = GlowLatent(batch_size,noise_im,hidden_channels=512,gaussian_dims=noise_im,gaussian_var=0.25,cond_dim=img_dim,coupling='full').cuda()



'''define optimizers'''
optimizerG_align = optim.Adam(glow_latent_align.parameters(),lr=0.0001)
optimizerG_image = optim.Adam(glow_latent_image.parameters(),lr=0.0001)
optimizerF_e = optim.Adam(feature_encoder.parameters(),lr=0.00001, betas=(0.0, 0.9))#, betas=(0.0, 0.9)
optimizerF_d = optim.Adam(feature_decoder.parameters(),lr=0.0001, betas=(0.0, 0.9))#, betas=(0.0, 0.9)
optimizerL_e = optim.Adam(labelEncoder.parameters(),lr=0.00001, betas=(0.0, 0.9))
optimizerL_d = optim.Adam(labelDecoder.parameters(),lr=0.00001, betas=(0.0, 0.9))

NLL = nn.NLLLoss(size_average=False, ignore_index=0)

for epoch in range(epochs):
  adjust_learning_rate(optimizerI_e, epoch, 0.00001)
  for i in tqdm(range(len(dataset)//batch_size)): 

    txtEncoder.zero_grad()
    txtDecoder.zero_grad()
    glow_latent_image.zero_grad()
    glow_latent_align.zero_grad()
    image_encoder.zero_grad()
    image_decoder.zero_grad()
    disc.zero_grad()
    glow_text_cond.zero_grad()
    optimizerF.zero_grad()
    optimizerG_image.zero_grad()
    optimizerG_align.zero_grad()
    optimizerI_e.zero_grad()
    optimizerI_d.zero_grad()
    optimizerD.zero_grad()
    optimizerG_cond_text.zero_grad()

    # Get Data
    try:
      data = next(dataloader_iterator)
    except StopIteration:
      dataloader_iterator = iter(dataloader)  
      data = next(dataloader_iterator)

    img_gan, img_vgg, captions = data # cap has shape batch_size*10
    captions_batch = get_sentence_targets(captions)
    seq, seq_lengths, sort_array, img_gan, img_vgg = get_properseq(captions_batch, img_gan, img_vgg)
    txtencoded_hidden = txtEncoder(seq,seq_lengths)
    txtencoded_hidden =  txtencoded_hidden+ 0.05*torch.randn(txtencoded_hidden.size()).to(device)
    txtencoded_hidden[:,img_dim:] =  txtencoded_hidden[:,img_dim:] + 0.10*torch.randn(txtencoded_hidden[:,img_dim:].size()).to(device)
    seq_lengths = seq_lengths - 1;

    if i%discriminator_iter == 0:
      logp = txtDecoder(seq,txtencoded_hidden,seq_lengths)
      NLL_loss, preds = get_sentence_NLL_loss(seq,logp)
    
    z, nll, _ = glow_latent_align(x=txtencoded_hidden[:,:img_dim].to(device), z_im=None, z=None, cond=None, eps_std=None, reverse=False) 
    z_im_text2img = z[:,:img_dim]

    z_im_full, image_rec_loss, z_im_true,nll_im = autoencode_image( image_encoder, image_decoder, glow_latent_image, img_gan.cuda(), img_vgg.cuda(), z_im_text2img )
    z_im_full = z_im_full[:,:].cuda()
    z_im = z_im_full[:,:img_dim]
    z_rev, _ = glow_latent_align(x=z_im.to(device), z_im=None, z=None, cond=None, eps_std=None, reverse=True)
    z_text, nll_text_cond, _  = glow_text_cond(x=txtencoded_hidden[:,img_dim:].to(device), z_im=None, z=None, cond=z_rev[:,:img_dim].to(device), eps_std=None, reverse=False)
    
    if i%discriminator_iter == 0:
      _z_rev = torch.cat((z_rev,txtencoded_hidden[:,img_dim:]),dim=1)
      logp = txtDecoder(seq,_z_rev.cuda(),seq_lengths)
      img2txt_loss, preds_inf_latent = get_sentence_NLL_loss(seq,logp)


    cond_loss_forward = F.mse_loss(z[:,:img_dim],z_im.to(device))
    if i%discriminator_iter == 0:
      z_im_decoded = image_decoder(t2i = z_im_text2img, z = z_im_full[:,img_dim:] )#z_im_text2img.view(z_im_text2img.size(0),z_im_text2img.size(1),1,1))#.clone().detach()
      text2img_loss = torch.sum(torch.abs( img_gan.cuda() - z_im_decoded.cuda() ), dim=(1,2,3)) / batch_size
      

    if i%discriminator_iter != 0:

      rev, _ = glow_latent_image(x=None,z_im=None, z=None, cond=z_im_text2img.to(device), eps_std=None, reverse=True)
      z_im_decoded = image_decoder(t2i = z_im_text2img, z = rev )
      out_fake = disc(z=z_im_decoded.detach(),t2i =txtencoded_hidden[:,:img_dim].detach()).view(-1).cuda()
      err_D_fake_tx = nn.ReLU()(1.0 - out_fake).mean()

      try:
        data = next(dataloader_iterator)
      except StopIteration:
        dataloader_iterator = iter(dataloader)  
        data = next(dataloader_iterator)

      img_gan, _, _ = data
      out_real = disc(z=img_gan.float().to(device),t2i=txtencoded_hidden[:,:img_dim].detach() ).view(-1).cuda()
      err_D_real = nn.ReLU()(1.0 + out_real).mean()
      err_D = (err_D_fake_tx + err_D_real).to(device)
      err_D.backward()
      optimizerD.step()

    else: 
      err_G_tx = disc(z=z_im_decoded, t2i=txtencoded_hidden[:,:img_dim].detach()).view(-1).cuda()
      err_G = err_G_tx
      mse_loss = 25.0*torch.mean(text2img_loss).to(device)+ 1.2*torch.mean(img2txt_loss).to(device)
      '''err_G, text2img_loss, img2txt_loss for training in reverse direction '''
      loss = lambda_1*torch.mean(nll_text_cond).to(device)+\
        lambda_2*torch.mean(nll_im).to(device)+\
        lambda_3*torch.mean(nll).to(device)+\
        lambda_4*NLL_loss.to(device)+\
        lambda_5*(torch.mean(image_rec_loss).to(device) + 25.0*torch.mean(err_G).to(device))+\
        mse_loss
        


      loss.backward()
      torch.nn.utils.clip_grad_value_(txtEncoder.parameters(), 1.0)
      torch.nn.utils.clip_grad_value_(glow_latent_align.parameters(), 1.0)
      torch.nn.utils.clip_grad_value_(glow_latent_image.parameters(), 5.0)
      torch.nn.utils.clip_grad_value_(txtDecoder.parameters(), 1.0)
      torch.nn.utils.clip_grad_value_(glow_text_cond.parameters(), 1.0)
      optimizerF.step()
      optimizerG_align.step()
      optimizerG_image.step()
      optimizerI_e.step()
      optimizerI_d.step()
      optimizerG_cond_text.step()

    if i%100==0:
      print(epoch)
      print('loss:'+ str(loss),'GRU_loss:'+str(NLL_loss),
        'Cond loss image: ' + str(torch.mean(image_rec_loss).item()),
        'Glow loss: ' + str(torch.mean(nll).item()),
        'Cond loss text2img: ' + str(torch.mean(text2img_loss).item()),
        'Cond loss img2txt:' + str(torch.mean(img2txt_loss).item()),
        'mse_im:' + str(torch.mean(cond_loss_forward).item()),
        'Cond loss txt:' + str(torch.mean(nll_text_cond).item()),
        'nll_im:' + str(torch.mean(nll_im).item()))

    if i%300==0:
      torch.save({
            'image_encoder_sd': image_encoder.module.state_dict(),
            'image_decoder_sd': image_decoder.module.state_dict(),
            'glow_latent_align_sd': glow_latent_align.module.state_dict(),
            'glow_latent_image_sd': glow_latent_image.state_dict(),
            'glow_text_cond_sd': glow_text_cond.state_dict(),
            'txtEncoder_sd': txtEncoder.module.state_dict(),
            'txtDecoder_sd': txtDecoder.state_dict(),
            'disc_sd': disc.module.state_dict(),
            'optimizer_image_encoder_sd': optimizerI_e.state_dict(),
            'optimizer_image_decoder_sd': optimizerI_d.state_dict(),
            'optimizer_glow_latent_align_sd': optimizerG_align.state_dict(),
            'optimizer_glow_latent_image_sd': optimizerG_image.state_dict(),
            'optimizer_glow_text_cond_sd': optimizerG_cond_text.state_dict(),
            'optimizer_txt_sd': optimizerF.state_dict(),
            'optimizerD_sd' : optimizerD.state_dict()
            }, './model_checkpoint.pt')





