#!/usr/bin/env python
# coding: utf-8

# # Vanilla UNet on the Carvana image dataset
# ----
# 
# Data: https://www.kaggle.com/c/carvana-image-masking-challenge/data

# In[ ]:


# %matplotlib inline
# import matplotlib.pyplot as plt

import logging
import unet
from image_dataset import ImageMaskDataset
import multiprocessing as mp
import torch
import os
import numpy as np
from torch.utils.data import random_split, DataLoader
import logging
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device: {device}')
ncpus = mp.cpu_count()
print(f'Using {ncpus} cpu''s')


# In[ ]:


class VanillaUNetCarvana:
    
    BATCH_SIZE = 4
    LEARNING_RATE = 0.1
    PAD_SIZE = (572-388) // 2
    
    def __init__(self, 
                 root_path, 
                 chk_folder,
                 split=[0.7, 0.2, 0.1]):
        
        # The problem with this Vanilla Ronnenberger UNet, the input and output size does
        # not match. Input is (572,572), output is (388,388).
        # To make the loss function works properly, the dataset will create 388x388 image & masks.
        # Then pad the image to get 572x572 size before feeding it to the network.
        ds = ImageMaskDataset.from_folder(
            size=(388,388),
            image_folder=os.path.join(root_path, 'train'),
            image_pattern='*.jpg',
            mask_fname_fun=lambda x: os.path.join(
                root_path, 'train_masks', x.replace('.jpg', '_mask.gif'))
        )
        print(f'The total number of images in the dataset: {len(ds)}')
        
        # we're going to split the dataset into training & test
        n_split = [int(item * len(ds)) for item in split]
        if sum(n_split)<len(ds):
            n_split[-1] += len(ds) - sum(n_split)
        assert sum(n_split)==len(ds), "Sum of split must equal to 1.0"
        
        # split
        sub_ds = random_split(ds, n_split)
        
        # create data loader
        self.train_loader = DataLoader(
            sub_ds[0], batch_size=self.BATCH_SIZE, shuffle=True, num_workers=ncpus, pin_memory=True
        )
        print(f"Training samples are {len(sub_ds[0])} or {100.0*len(sub_ds[0])/len(ds):.2f}%")
        self.val_loader = DataLoader(
            sub_ds[1], batch_size=self.BATCH_SIZE, shuffle=True, num_workers=ncpus, pin_memory=True
        )
        print(f"Validation samples are {len(sub_ds[1])} or {100.0*len(sub_ds[1])/len(ds):.2f}%")
        self.test_loader = DataLoader(
            sub_ds[2], batch_size=self.BATCH_SIZE, shuffle=False, num_workers=ncpus, pin_memory=True
        )
        print(f"Test samples are {len(sub_ds[2])} or {100.0*len(sub_ds[2])/len(ds):.2f}%")
        
        # create the model
        self.model = unet.VanillaUNet(3, 2).to(device)
        
        # optimizers & criterion
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), 
                                             lr=self.LEARNING_RATE,
                                             weight_decay=1e-8, momentum=0.9)
        self.criterion = torch.nn.BCEWithLogitsLoss()
        
        # checkpoints
        if not os.path.isdir(chk_folder):
            os.makedirs(chk_folder)
        self.chk_folder = chk_folder
        logging.basicConfig(
            filename=os.path.join(self.chk_folder, 'training.log'),
            level=logging.DEBUG)
        
    def train(self, num_epoch):
        
        logging.info('Start training:')
        logging.info(f'  Number of epochs:            {num_epoch}')
        logging.info(f'  Batch size:                  {self.BATCH_SIZE}')
        logging.info(f'  Learning rate:               {self.LEARNING_RATE}')
        logging.info(f'  Training/Validate/Test size: {len(self.train_loader)}/{len(self.val_loader)}/{len(self.test_loader)}')
        
        for epoch in range(num_epoch):
            # 1. Don't forget to set training mode for the model
            self.model.train()
            
            with tqdm(total=len(self.train_loader.dataset)) as pbar:
            
                last_loss = 0.00
                for step, (img, mask) in enumerate(self.train_loader):
                    # 2. Adjust the device. Otherwise it will raise unmatching type error.
                    img = img.to(device)
                    mask_true = mask.to(device)

                    # 2.5 Add the input image to get 572x572. Leave the mask_true
                    img = torch.nn.functional.pad(img, ([self.PAD_SIZE] * 4))

                    # 3. Predict
                    mask_pred = self.model(img)

                    # 4. Calculate the loss
                    loss = self.criterion(mask_pred, mask_true)
                    loss_diff = last_loss - loss.item()
                    last_loss = loss.item()

                    # 5. Backward processing
                    loss.backward()

                    # 6. Then forward step
                    self.optimizer.step()
                    
                    # update the progress bar
                    pbar.set_description(f"Epoch {epoch}")
                    pbar.set_postfix(
                        loss=f"{loss.item():.2f} ({loss_diff:.1f})"
                    )
                    pbar.update(n=img.shape[0])
                
        print("FINISHED TRAINING")


# In[ ]:


vunc = VanillaUNetCarvana(r'/tmp/carvana/', r'/tmp/chkpts/carvana/')


# In[ ]:


mask_pred, mask_true = vunc.train(1)


# In[ ]:




