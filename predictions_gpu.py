#!/usr/bin/env python
# coding: utf-8

# # Making predictions using the trained model

# ### In this project, it will be applied the U-net model build previously trained

# - Steps:
#     - 0. Select the images to apply the segmentation model
#     - 1. Split the images to four and save
#     - 2. Apply the model to the splitted images
#    

# In[1]:


# import libraries

from skimage import color
import numpy as np
import os
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2
from skimage import data, io, filters
import torch.nn as nn
import torch
import cv2
from PIL import Image, ImageFilter


# ### 0. Select the images to apply the segmentation model

# ### 1. Split the images and save

# In[2]:


# declare the folders' paths

raw_folder = "../02_data_retrain/retrain/test_1_image/"
#"../02_data_3/03_predictions/01_raw images/"
#load the folder name in the variable
#01_raw images


# In[7]:


# create a function to read the raw and b&w images, and filenames

def read_set(directory):
    #read subdirectory
    datas = [] # list of the data
    filenames = [] # list of the filenames
    
    for pathname in os.listdir(directory):
        #create path to subdirectory
        if "IMG" in pathname:
            dataPath = os.path.join(directory,pathname) #join the strings to store file paths
            if (os.path.exists(dataPath)): #function that checks if the specified path exists
                datas.append(io.imread(dataPath)/255.0) #including the images in the data list
                filenames.append(pathname) #including filenames in the list
    return datas,filenames #return the variables in list


# In[8]:


# read the images and store them as array

datas,filename = read_set(raw_folder) 

# using the defined function before, 
# read the original and ground truth images, and the filenames to the lists


# In[9]:


datas


# In[10]:


# check the shape of read images

print("Datas shape = ", datas[0].shape)

# gray scale images have to have 1 color in 3rd channel. So, let us fix it.


# In[11]:


# Creating a single image splitter function

def split(data,size):
    horizontalSplit = data.shape[1]/size[1] # widht of an image divided to width of size(256)
    result = [] #empty list
    if( data.shape[1]%size[1]>0):
        horizontalSplit += 1 #if the rest of division of width to width results is more than 0, add 1.
    horizontalSplit = int(horizontalSplit) # make this number integer
    verticalSplit = data.shape[0]/size[0] # height of an image divided to height of size(256)
    if( data.shape[0]%size[0]>0):
        verticalSplit += 1 #if the rest of division of height to height results is more than 0, add 1.
    verticalSplit = int(verticalSplit) # make this number integer
    
#Until here, we discovered in what number of parts we have to cut the image.
    
    for i in range(0,horizontalSplit): # 0 a 6
        xStart = i *size[1] # 0, 1280
        xEnd = xStart + size[1] # 256, 1536
        if (xEnd > data.shape[1]): # se 256>1762: Mas eh falso
            xEnd = data.shape[1]-1 #  xEnd == 1761
            xStart = xEnd - size[1] # xStart == 1761-256           
        for i in range(0,verticalSplit): # 0 a 13
            yStart = i *size[0] #0, 3328
            yEnd = yStart + size[0] #256, 3584
            if (yEnd > data.shape[0]): # se 256>3456: Mas eh falso
                yEnd = data.shape[0]-1 # yEnd == 3455
                yStart = yEnd - size[0] # yStart == 3455-256 
            result.append(data[yStart:yEnd,xStart:xEnd])
    return result
# This was created a list with 256x256 pixels images of a single complete image in data.


# In[12]:


# This is used to split several images at once.

def imageSplitter(datas,size): # 3 inputs: normal image, ground truth image, size 256x256
    splittedDatas = [] # Empty list to include splitted datas of original image
    for i in range(0,datas.__len__()): #for function to read images in datas one by one
        data = datas[i] #data is the single original image 
        tempData = split(data,size) # split the single image to the tempData
        splittedDatas.extend(tempData) #Add the splitted image to the list
    return splittedDatas #return the lists with splitted images.


# In[47]:


# apply splitting function

size = (216,324) #864,1296 #1728,2592 #Size of chunk of images that we want
# In this case, we want to divide an image to 4 pieces. 
# Thus, the size of width and height are the half of the original

splittedDatas = imageSplitter(datas,size) 
# using the function that was defined before, 


# In[13]:


# creating a list with filenames

new_filename = []

for i in filename:
    for j in range(1,257): #originally 5
        new = str(i[:8])+"_"+str(j)+'.png' # adding numerations for splitted images
        new_filename.append(new)
print(new_filename)


# In[49]:


# Create a temporary filename to plot the images

temp_filename = []

for i in new_filename:
    i = str(i)
    temp_filename.append(i[0:8])
    
print(temp_filename)


# In[14]:


# print some images to visualize

for x in range(0,16,4): # for each 4 images 
    plt.figure(figsize=(20,10)) # create a space for the images 16(width) 8(height) 
    # plt.suptitle("Splitted Images", fontsize=14, fontweight='bold') #put a superior title
    
    f, ax = plt.subplots(2,2)
    plt.tight_layout()
    plt.suptitle(temp_filename[x], fontsize=8, fontweight='bold') #put a superior title

    ax[0,0].imshow(splittedDatas[x])
    ax[0,1].imshow(splittedDatas[x+2])
    ax[1,0].imshow(splittedDatas[x+1])
    ax[1,1].imshow(splittedDatas[x+3])
    
    del x, f, ax


# In[51]:


# Save the Raw splitted images into the folder "02_cut images".
path_cut = "../02_data_retrain/retrain/gt_cut/"
#"../02_data_retrain/02_rope segmentation/01_train/train_raw/" 
#"../02_data_3/03_predictions/02_cut images/"

for i in range(0, len(splittedDatas)):
    if os.path.exists(os.path.join(path_cut, new_filename[i])) == False:
        plt.imsave(path_cut+str(new_filename[i]), splittedDatas[i],cmap="gray")
        
        #02_cut images


# ### 2. Apply the model to the splitted images

# #### Take the cut images and apply to the segmentation function

# In[15]:


# import the models used in training

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


# In[16]:


# import the models used training

class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.final_conv(x)


# In[17]:


# define the function to load the trained model

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


# #### Applying the model to each images

# In[20]:


# Apply an UNET to the model once
model = UNET(in_channels=3, out_channels=1)

# Apply the function to load effectively
load_checkpoint(torch.load("my_checkpoint_epoch100.pth.tar"), model)

# defining evaluation mode
trained_model = model.to("cpu")
trained_model.eval()


# In[42]:


# Declare the dataset transformner
# Normalize the values and transform to tensors

test_transform = A.Compose([A.Resize(1728,2592), #3456, 5184
                           A.Normalize(mean=(0,0,0),std=(1,1,1),max_pixel_value=255),
                           ToTensorV2()])


# In[18]:


# Select the path of the images to pass through model

pred_path = "../02_data_3/03_predictions/02_cut images/" #02_cut images
predicted_path = "../02_data_3/03_predictions/03_predicted images/" #03_predicted images


# In[44]:


# Putting images through model to segment rope


# iterate over all images
for im in os.listdir(pred_path):
    if os.path.exists(os.path.join(predicted_path, im)) == False:
        if "IMG" in im:
            img_pred = cv2.imread(os.path.join(pred_path,im))
            #change image color format
            #img_pred = cv2.cvtColor(img_pred, cv2.COLOR_RGB2RGB)
            img_pred = cv2.cvtColor(img_pred, cv2.COLOR_RGB2BGR)
            # apply the transform to the image
            test_image = test_transform(image = img_pred)
            del img_pred

            img = test_image["image"].unsqueeze(0)
            img = img.to("cpu") #to run in cpu
            del test_image

            pred = model(img)
            del img

            # configurating mask
            mask = pred.squeeze(0).cpu().detach().numpy()
            # print(mask.shape)
            mask = mask.transpose(1,2,0)
            del pred

            # setting the mask: above 0 is white, below zero is black
            mask[mask<0]=0
            mask[mask>0]=1

            mask = mask[:, :, 0]
            mask *= 255
            mask = mask.astype(np.uint8)
            mask = Image.fromarray(mask).convert('RGB') 
            mask.save(predicted_path+str(im))
            
            del mask

            # Then, the segmented images will be saved in the folder "03_predicted images"


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




