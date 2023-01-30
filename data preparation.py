# # Data preparation for the image segmentation

# ### This is the process of the creation of GT images from Raw Images

# - Steps:
#     1. Convert all the images to .png format
#     2. Apply the Color Detection
#     3. Paint the unnecessary white noise with black pencil manually
#     4. Tranform the GT images to PNG format and Gray scale
#     5. Read the images to prepare for image splitting (data augmentation for Segmentation training)
#     6. Split the Raw and Ground truth images
#     7. Save the images as png image type in folder

# ______

# ### 1. Convert all the images to .png format

# ### CONFIGURATION 

# importing the necessary libraries

from os import listdir #list all files and folder inside a directory
import os #operating system --> You can create a folder, delete or whatever you want inside one specific directory
from os.path import join # join will make 2 strings join 
from skimage import io,transform #modules to read and write images in various formats
from skimage.color import rgb2gray #skimage module to color conversion
import matplotlib.pyplot as plt #matplotlib to plot graphs
get_ipython().run_line_magic('matplotlib', 'inline')
#python magic line to enable showing graphs right below the codes and store it
import cv2 #csv editing library
import numpy as np #numpy
from PIL import Image #Python image editing library
import pickle


# Check if the path exists.

# Specify path
path = "../02_data_retrain/01_data preparation/01_downloaded"
  
# Check whether the specified path exists or not
isExist = os.path.exists(path)
if isExist == True: 
    print("The specified folder exists") 
elif isExist == False:
    print("The specified folder does not exist. So, the folder /01_downloaded was created. Please, put the downloaded images in this folder.") 
    os.mkdir("../02_data/01_data preparation/01_downloaded")

# convert the images in raw folder

# import library
from PIL import Image

# declare the paths
path1 = "../02_data_retrain/01_data preparation/01_downloaded"
path2 = "../02_data_retrain/01_data preparation/02_raw/"

# check if the paths exist already. if not, create one.
if os.path.exists(path2) == False:
    os.mkdir("../02_data_retrain/01_data preparation/02_raw")

# convert to .png format
for img in os.listdir(path1):
    #print(img)
    if ".JPG" in img: #only images
        im = Image.open(os.path.join(path1, img))
        png = img[:8]+".png"
        im.save(os.path.join(path2, png))


# ### 2. Apply the Color Detection

# Detecting blue color things in raw images

path = "../02_data_retrain/01_data preparation/02_raw/" #path of raw images
path_seg = "../02_data_retrain/01_data preparation/03_detected_blue/"

# check if the paths exist already. if not, create one.
if os.path.exists(path2) == False:
    os.mkdir("../02_data_retrain/01_data preparation/03_detected_blue/")

# Read the images and apply the color detection
for imgs in os.listdir(path):
    if "IMG" in imgs:
        img = cv2.imread(os.path.join(path, imgs)) #read the images

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #convert the image format from BGR to HSV

        #######################################################
        # Change the configuration of the color if necessary. #
        #######################################################

        # color configuration
        low_blue = np.array([90, 100, 0]) # H [0,179], S [0,255], V [0,255] 
        high_blue = np.array([110, 255, 255])

        # masking
        blue_mask = cv2.inRange(hsv, low_blue, high_blue) 

        # generating 
        blue = cv2.bitwise_and(img, img, mask=blue_mask)

        # Putting a filter
        blur = cv2.GaussianBlur(blue, (7, 7), 0)

        img_name = imgs
        path_last = os.path.join(path_seg, img_name)

        cv2.imwrite(path_last, blur) # saved into the folder "03_detected_blue"

# ### 3. Paint the unnecessary white noise with black pencil manually

# - Import the image and paint the image with black pencil manually.
# - In this case, it was used the GIMP software.
# - Then, save the images into the folder "04_fixed_manually".

# ### 4. Tranform the GT images to PNG format and Gray scale

# convert the images just to make sure that the files are in the same format

path1 = "../02_data_3/01_data preparation/02_raw"

for img in os.listdir(path1):
    #print(img)
    im = Image.open(os.path.join(path1, img))
    png = img[:8]+".png"
    im.save(os.path.join(path1, png))

# transform all the png images to gray scale

path1 = "../02_data_retrain/01_data preparation/04_fixed_manually"
path2 = "../02_data_retrain/01_data preparation/05_gt"

for imgs in os.listdir(path1):
    img = cv2.imread(os.path.join(path1, imgs))
    print(img.shape)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    path_fix = os.path.join(path2, imgs)
    print(gray_img.shape)

    cv2.imwrite(path_fix, gray_img)
    


# ### 5. Read the images to prepare for image splitting (data augmentation for Segmentation training)

# declare the folders' paths
raw_folder = "../02_data_retrain/01_data preparation/02_raw/" #load the folder name in the variable
gt_folder = "../02_data_retrain/01_data preparation/05_gt/" #load the folder name in the variable


# create a function to read the raw and b&w images, and filenames

def read_set(gtdirectory,directory):
    #read subdirectory
    datas = [] # list of the data
    gts = [] # list of the ground truth data
    filenames = [] # list of the filenames
    
    for im_data in os.listdir(gtdirectory):
        #create path to subdirectory
        if "IMG" in im_data:
            gtPath = os.path.join(gtdirectory,im_data) #join the strings to store gt file paths
            dataPath = os.path.join(directory,im_data) #join the strings to store file paths
            if (os.path.exists(dataPath)): #function that checks if the specified path exists

                datas.append(io.imread(dataPath)/255.0) #including the images in the data list
                gts.append(cv2.imread(gtPath)/255.0) #including the gt images in gt list
                filenames.append(im_data) #including filenames in the list
    return datas,gts,filenames #return the variables in list

# read the images and store them as array
datas,gts,filename = read_set(gt_folder, raw_folder) 

# using the defined function before, 
# read the original and ground truth images, and the filenames to the lists

# check the shape of read images
print("Ground truth shape = ", gts[0].shape)
print("Datas shape = ", datas[0].shape)
# gray scale images have to have 1 color in 3rd channel. So, let us fix it.

# ### 6. Split the Raw and Ground truth images

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

# This is used to split several images at once.

def imageSplitter(datas,gts,size): # 3 inputs: normal image, ground truth image, size 256x256
    splittedDatas = [] # Empty list to include splitted datas of original image
    splittedGts = [] # Empty list to include splitted datas of ground truth image
    for i in range(0,datas.__len__()): #for function to read images in datas one by one
        data = datas[i] #data is the single original image 
        gt =gts[i] #gt is the single ground truth image
        tempData = split(data,size) # split the single image to the tempData
        tempGt = split(gt,size) # split the single GT image to the tempGt
        splittedDatas.extend(tempData) #Add the splitted image to the list
        splittedGts.extend(tempGt) #Add the splitted gt image to the list
    return splittedDatas,splittedGts #return the lists with splitted images.

# apply splitting function

size = (1728,2592) #Size of chunk of images that we want
# In this case, we want to divide an image to 4 pieces. 
# Thus, the size of width and height are the half of the original

splittedDatas,splittedGts = imageSplitter(datas,gts,size) 
# using the function that was defined before, 
# we split the images into the chunk of images that we use to train the model

# print some images to visualize

for x in range(60,64): # for each 10 images 
    plt.figure(figsize=(16,8)) # create a space for the images 16(width) 8(height) 
    plt.suptitle("Comparison", fontsize=14, fontweight='bold') #put a superior title

    plt.subplot(1,2,1) #figure has 1 row, 2 columns, and this is in 1st plot
    plt.imshow(splittedDatas[x]) #plotting the image in the first plot
    plt.title("Original") #title for the 1st image
    
    plt.subplot(1,2,2) #create a figure that has 1 row, 2 columns. And this is in 2nd plot
    plt.imshow(splittedGts[x]) #Showing gt images beside the original one
    plt.title("Ground Truth") #Adding subtitle into the gt images

# Count the number of total splitted images
print(f"splittedGts is a list of = {splittedGts.__len__()} ground truth images") # still in list format
print(f"splittedDatas is a list of = {splittedDatas.__len__()} rgb images") # still in list format

# Using np.asarray(), convert the input to a numpy array
splittedGts = np.asarray(splittedGts)
splittedDatas = np.asarray(splittedDatas)
print(f"splittedGts now is an 4 dimension array with {splittedGts.shape} shape")
print(f"splittedDatas now is an 4 dimension array with {splittedDatas.shape} shape")

# ### 7. Save the images as png image type in folder

# creating a list with filenames
new_filename = []

for i in filename:
    for j in range(1,5):
        new = str(i[:8])+"_"+str(j)+'.png' # adding numerations for splitted images
        new_filename.append(new)

# check if the filenames are in the list
print(new_filename)

# Save the Raw splitted images into the folder "06_raw_cut".
for i in range(0, len(splittedDatas)):
    plt.imsave("../02_data_retrain/01_data preparation/06_raw_cut/" +str(new_filename[i]), splittedDatas[i])

# Save the Ground truth splitted images into the folder "07_gt_cut".
for i in range(0, len(splittedGts)):
    plt.imsave("../02_data_retrain/01_data preparation/07_gt_cut/" +str(new_filename[i]), splittedGts[i], cmap='gray')


