# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 13:06:56 2021

@author: 
"""
import cv2
import numpy as np
import matplotlib.pylab as plt
from scipy.signal import convolve2d
from PIL import Image
#from time import perf_counter
import time
import glob

##    Directories to save process resulting images
images_path  = 'Filters_Output/'
image_path_median  = 'Filters_Output/Median/'
image_path_gaussian  = 'Filters_Output/Gaussian/'
##   Image inserting
image = plt.imread("1-a.jpg")   


##                                     ##
##   Convolution of images             ##
##                                     ## 
def convolve_all_colours(im, window):
    ims = []
    for d in range(3):
        im_conv_d = convolve2d(im[:,:,d], window)
        ims.append(im_conv_d)
    im_conv = np.stack(ims, axis=2).astype("uint8")
    return im_conv
##                                   ##
##   Gaussina Kernel Definition      ##
##                                   ##
def Gaussian_Kernel(matrix_size, sigma):
    n = int((matrix_size)/2)
    numerator = np.asarray([[x**2 + y**2 for x in range(-n,n+1)] for y in range(-n,n+1)])
    returned=(np.exp(-numerator/(2*sigma**2)))/((2 * np.pi)* sigma**2)
    return returned
##                              ##
##   Downsampling function      ##
##                              ##
def Downsampling(img,scale):  
   width = int(img.shape[1] * 1/scale)  
   height = int(img.shape[0]* 1/scale)  
   dim = (width, height)  
   # resizing the image  
   resized = cv2.resize(img, dim)  
   return resized
##                              ##
##   Upsampling function        ##
##                              ##    
def Upsampling(img,scale):
   width = int(img.shape[1]*scale)  
   height = int(img.shape[0]*scale)  
   dim = (width, height)   
   ## Cubic Interpolation was selected from interpolating methods 
   resized = cv2.resize(img, dim, interpolation=cv2.INTER_CUBIC)  
   return resized
kernel_size = [3,5,7]   #kernel to be used
sampling_scale=[2,4,8]
size=len(kernel_size)
t1_start = time.time()
for i in range (0,size):
    gaussian_Kernel = Gaussian_Kernel(kernel_size[i],i+0.5)
    gaussian_Kernel=gaussian_Kernel/np.sum(gaussian_Kernel)
    image_gaussian_blur=convolve_all_colours(image, gaussian_Kernel)
    down_sampled = Downsampling(image_gaussian_blur,sampling_scale[i])#cv2.resize(imagegaussianblur, (0, 0), fx = w/2, fy =w/2)
    upsampled=Upsampling(down_sampled,sampling_scale[i])
    cv2.imwrite(image_path_gaussian + 'Blur_Gaussian_{}.jpg'.format(kernel_size[i]), image_gaussian_blur)
    cv2.imwrite(image_path_gaussian + 'Down_sampled_by_{}.jpg'.format(sampling_scale[i]), down_sampled)
    cv2.imwrite(image_path_gaussian + 'Up_sampled_from_{}.jpg'.format(sampling_scale[i]), upsampled)
t1_stop =  time.time()
#
#  Median Filter operation
t2_start = time.time()
for i in range (0,size):
    median_Blured = cv2.medianBlur(image,kernel_size[i]) 
    down_sampled = Downsampling(median_Blured,sampling_scale[i])
    upsampled=Upsampling(down_sampled,sampling_scale[i])
    cv2.imwrite(image_path_median + 'Median_blur_Gaussian_{}.jpg'.format(sampling_scale[i]), median_Blured)
    cv2.imwrite(image_path_median + 'Median_down_sampled_by_{}.jpg'.format(sampling_scale[i]), down_sampled)
    cv2.imwrite(image_path_median + 'Up_sampled_from_{}.jpg'.format(sampling_scale[i]), upsampled)
t2_stop = time.time() 
##                                    ##
##   Animation For Gaussian Fi ter    ##
##                                    ## 
frames = []
imgs = glob.glob("Filters_Output\Gaussian\*.jpg")
for i in imgs:
    new_frame = Image.open(i)
    frames.append(new_frame)
# Save into a GIF file that loops forever
frames[0].save('Animation\Animation_For_Gaussian_Filter.gif', format='GIF',append_images=frames[1:],save_all=True,duration=2000, loop=0)
##                                  ##
##   Animation For Median Filter    ##
##                                  ##                
frames = []
imgs = glob.glob("Filters_Output\Median\*.jpg")
for i in imgs:
    new_frame = Image.open(i)
    frames.append(new_frame)
frames[0].save('Animation\Animation_For_Median_Filter.gif', format='GIF',append_images=frames[1:],save_all=True,duration=2000, loop=0)
print("\nCode compiled and execute successfully !!\n\nPlease check the project folder for resulting outputs.")
print("\nElapsed Time During Gaussian Operation:",t1_stop-t1_start)
print("Elapsed Time During Midean Operation:",t2_stop-t2_start) 