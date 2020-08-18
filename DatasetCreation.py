#This code resizes the images and creates a random 80/20 train test split

#Importing libraries
from PIL import Image
import os
import shutil
import random

#Taking the directory location of Images as Input
source=input("Enter the path of the source(Images): ")
source=source.replace('\\','//')+'/'

#Creating a directory for resizing the images and placing them
os.mkdir('ResizedImages')
dest='ResizedImages/'

#Taking as input the choice of image formats
choice=input("What kind of files are located in your dataset:jpg,jpeg,JPG or png: ")

#Entering the dimension choice of the images
size=input("Dimensions to change into: ")

#Splitting on x to get the dimensions
size=size.split('x')
#Converting the string elements into integer
res = [int(i) for i in size]

#Getting all the images in the directory
img = [f for f in os.listdir(source) if f.endswith('.'+choice)]

#Creating the resized Images
for i in range(0,len(img)):
    im = Image.open(source+img[i])
    im_resized = im.resize(res, Image.ANTIALIAS)
    im_resized.save(dest+"/"+str(i+40)+".png")
print("Finished Creating resized Images")

os.mkdir(dest+'Train')
os.mkdir(dest+'Test')

#Finding all the images in the directory
img = [f for f in os.listdir(dest) if f.endswith('.png')]

#Creating a random list of train and test with a 80/20 split
train=random.sample(img, int(len(img)*0.8))
test=set(img).difference(set(train))
test=list(test)

#Moving the Images
for i in range(len(train)):
    shutil.move(dest+train[i], source+"Train/")
for j in range(len(test)):
    shutil.move(dest+test[j], source+"Test/") 