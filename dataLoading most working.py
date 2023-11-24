


## 40 frame version 

import torch.nn.functional as F

import torch.nn as nn
import torch
import torch.optim 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


import os 
import cv2


# more imports 
import torchvision
import os
import matplotlib as plt
import numpy as np

#dataTransformResizeTrainData = transforms.Compose([transforms.Resize(224,224),transforms.ToTensor()])
## maybe I can also normalize the dataset 



# the 

##def creatEUsable dataset 


def getCategoryFileNames(categoryFolder):
    """
    Input: string folder containing the categories > i.e the root folder
    Output: list (all categories in this dataset)
    """
    categories = os.listdir(categoryFolder)
    return categories

def getVideoFileNames(folderWithVideos):
    """
    Input: folder containing the category of the videos
    Output: paths to the videos
    """
    videoLocations = []
    videoNames=os.listdir("%s/" % folderWithVideos)
    
    for location in videoNames:
        videoLocations.append(folderWithVideos + "/" +location)
    return videoLocations


# reference to the code on cv2 

def removeVideosWithMissingFrames(videoLocation,videoNumberInCategory, rootFolder, categoryName,categoryNumber):
## reference to the opencv website from where the code was adapted from #
    
    print("End of input")
    """
    Input: string of videoLocation + string videoFolderName + string frameFolderLocation 
    Output: folder with video frame of the video + number of frames in the video + 
    """
    #videoLocation = rootFolder + "/" + ""
    frameFolderLocation = rootFolder + "/" + categoryName + "/" ### 
    print("The frameFolderLocation is: " + frameFolderLocation)

    print(frameFolderLocation)
    captureObj = cv2.VideoCapture(videoLocation)


    videoFormatTemplateFolder = "{:04d}"
    videoFolderName = videoFormatTemplateFolder.format(videoNumberInCategory) 
    
    captureObjectIsOpened = captureObj.isOpened()
    frameCount = int(captureObj.get(cv2.CAP_PROP_FRAME_COUNT))

    imageNumberRemoval = 0 # for frames to remove 
    if (captureObjectIsOpened == True): # checks if teh object was opened 
        print("Video was opened") ## 
        
        videoFrameList = range(int(captureObj.get(cv2.CAP_PROP_FRAME_COUNT))) # gets teh number of frames of the video
        
        for videoFrame in videoFrameList:  # iterates through teh list of frames
        
            ret, frameRead = captureObj.read() 
            if ret == False: # checks if a frame was returned 
                print("Empty frame was encountered")
                captureObj.release()
                annotationsFile = open(('videosWithMissingFrames.txt'), "a" ) 
                annotationsFile.write(videoLocation) 
                annotationsFile.close()  
                os.remove(videoLocation)
                return "video was removed"
            else:
                continue
            
            imageNumberRemoval = imageNumberRemoval + 1 # keeps track of number of frames (images)

            if imageNumberRemoval == 31:
                captureObj.release()








def convertVideoToFrames(videoLocation,videoNumberInCategory, rootFolder, categoryName,categoryNumber):
## reference to the opencv website from where the code was adapted from #
   
    """
    Input: string of videoLocation + string videoFolderName + string frameFolderLocation 
    Output: folder with video frame of the video + number of frames in the video + 
    """
    #videoLocation = rootFolder + "/" + ""
    frameFolderLocation = rootFolder + "/" + categoryName + "/" ### 
    print("The frameFolderLocation is: " + frameFolderLocation)

    #print(frameFolderLocation)
    captureObj = cv2.VideoCapture(videoLocation) # opens the video



    

    ## section for creating teh frames 
    captureObj = cv2.VideoCapture(videoLocation) # opens the video

    videoFormatTemplateFolder = "{:04d}"
    videoFolderName = videoFormatTemplateFolder.format(videoNumberInCategory)
    os.makedirs(frameFolderLocation + "/" + videoFolderName) # makes a folder where the frames of the videos will be stored 
    
    captureObjectIsOpened = captureObj.isOpened()
    frameCount = int(captureObj.get(cv2.CAP_PROP_FRAME_COUNT))

    imageNumber = 0
    if (captureObjectIsOpened == True): # checks if teh object was opened 
        ## 
        
        videoFrameList = range(int(captureObj.get(cv2.CAP_PROP_FRAME_COUNT))) # gets teh number of frames of the video
        
        for videoFrame in videoFrameList:  # iterates through teh list of frames
        
            ret, frameRead = captureObj.read() 
            if frameRead is None: # checks if a frame was returned 
                annotationsFile = open(('videosWithMissingFrames.txt'), "a" ) 
                annotationsFile.write(videoLocation + "\n") 
                annotationsFile.close() 
                print("Empty frame was encountered")
                
                continue

            #videoFolderLocation = 

            imageFormatTemplate = "img_{:05d}.jpg"
            imageNameGenerator = imageFormatTemplate.format(imageNumber)

            #cv2.imwrite("%s%s/000%d.jpg" % (frameFolderLocation, videoFolderName, imageNumber), frameRead)##
            cv2.imwrite("%s%s/%s" % (frameFolderLocation, videoFolderName, imageNameGenerator), frameRead)## adds the frame to teh folder 

            #cv2.imwrite("%s%s/000%d.jpg" % )##

            imageNumber = imageNumber + 1 # keeps track of number of frames (images)

            if imageNumber == 41:
                captureObj.release()

    else:
        print("Error: The video wasn't opened")
    
    #os.remove(videoLocation)
    #print("Video was removed")
    
    annotationsFilePath = "annotations.txt" # path to where file will be created
    videoPath = (categoryName + "/" + videoFolderName)
    videoDetails = videoPath + " " + "1" + " " + str(imageNumber-1) + " " + str(categoryNumber) + "\n"
    annotationsFile = open((annotationsFilePath), "a" ) 
    annotationsFile.write(videoDetails) 
    annotationsFile.close()  

    return videoFolderName + "was created"



#convertVideoToFrames("datasets copy/Violence/v_Violence_n_s063_c001.avi", "v_Violence_n_s063_c001","datasets copy/Violence/")

#convertVideoToFrames(videoLocation,videoNumberInCategory, rootFolder, frameFolderLocation,categoryName):

#print(convertVideoToFrames("datasets copy/Violence/v_Violence_a_s003_c001.avi", 3,"datasets copy","Violence"))

#C:\Users\Aleks\Desktop\Final_Year_Project\datasets copy\Violence\v_Violence_a_s002_c001.avi



def removeEmptyFrames(datasetFolderLocation):
    
    """
    Input: example > "dataset copy/" the datasetFolderLocati
    Output: example > 
    """
    rootFolder = datasetFolderLocation
    categoryList =  os.listdir(datasetFolderLocation)
    categoryCounter = 0 
    for category in categoryList: # iterates through all categories in teh dataset
        videosInCategory = os.listdir(datasetFolderLocation + "/" + category)
        
        categoryNumber = categoryList.index(category) # gets the index of the category > needed for annotatiions.txt
        
        videosInCategoryCounter = 0
        for video in videosInCategory:
            videoLocation = datasetFolderLocation + "/" + category + "/" + video
            videoNumberInCategory = videosInCategoryCounter
            rootFolder = datasetFolderLocation
            categoryName = category
            

            removeVideosWithMissingFrames(videoLocation,videoNumberInCategory, rootFolder, categoryName,categoryNumber)
            #convertVideoToFrames(videoLocation,videoNumberInCategory, rootFolder, categoryName,categoryNumber)
            videosInCategoryCounter = videosInCategoryCounter + 1

            categoryCounter = categoryCounter + 1

def convertVideoDatasetToFramesDataset(datasetFolderLocation):
    
    """
    Input: example > "dataset copy/" the datasetFolderLocati
    Output: example > 
    """
    rootFolder = datasetFolderLocation
    categoryList =  os.listdir(datasetFolderLocation)
    categoryCounter = 0 
    for category in categoryList: # iterates through all categories in teh dataset
        videosInCategory = os.listdir(datasetFolderLocation + "/" + category)
        
        categoryNumber = categoryList.index(category) # gets the index of the category > needed for annotatiions.txt
        
        videosInCategoryCounter = 0
        for video in videosInCategory:
            videoLocation = datasetFolderLocation + "/" + category + "/" + video
            videoNumberInCategory = videosInCategoryCounter
            rootFolder = datasetFolderLocation
            categoryName = category

            #removeVideosWithMissingFrames(videoLocation,videoNumberInCategory, rootFolder, categoryName,categoryNumber)
            convertVideoToFrames(videoLocation,videoNumberInCategory, rootFolder, categoryName,categoryNumber)
            videosInCategoryCounter = videosInCategoryCounter + 1

            categoryCounter = categoryCounter + 1

#removeEmptyFrames("Real Life Violence Dataset")
convertVideoDatasetToFramesDataset("Real Life Violence Dataset")


#print(convertVideoToFrames("datasets copy/Violence/v_Violence_a_s003_c001.avi", 3,"datasets copy","Violence"))

#convertVideoDatasetToFramesDataset("datasets copy/")




"""



    # by location I mean path 

    

    categoryList = getCategoryFileNames(datasetFolderLocation)

    
    categoryCounter = 0 
    for category in categoryList:
        videoFileNames = getVideoFileNames(datasetFolderLocation + category)

    #for category in categoryList:
    #    videoList = os.listdir(category)
        
        #for video in category:



        #for video in 
        #Test 
        
        #Test ##
        categoryCounter = categoryCounter + 1

 
        #abbotationsDetails  = [[],[]]



"""





"""   



#works 
captureObj =cv2.VideoCapture(videoLocation)
    if not captureObj.isOpened():
        print("Video not accessable")
        exit()
    
    #os.makedirs("datasets copy/video4")#
    os.makedirs(frameFolderLocation + "/" + videoFolderName)
    imageNumber = 0
    while True:
        ret, frameRead = captureObj.read() ## reads one frame at a time 

        cv2.imwrite(("datasets copy/video4/myFirstimg%d.jpg" % imageNumber),frameRead) # creates a jpg file for that frame 
        imageNumber = imageNumber + 1


#works












#captureObj = cv2.VideoCapture("datasets/Fire/v_Fire_a_s001_c001.avi")

videosToConvert = getVideoFileNames("datasets copy/Fire")
print(videosToConvert)
#Input:




videosToConvert[4]

captureObj = cv2.VideoCapture(videosToConvert[4])
if not captureObj.isOpened():
    print("Video not accessable")
    exit()

##for video in 

os.makedirs("datasets copy/video4")
imageNumber = 0
while True:

    ret, frameRead = captureObj.read() ## reads one frame at a time 
    #print(frameRead)
    

    cv2.imwrite(("datasets copy/video4/myFirstimg%d.jpg" % imageNumber),frameRead) # creates a jpg file for that frame 
    imageNumber = imageNumber + 1

"""


