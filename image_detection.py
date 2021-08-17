#!/usr/bin/env python
# coding: utf-8

# # Graduate Rotational Internship Program- The Sparks Foundation

# # Domain: Computer Vision & Internet ofThings

# # Name: SHIVANI

# # Title: Object Detection

# In[112]:


import cv2 #command to install: pip install opencv-python


# In[113]:


import matplotlib.pyplot as plt #command to install:pip install matplotlib 


# In[114]:


config_file='ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt' # download the files (MobileNet-SSD v3 2020_01_14)	weights and config from 'https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API'
frozen_model='frozen_inference_graph.pb'# place it in the same directory


# IMAGE DETECTION

# In[115]:


model = cv2.dnn_DetectionModel(frozen_model,config_file)
# or use cv2.dnn.r and press tab


# In[116]:


file_name= 'coco_names.txt' # text file with all the labels
with open(file_name,'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')


# In[117]:


print(classLabels) # print the contents


# In[118]:


print(len(classLabels)) 


# In[119]:


model.setInputSize(320,320) #size 320*320
model.setInputScale(1.0/127.5) 
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)


# In[120]:


img = cv2.imread('hi.jpeg') #make sure the image is in the same directoery


# In[121]:


plt.imshow(img) 


# In[122]:


plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# In[123]:


ClassIndex, confidence, bbox= model.detect(img,confThreshold=0.5) #threshold=50%


# In[124]:


print(ClassIndex) # prints the image content wrt to the file coco_names


# In[125]:


font_scale=3
for ClassInd, conf, boxes in zip(ClassIndex.flatten(),confidence.flatten(),bbox):
    cv2.rectangle(img, boxes,color=(255,0,0), thickness=2)
    cv2.putText(img, classLabels[ClassInd-1].upper(), (boxes[0]+10,boxes[1]+30), cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)


# In[126]:


plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


# In[ ]:




