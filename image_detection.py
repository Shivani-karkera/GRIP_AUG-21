import cv2 #command to install: pip install opencv-python

import matplotlib.pyplot as plt #command to install:pip install matplotlib 

config_file='ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt' # download the files (MobileNet-SSD v3 2020_01_14)	weights and config from 'https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API'
frozen_model='frozen_inference_graph.pb'# place it in the same directory

# IMAGE DETECTION
model = cv2.dnn_DetectionModel(frozen_model,config_file)
# or use cv2.dnn.r and press tab

file_name= 'coco_names.txt' # text file with all the labels
with open(file_name,'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')

print(classLabels) # print the contents
print(len(classLabels)) 

model.setInputSize(320,320) #size 320*320
model.setInputScale(1.0/127.5) 
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)

img = cv2.imread('sample.jpeg') #make sure the image is in the same directoery

plt.imshow(img) 

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

ClassIndex, confidence, bbox= model.detect(img,confThreshold=0.5) #threshold=50%

print(ClassIndex) # prints the image content wrt to the file coco_names

font_scale=3
for ClassInd, conf, boxes in zip(ClassIndex.flatten(),confidence.flatten(),bbox):
    cv2.rectangle(img, boxes,color=(255,0,0), thickness=2)
    cv2.putText(img, classLabels[ClassInd-1].upper(), (boxes[0]+10,boxes[1]+30), cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))



