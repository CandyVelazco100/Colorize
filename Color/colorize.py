import numpy as np

import cv2
import os

image_path = './images/girl.jpg'

#load the model
print("Loading the model")
""" net = cv2.dnn.readNetFromCaffe('C:/Users/admin/Desktop/usbIA2/IA2/Colorization/model/colorization_deploy_v2.prototxt',  'C:/Users/admin/Desktop/usbIA2/IA2/Colorization/model//colorization_release_v2.caffemodel')
pts = np.load('C:/Users/admin/Desktop/usbIA2/IA2/Colorization/model/pts_in_hull.npy') """
net = cv2.dnn.readNetFromCaffe('./model/colorization_deploy_v2.prototxt',  './model/colorization_release_v2.caffemodel')
pts = np.load('./model/pts_in_hull.npy')

#Load centers for ab chANNEL QUANTIZATION used for rebalancing
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
pts = pts.transpose().reshape(2,313,1,1)
net.getLayer(class8).blobs = [pts.astype("float32")]
net.getLayer(conv8).blobs = [np.full([1,313], 2.606,dtype = "float32")]

#Load the input image
image = cv2.imread(image_path)
scaled = image.astype("float32")/255.0
lab = cv2.cvtColor(scaled,cv2.COLOR_BGR2LAB)

resized = cv2.resize(lab,(224,224))
L = cv2.split(resized)[0]
L -= 50

print("Colorizing the image")
net.setInput(cv2.dnn.blobFromImage(L))
ab = net.forward()[0,:,:,:].transpose((1,2,0))

ab = cv2.resize(ab,(image.shape[1], image.shape[0]))

L = cv2.split(lab)[0]
colorized = np.concatenate((L[:, :, np.newaxis],ab), axis =2)

colorized = cv2.cvtColor(colorized,cv2.COLOR_LAB2BGR)
colorized = np.clip(colorized,0,1)

colorized = (255*colorized).astype("uint8")

cv2.imshow("Original",image)
cv2.imshow("Colorized",colorized)
cv2.waitKey(0)
