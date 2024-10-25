import numpy as np
import cv2
from cv2 import dnn

#-------- Configuración de rutas --------#
PROTO_FILE = './model/colorization_deploy_v2.prototxt'
MODEL_FILE = './model/colorization_release_v2.caffemodel'
HULL_PTS = './model/pts_in_hull.npy'
IMG_PATH = './images/katy-jurado.jpg'
RESULT_PATH = './results/colorized_final.png'
IMG_SIZE = (640, 640)

net = cv2.dnn.readNetFromCaffe(PROTO_FILE, MODEL_FILE)
points = np.load(HULL_PTS)
points = points.transpose().reshape(2, 313, 1, 1)
net.getLayer(net.getLayerId('class8_ab')).blobs = [points.astype(np.float32)]
net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313, 1, 1], 2.606, np.float32)]

bwImage = cv2.imread(IMG_PATH)
normed = bwImage.astype(np.float32) / 255.0
lab = cv2.cvtColor(normed, cv2.COLOR_BGR2Lab)
resized = cv2.resize(lab, (224, 224))
l = cv2.split(resized)[0]
l -= 50

net.setInput(cv2.dnn.blobFromImage(l))
ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
ab = cv2.resize(ab, (bwImage.shape[1], bwImage.shape[0]))

l = cv2.split(lab)[0]
colorized = np.concatenate((l[:, :, np.newaxis], ab), axis=2)
colorized = cv2.cvtColor(colorized, cv2.COLOR_Lab2BGR)
colorized = (255 * np.clip(colorized, 0, 1)).astype(np.uint8)

cv2.imshow('Black and White Image', bwImage)
cv2.imshow('Colorized Image', colorized)
cv2.waitKey(0)
cv2.destroyAllWindows()