import cv2
import numpy as np
import math
# img = cv2.imread('/home/geesara/Pictures/bp8OO.jpg', 0)
# img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]  # ensure binary
# ret, labels = cv2.connectedComponents(img)
#
# print("Number of labels" , len(labels))
#
# def imshow_components(labels):
#     # Map component labels to hue val
#     label_hue = np.uint8(179*labels/np.max(labels))
#     blank_ch = 255*np.ones_like(label_hue)
#     labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
#
#     # cvt to BGR for display
#     labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
#
#     # set bg label to black
#     labeled_img[label_hue==0] = 0
#
#     cv2.imshow('labeled.png', labeled_img)
#     cv2.waitKey()

# imshow_components(labels)



def sigmoid(x):
  return 1 / (1 + math.exp(-x))


d = sigmoid(2)

ddf = 45


f = 0.869
P = 0.645

R = (f*P)/(2*P-f)