#!/usr/bin/env python
import time

import roslib; roslib.load_manifest('rtg_proje')
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from cv_bridge import CvBridge
import rospy  # Python library for ROS
from sensor_msgs.msg import Image  # Image is the message type
from cv_bridge import CvBridge  # Package to convert between ROS and OpenCV Images
import cv2  # OpenCV library
from skimage.metrics import structural_similarity
import time

images = []
imageNames = ["h1", "h2", "h3", "h4", "h5", "h6", "h7", "h8", "h9", "h10", "h11", "h12", "varil"]

for i in range(1, 13):
    images += [cv.imread('images/h' + str(i) + '.png', cv.IMREAD_GRAYSCALE)]

#images += [cv.imread('images/varilmask.png', cv.IMREAD_GRAYSCALE)]
varil_img =  cv.imread('images/varilmask.png', cv.IMREAD_GRAYSCALE)

br = CvBridge()
sift = cv.SIFT_create()
bf = cv.BFMatcher()

descriptors = []
for img in images:
    kp1, des1 = sift.detectAndCompute(img, None)
    descriptors.append([kp1, des1])

def callback(data):

    # Convert ROS Image message to OpenCV image
    current_frame = br.imgmsg_to_cv2(data)
    img_real = cv.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
    kp2, des2 = sift.detectAndCompute(img_real, None)

    #images = [cv.imread('images/h1.png', cv.IMREAD_GRAYSCALE)]
    for desc, img, i in zip(descriptors, images, imageNames):

        # Initiate SIFT detector

        # find the keypoints and descriptors with SIFT
        #kp1, des1 = sift.detectAndCompute(img, None)


        kp1, des1 = desc[0], desc[1]



        # BFMatcher with default params
        try:
            matches = bf.knnMatch(des1, des2, k=2)
            # Apply ratio test
            good = []
            for m, n in matches:
                if m.distance < 0.50 * n.distance:
                    good.append([m])
            # cv.drawMatchesKnn expects list of lists as matches.
        except:
            continue

        if len(good) > 10:
            #img3 = cv.drawMatchesKnn(img, kp1, img_real, kp2, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            print(i , " is found")

    # Convert BGR to HSV
    hsv = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([0, 50, 20], dtype=np.uint8)
    upper_blue = np.array([5, 255, 255], dtype=np.uint8)

    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Bitwise-AND mask and original image
    #res = cv2.bitwise_and(current_frame, current_frame, mask=mask)
    #cv2.imshow("camera1", hsv)
    #cv2.imshow("camera2", mask)
    #cv2.imshow("camera3", res)
    varilConf = meanMatrix(mask)
    #print(varilConf)
    if(varilConf > 40):
        print("Varil is found")
    print("*******************************************")

def meanMatrix(matrix):
    matrix = np.array(matrix)
    return matrix.mean()

def receive_message():
    # Tells rospy the name of the node.
    # Anonymous = True makes sure the node has a unique name. Random
    # numbers are added to the end of the name.

    rospy.init_node('video_sub_py', anonymous=True)


    # Node is subscribing to the video_frames topic
    rospy.Subscriber('/camera/rgb/image_raw', Image, callback)

    #rospy.Rate(1)
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


    # Close down the video stream when done
    #cv2.destroyAllWindows()


if __name__ == '__main__':
    receive_message()

