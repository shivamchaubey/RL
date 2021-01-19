#!/usr/bin/env python

# Python libs
import sys, time

# numpy and scipy
import numpy as np
from scipy.ndimage import filters
import matplotlib.pyplot as plt

# OpenCV
import cv2

# Ros libraries
import roslib
import rospy

# from lane_segment import clustering
# Ros Messages
from sensor_msgs.msg import CompressedImage

VERBOSE=False

class image_feature:

    def __init__(self):
        '''Initialize ros publisher, ros subscriber'''
        # topic where we publish
        self.image_pub = rospy.Publisher("/output/image_raw/compressed",
            CompressedImage,  queue_size = 1)
        # self.bridge = CvBridge()

        # subscribed Topic
        self.subscriber = rospy.Subscriber("/camera/rgb/image_raw/compressed",
            CompressedImage, self.callback,  queue_size = 1)
        if VERBOSE :
            print "subscribed to /camera/image/compressed"


    def callback(self, ros_data):
        '''Callback function of subscribed topic. 
        Here images get converted and features detected'''
        if VERBOSE :
            print 'received image of type: "%s"' % ros_data.format

        #### direct conversion to CV2 ####
        np_arr = np.fromstring(ros_data.data, np.uint8)
        # image_np = cv2.imdecode(np_arr, cv2.CV_LOAD_IMAGE_COLOR)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # OpenCV >= 3.0:

        # seg_img = clustering(image_np)
        cv2.imshow('clustering image', image_np)
        cv2.waitKey(2)

        # plt.imshow(seg_img)
        # plt.show()
        #### Feature detectors using CV2 #### 
        # "","Grid","Pyramid" + 
        # "FAST","GFTT","HARRIS","MSER","ORB","SIFT","STAR","SURF"
        # method = "GridFAST"

        # feat_det = cv2.ORB_create() # OpenCV >= 3.0
        # # feat_det = cv2.FeatureDetector_create(method) #older opencv
        # time1 = time.time()

        # disp = rospy.get_param("/sim_observer/disp")
    
        # # convert np image to grayscale
        # featPoints = feat_det.detect(
        #     cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY))
        # time2 = time.time()
        # if VERBOSE :
        #     print '%s detector found: %s points in: %s sec.'%(method,
        #         len(featPoints),time2-time1)

        # for featpoint in featPoints:
        #     x,y = featpoint.pt
        #     cv2.circle(image_np,(int(x),int(y)), 3, (0,0,255), -1)
        # if disp == 1:
	       #  cv2.imshow('cv_img', image_np)
	       #  cv2.waitKey(2)

        # #### Create CompressedIamge ####
        # msg = CompressedImage()
        # msg.header.stamp = rospy.Time.now()
        # msg.format = "jpeg"
        # msg.data = np.array(cv2.imencode('.jpg', image_np)[1]).tostring()
        # # Publish new image
        # self.image_pub.publish(msg)
        
        #self.subscriber.unregister()

def main(args):
    '''Initializes and cleanup ros node'''
    
    rospy.init_node('image_feature', anonymous=True)
    loop_rate       = 30
    rate            = rospy.Rate(loop_rate)
    ic = image_feature()
    while not (rospy.is_shutdown()):
        rate.sleep()
    cv2.destroyAllWindows()

# if __name__ == '__main__':
#     main(sys.argv)

if __name__ == "__main__":

    try:
        main(sys.argv)

    except rospy.ROSInterruptException:
        pass
