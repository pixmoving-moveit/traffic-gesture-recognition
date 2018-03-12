#! /usr/bin/env python2

import roslib
roslib.load_manifest('traffic_gesture_recognition')

import rospy
import rospkg
import message_filters

from copy import deepcopy
import numpy as np
import cv2 as cv
import os

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

from autoware_msgs.msg import image_obj

from traffic_gesture_recognition.msg import police_gesture
from inception_v4 import create_inception_v4


class GestureRecog:
    def __init__(self, debug=False):
        package_name = "traffic_gesture_recognition"
        model_path_prefix = os.path.join(rospack.get_path(package_name), package_name, "models")

        self.policeman_verifier = create_inception_v4(nb_classes=2)
        self.gesture_dectector = create_inception_v4(nb_classes=2)        

        self.policeman_verifier.load_weights(os.path.join(model_path_prefix, "police_inceptionv4.45-0.28.hdf5"))
        self.policeman_verifier.load_weights(os.path.join(model_path_prefix, "gesture_inceptionv4.45-0.28.hdf5"))

        self.debug = debug

        self.gesture_pub = rospy.Publisher("/police_gesture", police_gesture, queue_size=10)
        self.gesture_overlay_pub = rospy.Publisher("/police_gesture/image_overlay", Image, queue_size=10)
        
        image_topic_name = "/image_raw"
        param_name = "~image_raw_node"
        if rospy.has_param(param_name):
            image_topic_name = rospy.get_param(param_name)

        self.image_sub = message_filters.Subscriber(image_topic_name, Image)
        self.pedestrian_sub = message_filters.Subscriber("/obj_person/image_obj", image_obj)

        self.queue_size = 15
        time_sync = message_filters.TimeSynchronizer([self.image_sub, self.pedestrian_sub], self.queue_size)
        time_sync.registerCallback(self.process)

        self.bridge = CvBridge()
        self.image_to_process = None
        self.incoming_bb = None

        self.detected_gesture = police_gesture()
        self.network_size = (299, 299) #square

    def spin(self):
        rospy.spin()

    def process(self, image, bb):
        if not data.type == "person":
            return
        try:
            self.image_to_process = self.bridge.imgmsg_to_cv2(image, "bgr8")
        except CvBridgeError as e:
            print(e)
            return

        arr = []
        for box in bb.obj:
            new_box = self.correct_size(image, box)
            arr.append(new_box)
        batch_inp = np.stack(arr, axis=0)
        batch_result, batch_confidence = predict_policeman(batch_inp)

        #assume there's only one policeman
        indices = [i for i, x in enumerate(batch_result) if x]
        
        detected_gesture = police_gesture()
        detected_gesture.header = bb.header
        gesture = detect_gesture(arr[indices[0]]) if len(indices) else 0

        self.detected_gesture.gesture = gesture[0]
        self.detected_gesture.confidence = gesture[1]

        publish_gesture(detected_gesture)
        if self.debug and len(indices):
            self.publish_gesture_overlay(image, detected_gesture, bb[indices[0]])

        return detected_gesture

    # img_rect is of type autoware_msgs::image_rect
    def correct_size(self, image, img_rect):
        x_right = img_rect.x + img_rect.width
        y_bottom = img_rect.y + img_rect.height
        img = image[img_rect.x:x_right, img_rect.y:y_bottom]

        zero_mean = (img - np.mean(img, axis=0))/128.
        scaled_img = cv.resize(zero_mean, network_size)
        return scaled_img

    def publish_gesture(self, gesture=None):
        if not gesture:
            gesture = self.detected_gesture
        # @TODO: check confidence
        self.gesture_pub.publish(gesture)

    def publish_gesture_overlay(self, image=None, gesture=None, bb=None):
        if not image:
            image = self.image_to_process
        if not gesture:
            gesture = self.detected_gesture
        if not bb:
            return
        img_gesture_overlay = np.copy(image)

        # overlay here
        color = (0, 255, 0)
        if gesture.gesture == 1:
            color = (0, 0, 255)
        cv.rect(img_gesture_overlay, (bb.x, bb.y), (bb.x+bb.width, bb.y+bb.height), color, thickness=3)

        img = cv.resize(img_gesture_overlay, (320, 240))

        msg = self.bridge.cv2_to_imgmsg(img, "bgr8")
        msg.header = gesture.header
        self.gesture_overlay_pub.publish(msg)

    def predict_policeman(self, batch_inp):
        batch_result = self.policeman_verifier.predict(batch_inp)
        is_police = np.argmax(batch_result, axis=1)
        # @TODO: no need to compute twice, do lookup instead
        confidence = np.max(batch_result, axis=1)
        return (is_police, confidence)

    def detect_gesture(self, inp):
        result = self.gesture_dectector.predict(inp)
        gesture = np.argmax(result, axis=1)
        # @TODO: no need to compute twice, do lookup instead
        confidence = np.max(result, axis=1)
        return (gesture, confidence)


if __name__ == "__main__":
    ges = GestureRecog(debug=False)
    ges.spin()