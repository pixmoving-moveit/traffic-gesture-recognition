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
import tensorflow as tf


class GestureRecog:
    def __init__(self, debug=False):
        package_name = "traffic_gesture_recognition"
        rospack = rospkg.RosPack()
        model_path_prefix = os.path.join(rospack.get_path(package_name), package_name, "models")

        #self.policeman_verifier = tf.keras.models.load_model(os.path.join(model_path_prefix, "policeman.h5"))
        self.gesture_dectector = tf.keras.models.load_model(os.path.join(model_path_prefix, "gesture.h5"))

        self.graph = tf.get_default_graph()

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
        self.incoming_bbox = None

        self.detected_gesture = police_gesture()
        self.network_size = (299, 299) #square

    def spin(self):
        rospy.spin()

    def process(self, image, bbox):
        if not bbox.type == "person":
            return
        try:
            self.image_to_process = self.bridge.imgmsg_to_cv2(image, "bgr8")
        except CvBridgeError as e:
            print(e)
            return

        arr = []
        for box in bbox.obj:
            new_box = self.correct_size(self.image_to_process, box)
            arr.append(new_box)
        if not len(arr):
            return
       
        batch_inp = np.stack(arr, axis=0)
        batch_result, batch_confidence = self.predict_policeman(batch_inp)

        print(batch_result, batch_confidence)

        #assume there's only one policeman
        indices = [i for i, x in enumerate(batch_result) if x]
        
        detected_gesture = police_gesture()
        detected_gesture.header = bbox.header

        inp = []
        for i in indices:
            inp.append(arr[indices[i]])
        if len(indices):
            batch_inp = np.stack(inp, axis=0)
            gesture = self.detect_gesture(batch_inp)
        else:
            gesture = ([0], [1])

        print("Gesture detected:", gesture)
       
        detected_gesture.gesture = np.any(gesture[0])
        detected_gesture.confidence = -1;  # gesture[1]

        self.detected_gesture = deepcopy(detected_gesture)

        self.publish_gesture(detected_gesture)
        if self.debug and len(indices):
            self.publish_gesture_overlay(self.image_to_process, detected_gesture, [bbox.obj[i] for i in indices])

    # img_rect is of type autoware_msgs::image_rect
    def correct_size(self, image, img_rect):
        x_right = img_rect.x + img_rect.width
        y_bottom = img_rect.y + img_rect.height
        img = image[img_rect.y:y_bottom, img_rect.x:x_right, :]

        zero_mean = (img - np.mean(img, axis=0))/128.

        scaled_img = cv.resize(zero_mean, self.network_size)
        return scaled_img

    def publish_gesture(self, gesture=None):
        if not gesture:
            gesture = self.detected_gesture
        # @TODO: check confidence
        self.gesture_pub.publish(gesture)

    def publish_gesture_overlay(self, image=None, gesture=None, batch_bbox=None):
        if image is None:
            image = self.image_to_process
        if gesture is None:
            gesture = self.detected_gesture
        if batch_bbox is None:
            return
        img_gesture_overlay = np.copy(image)

        # overlay here
        for bbox in batch_bbox:
            color = (0, 255, 0)
            if gesture.gesture == 1:
                color = (0, 0, 255)
            cv.rectangle(img_gesture_overlay, (bbox.x, bbox.y), (bbox.x+bbox.width, bbox.y+bbox.height), color, thickness=6)

        img = cv.resize(img_gesture_overlay, (320, 240))

        msg = self.bridge.cv2_to_imgmsg(img, "bgr8")
        msg.header = gesture.header
        self.gesture_overlay_pub.publish(msg)

    def predict_policeman(self, batch_inp):
        return ([1]*len(batch_inp), [1]*len(batch_inp))
        with self.graph.as_default():
            batch_result = self.policeman_verifier.predict(batch_inp)
        is_police = np.argmax(batch_result, axis=1)
        # @TODO: no need to compute twice, do lookup instead
        confidence = np.max(batch_result, axis=1)
        return (is_police, confidence)

    def detect_gesture(self, inp):
        with self.graph.as_default():
            result = self.gesture_dectector.predict(inp)
        gesture = np.argmax(result, axis=1)
        # @TODO: no need to compute twice, do lookup instead
        confidence = np.max(result, axis=1)
        return (gesture, confidence)


if __name__ == "__main__":
    rospy.init_node('gesture_detector')
    ges = GestureRecog(debug=True)
    ges.spin()
