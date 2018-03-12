#! /usr/bin/env python2

import roslib
roslib.load_manifest('traffic_gesture_recognition')

import rospy

from copy import deepcopy
import numpy as np
import cv2 as cv
from Queue import Queue


from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

from autoware_msgs.msg import traffic_light
from autoware_msgs.msg import image_obj

from traffic_gesture_recognition.msg import police_gesture


class GestureRecog:

    def __init__(self, debug=False):
        self.policeman_verifier = create_model(weights="weights.hdf5")
        self.debug = debug

        self.gesture_pub = rospy.Publisher("/police_gesture", police_gesture, queue_size=10)
        self.traffic_light_pub = rospy.Publisher("/light_color", traffic_light, queue_size=10)
        self.gesture_overlay_pub = rospy.Publisher("/police_gesture/image_overlay", Image, queue_size=10)
        
        image_topic_name = "/image_raw"
        param_name = "~image_raw_node"
        if rospy.has_param(param_name):
            image_topic_name = rospy.get_param(param_name)
        self.image_sub = rospy.Subscriber(image_topic_name, Image, self.raw_img_cb)        

        self.pedestrian_sub = rospy.Subscriber("/obj_person/image_obj", image_obj, self.bb_cb)

        self.bridge = CvBridge()
        self.image_queue_size = 15
        self.incoming_img_q = Queue(self.image_queue_size)
        self.image_to_process = None
        self.incoming_bb = None

        self.detected_gesture = police_gesture()
        self.network_size = (299, 299) #square

    def spin(self):
        rospy.spin()

    def raw_img_cb(self, data):
        if self.incoming_img_q.full():
            self.incoming_img_q.get_nowait()
        try:
            self.incoming_img_q.put_nowait(deepcopy(data))
        except CvBridgeError as e:
            print(e)
            return

    def bb_cb(self, data):
        if not data.type == "person":
            return
        self.incoming_bb = deepcopy(data)
        if self.incoming_img_q.empty():
            # This should not occur, ERROR
            return
        target_image = None

        while True:
            target_image = self.incoming_img_q.get_nowait()
            if target_image.header == self.incoming_bb.header:
                break
        self.image_to_process = self.bridge.imgmsg_to_cv2(target_image, "bgr8")
        self.process(np.copy(self.image_to_process), deepcopy(self.incoming_bb))

    def publish_gesture(self, gesture=None):
        if not gesture:
            gesture = self.detected_gesture
        self.gesture_pub.publish(gesture)

        tl = traffic_light()
        tl.header = gesture.header
        # @TODO
        tl.color = 1
        self.traffic_light_pub.publish(tl)

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
        if gesture.gesture == 1
            color = (0, 0, 255)
        cv.rect(img_gesture_overlay, (bb.x, bb.y), (bb.x+bb.width, bb.y+bb.height), color, thickness=3)

        img = cv.resize(img_gesture_overlay, (320, 240))

        msg = self.bridge.cv2_to_imgmsg(img, "bgr8")
        msg.header = gesture.header
        self.gesture_overlay_pub.publish(msg)

    def process(self, image=None, bb=None):
        if not image:
            image = self.image_to_process
        if not bb:
            bb = self.incoming_bb
        arr = []
        for box in bb.obj:
            new_box = self.correct_size(image, box)
            arr.append(new_box)
        batch_inp = np.stack(arr, axis=0)
        batch_result = pedict_policeman(batch_inp)

        batch_result[i] 

        #assume there's only one policeman
        indices = [i for i, x in enumerate(batch_result) if x]
        
        detected_gesture = police_gesture()
        detected_gesture.header = bb.header
        detected_gesture.gesture = detect_gesture(arr[indices[0]]) if len(indices) else 0

        self.detected_gesture = detected_gesture
        publish_gesture(detected_gesture)
        if self.debug and len(indices):
            self.publish_gesture_overlay(image, detected_gesture, bb[indices[0]])
        return detected_gesture

    # img_rect is of type autoware_msgs::image_rect
    def correct_size(self, image, img_rect):
        x_right = img_rect.x + img_rect.width
        y_bottom = img_rect.y + img_rect.height
        img = image[img_rect.x:x_right, img_rect.y:y_bottom]

        scaled_img = cv.resize(img, network_size)
        return scaled_img

if __name__ == "__main__":
    ges = GestureRecog(debug=False)
    ges.spin()