#! /usr/bin/env python2

import roslib
roslib.load_manifest('traffic_gesture_recognition')

import rospy

from collections import Counter

from autoware_msgs.msg import traffic_light

from traffic_gesture_recognition.msg import police_gesture

class GestureFilter:
    def __init__(self):
        self.gesture_sub = rospy.Subscriber("/police_gesture", police_gesture, self.gesture_cb)
        self.traffic_light_pub = rospy.Publisher("/light_color", traffic_light, queue_size=10)
        self.gesture_queue_size = 15
        self.gesture_q = []

    def gesture_cb(self, gesture):
        if len(self.gesture_q) == self.gesture_queue_size:
            self.gesture_q = self.gesture_q[1:]

        self.gesture_q.append(gesture.gesture)

        tl = traffic_light()
        tl.header = gesture.header
        tl.traffic_light = 1

        most_common, num_most_common = Counter(self.gesture_q).most_common(1)[0]
        print(most_common, num_most_common)
        if most_common == 1:
            tl.traffic_light = 0
        # confidence = len(self.gesture_q)/num_most_common
        self.traffic_light_pub.publish(tl)

    def spin(self):
        rospy.spin()


if __name__ == "__main__":
    rospy.init_node('gesture_to_traffic_signal')
    filter = GestureFilter()
    filter.spin()
