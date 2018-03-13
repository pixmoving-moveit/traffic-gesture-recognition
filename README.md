# Gesture detection

## Dependencies:
* Tensorflow
* autoware_msgs

## Running:
* Run YOLO2
```
$ roslaunch cv_tracker yolo2.launch
```

* Main code:
```
$ rosrun traffic_gesture_recognition ros_gesture_detection.py
```

* Filter:
```
$ rosrun traffic_gesture_recognition gesture_filter.py
```

* Show debug images:
```
$ rosrun image_view image_view image:=/police_gesture/image_overlay
```

* Toggle debug images:
```
$ rosservice call /set_debug "data: true"
$ rosservice call /set_debug "data: false"
```

* Toggle policeman detection
```
$ rosservice call /verify_policeman "data: true"
$ rosservice call /verify_policeman "data: false"
```

* Run from video_file
```
roslaunch traffic_gesture_recognition video_file.launch
```

* Set correct topic for launch (change videofile/image_raw to topic of your choice)
```
rosrun topic_tools relay /videofile/image_raw /image_raw
```
