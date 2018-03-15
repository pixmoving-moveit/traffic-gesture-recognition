# Chinese Traffic Gesture Dataset(CTGD)

We collected and labeled a dataset called Chinese Traffic Gesture Dataset(CTGD) for training a deep learning traffic gesture recognition model. This dataset contains more than 160000 traffic gesture images and more than 20000 images of policeman and pedestrian. The two subset in CTGD:

* The traffic policeman dataset：
* The traffic gesture dataset：

## Traffic Policeman Dataset
Images of traffic policeman and pedestrian are collected and labeled. You can download with this link:
## Traffic Gesture Dataset
Four types of traffic gesture have been collected from volunteers with 6 cameras in different directions. The four gestures are:

* stop
* go_stright
* turn_right
* park_right

Each traffic gesture is a series of images with order. This dataset contains more than 170000 images of . The structure of this dataset is organized as:

```
---stop:
-------"cam_id"_"segment_id"_"frame_id"_"person_id".png
.......

---go_stright:
-------"cam_id"_"segment_id"_"frame_id"_"person_id".png
.......

--turn_right
-------"cam_id"_"segment_id"_"frame_id"_"person_id".png
.......

--park_right
-------"cam_id"_"segment_id"_"frame_id"_"person_id".png
.......
```

File naming rules：

* camera_id: 1, 2, 4, 6, 7, 8. denotes the camera in different directions
* segment_id: the segment id in the video
* frame_id: the frame number in a segment
* person_id: 0/1, it doesn't make sense
