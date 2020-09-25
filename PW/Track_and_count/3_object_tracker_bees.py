from absl import flags
import sys, os
FLAGS = flags.FLAGS
FLAGS(sys.argv)

import time
import numpy as np
import cv2
import matplotlib.pyplot as plt

import tensorflow as tf
from yolov3_tf2.models import YoloV3
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import convert_boxes

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

# params
in_file=r'C:\_me\Data\CV_Bee_Data\3_Cheng_Pollen_Sacs_data\Test_videos\MP41080p20160423GOPR0983_10am_Prot.mp4' 
# in_file=r'C:\_me\Data\CV_Bee_Data\1_KoryCam\2020-08-18\Trim\GOPR2253_top_green_6s.mp4' 
class_names_file=r'../data/yolo_training/5_cheng_bees_1class_zhou_wang_manual_ds/Configs/classes.txt'
tf_weights_file=r'../data/yolo_training/5_cheng_bees_1class_zhou_wang_manual_ds/Models_TF_converted/yolov3.tf'
out_video='../data/track_n_count/out/out_tnc_{}.avi'.format(os.path.basename(in_file))

# logic
class_names = [c.strip() for c in open(class_names_file).readlines()]
# class_names = [c.strip() for c in open('./data/labels/coco.names').readlines()]
yolo = YoloV3(classes=len(class_names))
yolo.load_weights(tf_weights_file) #'./weights/yolov3.tf')

# identify objects
max_cosine_distance = 0.5# 0.2 #0.5
nn_budget = None#30 #None #100
nms_max_overlap = 0.8 #0.8

model_filename = 'model_data/mars-small128.pb'
encoder = gdet.create_box_encoder(model_filename, batch_size=1)
metric = nn_matching.NearestNeighborDistanceMetric('cosine', max_cosine_distance, nn_budget)
tracker = Tracker(metric)

vid = cv2.VideoCapture(in_file) #'./data/video/test/car_ibm_example.mp4')
# vid = cv2.VideoCapture('./data/video/car_2.mp4')

codec = cv2.VideoWriter_fourcc(*'XVID')
vid_fps =int(vid.get(cv2.CAP_PROP_FPS))
vid_width,vid_height = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(out_video, codec, vid_fps, (vid_width, vid_height))
# out = cv2.VideoWriter('./data/video/out_bees_1.avi', codec, vid_fps, (vid_width, vid_height))

from _collections import deque
pts = [deque(maxlen=30) for _ in range(1000)]

counter = []

frame_id=0
while True:
    _, img = vid.read()
    frame_id+=1
    print(frame_id)
    if img is None:
        print('Completed')
        break

    img_in = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_in = tf.expand_dims(img_in, 0)

    # Dimensions of inputs should match: shape[0] = [1,66,120,256] vs. shape[1] = [1,67,120,512]
    img_in = transform_images(img_in, 416)

    t1 = time.time()

    boxes, scores, classes, nums = yolo.predict(img_in, steps=1)

    classes = classes[0]
    names = []
    for i in range(len(classes)):
        names.append(class_names[int(classes[i])])
    names = np.array(names)
    # print('names:',names)

    converted_boxes = convert_boxes(img, boxes[0])
    features = encoder(img, converted_boxes)

    detections = [Detection(bbox, score, class_name, feature) for bbox, score, class_name, feature in
                  zip(converted_boxes, scores[0], names, features)]

    boxs = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    classes = np.array([d.class_name for d in detections])
    indices = preprocessing.non_max_suppression(boxs, classes, nms_max_overlap, scores)
    detections = [detections[i] for i in indices]

    tracker.predict()
    tracker.update(detections)

    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i)[:3] for i in np.linspace(0,1,20)]

    current_count = int(0)

    for track in tracker.tracks:
        if not track.is_confirmed() or track.time_since_update >1:
            continue
        bbox = track.to_tlbr()
        class_name= track.get_class()
        color = colors[int(track.track_id) % len(colors)]
        color = [i * 255 for i in color]

        cv2.rectangle(img, (int(bbox[0]),int(bbox[1])), (int(bbox[2]),int(bbox[3])), color, 2)
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1]-30)), (int(bbox[0])+(len(class_name)
                    +len(str(track.track_id)))*17, int(bbox[1])), color, -1)
        cv2.putText(img, class_name+"-"+str(track.track_id), (int(bbox[0]), int(bbox[1]-10)), 0, 0.75,
                    (255, 255, 255), 2)

        center = (int(((bbox[0]) + (bbox[2]))/2), int(((bbox[1])+(bbox[3]))/2))
        pts[track.track_id].append(center)

        for j in range(1, len(pts[track.track_id])):
            if pts[track.track_id][j-1] is None or pts[track.track_id][j] is None:
                continue
            thickness = int(np.sqrt(64/float(j+1))*2)
            cv2.line(img, (pts[track.track_id][j-1]), (pts[track.track_id][j]), color, thickness)

        height, width, _ = img.shape

        # #Identification Zone (white jd)
        # zone_y_max=height -20
        # zone_y_min=height -100

        #Identification Zone (cheng)
        zone_y_max=height -180
        zone_y_min=height -260

        cv2.line(img, (0, int(zone_y_max)), (width, int(zone_y_max)), (0,0, 255), thickness=2) #BGR
        cv2.line(img, (0, int(zone_y_min)), (width, int(zone_y_min)), (0, 0,255), thickness=2)

        # cv2.line(img, (0, int(3*height/6+height/20)), (width, int(3*height/6+height/20)), (0, 255, 0), thickness=2)
        # cv2.line(img, (0, int(3*height/6-height/20)), (width, int(3*height/6-height/20)), (0, 255, 0), thickness=2)

        center_y = int(((bbox[1])+(bbox[3]))/2)

        #Identification Zone
        if center_y <= int(zone_y_max) and center_y >= int(zone_y_min):
            if class_name == 'b' or class_name == 'p':
                counter.append(int(track.track_id))
                current_count += 1

    total_count = len(set(counter))
    # cv2.putText(img, "Current Count: " + str(current_count), (0, 80), 0, 1, (0, 0, 255), 2)
    cv2.putText(img, "Count: " + str(total_count), (0,80), 0, 1, (0,0,255), 2)

    fps = 1./(time.time()-t1)
    cv2.putText(img, "Frame# {} ; FPS: {:.2f}".format(frame_id, fps), (0,30), 0, 1, (0,0,255), 2)
    # cv2.resizeWindow('output', 1024, 768)
    cv2.imshow('output', img)
    out.write(img)

    if cv2.waitKey(1) == ord('q'):
        break
vid.release()
out.release()
cv2.destroyAllWindows()