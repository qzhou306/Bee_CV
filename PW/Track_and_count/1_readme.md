# Single-Multiple-Custom-Object-Detection-and-Tracking
git@github.com:emasterclassacademy/Single-Multiple-Custom-Object-Detection-and-Tracking.git

Please visit https://www.youtube.com/watch?v=zi-62z-3c4U for the full course - Real-time Multiple Object Tracking (MOT) with Yolov3, Tensorflow and Deep SORT

# pw3
. activate py373-cv341-ds 

## pw3 installation
conda install -c anaconda tensorflow-gpu
conda install -c anaconda seaborn

## check GPU
python check_GPU.py

## convert model of YOLO to Tensorflow. Expect to see some yolov3.tf.* created in the weights/ directory
- weights/yolo{x}.wights -> yolo{x}.tf.{index, data*}
python 2_convert_bee.py

## How to detect, recognize and track multiple objects
python 3_object_tracker_bees.py
    - class_names: './data/labels/coco.names'
    - weights: './weights/yolov3.tf'
    - model_filename: 'model_data/mars-small128.pb'
    - input: './data/video/test/car_ibm_example.mp4'
    - output: './data/video/results.avi'

## data download
Yolov3 weights
     https://www.youtube.com/redirect?q=https%3A%2F%2Fpjreddie.com%2Fmedia%2Ffiles%2Fyolov3.weights&v=zi-62z-3c4U&redir_token=QUFFLUhqbll0dkxmVXM0T2xHcEF1akdXVms5R2tPVTExUXxBQ3Jtc0trNmVVaFpkVjduRHVhNXhQTlR6M2ljdElfS1ZGZUJ6U2czeXBnR05tT1VOV3NXNW5kbU5JQ0xISVNuU09xQUFmWFZTajJuSlRpUzFvajVqZmFyamktdmt1WnN1WTNpelJlbC1weFlCVGF0WGo1YUlUVQ%3D%3D&event=video_description

Yolov3-tiny weights
     https://www.youtube.com/redirect?q=https%3A%2F%2Fpjreddie.com%2Fmedia%2Ffiles%2Fyolov3-tiny.weights&v=zi-62z-3c4U&redir_token=QUFFLUhqblhMWTJEMFZHb193NUpfV3JBRG9jTUMyTFFzd3xBQ3Jtc0tsQVE0Zk9rNExaWTJ2MGJ1U01keEJ6OVFfeHVYS2RjQkI4YzFkMWxjTzB6YTdzTGpZY2tkUDRZNXBmOFVWcVVyVkRwMVd1SWRaN2gtVU1PTmZXcFBLNHZkcDJueUs5SlJZWVhFNjljRTg5TXZsUElsOA%3D%3D&event=video_description

car video: 
    youtube-dl https://www.youtube.com/watch?v=wqctLW0Hb_0&feature=youtu.be
    wget https://learnml.s3.eu-north-1.amazonaws.com/road.mp4
