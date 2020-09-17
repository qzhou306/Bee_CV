import os, glob
import cv2
import numpy as np

# params bees
# proj_name='2_syn_cheng_mock'
proj_dir=r'../data/yolo_training/5_cheng_bees_1class_zhou_wang_manual_ds'
in_img_file_ext='jpg'
# show_img=True 
model_id='2000'
in_img_dir_or_file=r'../data/test/MP41080p20160313GOPR0850_4pm_Trim_40s-50s_Frames'
# in_img_dir_or_file=r'../data/test/sacs2_mocks.jpg'

model_name='yolov3_training_{}.weights'.format(model_id)

cfg_dir=os.path.join(proj_dir,'Configs')
cfg_file=os.path.join(cfg_dir,'yolov3_testing.cfg')
classes_file=os.path.join(cfg_dir,'classes.txt')
model_file=os.path.join(proj_dir, 'Models', model_name)
out_dir=os.path.join(proj_dir,"Out",model_name[:-8])

font = cv2.FONT_HERSHEY_PLAIN
# colors = np.random.uniform(0, 255, size=(100, 3))

net = cv2.dnn.readNet(model_file, cfg_file)

classes = []
with open(classes_file, "r") as f:
    classes = f.read().splitlines()

#init
out_target_img_dir=os.path.join(out_dir,os.path.basename(in_img_dir_or_file))
# print(out_target_img_dir)
os.makedirs(out_target_img_dir,0o777,exist_ok=True)


# params sacs
# proj_name2='4_cheng_sacs'
proj_dir2=r'../data/yolo_training/4_cheng_sacs'
# in_img_file_ext='png'
# show_img=True 
model_id2='2000'
# in_img_dir_or_file=r'C:\_me\Code\1jd_git\1_mine\iBee\yolo_detection\eval\4_cheng_sacs\Ds_test\P_Training_1002frame1PollenBee.png'
# in_img_dir_or_file=r'C:\_me\Code\1jd_git\1_mine\iBee\yolo_detection\eval\4_cheng_sacs\Ds_test'

model_name2='yolov3_training_{}.weights'.format(model_id2)

cfg_dir2=os.path.join(proj_dir2,'Configs')
cfg_file2=os.path.join(cfg_dir2,'yolov3_testing.cfg')
classes_file2=os.path.join(cfg_dir2,'classes.txt')
model_file2=os.path.join(proj_dir2, 'Models', model_name2)
out_dir=os.path.join(proj_dir2,"Out",model_name2[:-8])

font2 = cv2.FONT_HERSHEY_PLAIN

net2 = cv2.dnn.readNet(model_file2, cfg_file2)

classes2 = []
with open(classes_file2, "r") as f:
    classes2 = f.read().splitlines()
# out_target_img_dir2=os.path.join(out_dir2,os.path.basename(in_img_dir_or_file))
# # print(out_target_img_dir)
# os.makedirs(out_target_img_dir,0o777,exist_ok=True)


#logic
img_id=0

def predict_objects_in_img(net=None, img_fp=None, img=None, out_target_img_dir=None, 
        conf_threshold=0.2, nms_threshold=0.4, 
        bbox_color_bgr=(255,255,255), text_color_bgr=(255,255,255), font_scale=0.6,
        show_img=True, save_img=True ):

    if img is None:
        img=cv2.imread(img_fp)
        print('predict_objects_in_img():', img_fp)
    else:
        print('passed in img, shape:', img.shape)

    height, width, _ = img.shape

    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    if len(indexes)>0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            #use zero if x, y is negative
            x=max(x,0)
            y=max(y,0)
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i],2))
            
            #crop the bee
            print('crop x, y, w, h: ', x, y, w, h)
            bee_img=img[y:y+h, x:x+w]
            # cv2.imshow('bee_{}'.format(i), bee_img)
            ret_img, obj_count=predict_objects_in_img(net=net2, img=bee_img, out_target_img_dir=out_target_img_dir, bbox_color_bgr=(0,0,255), show_img=False, save_img=False)
            img[y:y+h, x:x+w]= ret_img
            print('obj_count', obj_count)

            # anno bee
            # cv2.putText(img, text, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]]) 
             
            if obj_count==0:
                line_thickness=1
                bbox_color_bgr_sacs=bbox_color_bgr
            else:
                line_thickness=2
                bbox_color_bgr_sacs=(255,0,255)

            cv2.rectangle(img, (x,y), (x+w, y+h), bbox_color_bgr_sacs, line_thickness)
            cv2.putText(img, confidence, (x, y+35), font, font_scale, text_color_bgr, 1)

            # cv2.waitKey(0)

    # cv2.putText(img, str(img_id), (0,30), font, 2, (255,255,255), 2)

    if show_img: 
        cv2.imshow('Image', img)
    if save_img: 
        cv2.imwrite(os.path.join(out_target_img_dir, '{}'.format(os.path.basename(img_fp)) ), img)

    return img,len(indexes)


# pre-process input
in_img_fps=[]
if os.path.isfile(in_img_dir_or_file):
    in_img_fps.append(in_img_dir_or_file)
else:
    in_img_fps=glob.glob('{}/*.{}'.format(in_img_dir_or_file, in_img_file_ext))    

# predict images
for in_img_fp in in_img_fps:
    predict_objects_in_img(net=net, img_fp=in_img_fp, out_target_img_dir=out_target_img_dir, show_img=show_img)

if show_img:
    while True:
        key = cv2.waitKey(1)
        if key==27:
            break

cv2.destroyAllWindows()
