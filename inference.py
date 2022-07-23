import numpy as np
import pickle as pkl
import time
import cv2
import sys

from utils import load_model, clean_boxes, scale_boxes, scale_boxes_r
from tracker import Tracker, Projector

################################### Paths
model_dir = './models/'
model_name = 'rYOLO1_1645960395'

video_dir = './videos/'
video_name = 'traffic-1.mp4'

classes = ['car', 'van', 'bus']
colors = [(0,255,0),(255,0,0),  (0,0,255)]

################################### Load model
print("Loading model...")
model_file = model_name + '.json'
weights_file = model_name + '.h5'
model, input_h, input_w = load_model(model_dir, model_name)
print("Done")
model.summary()

################################### Load video
cap = cv2.VideoCapture(video_dir + video_name)
ret, frame = cap.read()

frame_big_w = frame.shape[1]
frame_big_h = frame.shape[0]

################################### Inference parameters
iou_thr = 0.2
obj_thr = 0.5
class_thr = 0.5
display_h = 640

anchor = 128 # anchor size for bounding boxes
grid_w = 16 # pixel height of grid cells
grid_h = grid_w

yroi = [0.5, 0.9] # region of interest y coordinates

USE_TRACKER = True
USE_PROJECTOR = False
det_thr = 1 # tracker detection threshold - consecutive frames
dist_thr = 0.03 # needs to be bigger for generally faster objects

################################### Build tracker and projector
if USE_TRACKER:
	tracker = Tracker(input_h, input_w, dist_thr=dist_thr, lookback=10, yroi=yroi)
if USE_PROJECTOR:
	projector = Projector(1, 0.2, input_h)

################################### Run
while (cap.isOpened()):
	ret, frame = cap.read()

	# Cut and resize frame
	resize_w = int(frame_big_w * (input_h / frame_big_h))
	cut_w = int(frame_big_h * input_w / input_h)
	xstart = int((frame_big_w - cut_w)/2)
	frame = frame[:, xstart:xstart+cut_w, :]

	image = cv2.resize(frame, (input_w, input_h), interpolation=cv2.INTER_AREA)
	image = image.reshape((1,input_h, input_w,3))/256

	new_h, new_w = frame.shape[:2]

	display_w = int(new_w * display_h/new_h)
	frame = cv2.resize(frame, (display_w, display_h), interpolation=cv2.INTER_AREA)

	# Draw roi lines
	cv2.line(frame, (0,int(yroi[0]*display_h)), (display_w,int(yroi[0]*display_h)), (0,255,0), 1)
	cv2.line(frame, (0,int(yroi[1]*display_h)), (display_w,int(yroi[1]*display_h)), (0,255,0), 1)

	# Predict and NMS
	prediction = model.predict(image)[0]
	boxes = clean_boxes(prediction, iou_thr, obj_thr, class_thr, anchor, input_h, input_w, grid_h, grid_w)

	detected_objects = []

	if USE_TRACKER:
		objects = tracker.track(boxes, image)
		for obj in objects:
			if obj['life'] > det_thr:

				xr, yr = obj['xy']
				w, h = obj['wh']
				detected_objects.append([[xr, yr],[w, h]])

				class_idx = obj['class_idx']
				obj_id = obj['id']

				x1,y1,x2,y2 = scale_boxes_r(xr, yr, w, h, display_w, display_h)

				if yr > yroi[0] and yr < yroi[1]:
					cv2.rectangle(frame, (x1,y1), (x2,y2), colors[class_idx], 2)
					cv2.putText(frame, str(obj_id)+' '+classes[class_idx], (x1,y1-3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[class_idx], 1, cv2.LINE_AA)

		if USE_PROJECTOR:
			projection = projector.project(detected_objects)
			projection = cv2.resize(projection, (300,300), interpolation=cv2.INTER_AREA)
			cv2.imshow("projection", projection)

	else:
		for box in boxes:
			x1,y1,x2,y2 = box[0]	
			class_idx = np.argmax(box[1])
			x1,y1,x2,y2 = scale_boxes(x1,y1,x2,y2,display_w,display_h,input_w,input_h)
			cv2.rectangle(frame, (x1,y1), (x2,y2), colors[class_idx], 1)

	cv2.imshow("inference", frame)

	k = cv2.waitKey(20) & 0xFF
	if k == 27:
		break


cap.release()
cv2.destroyAllWindows()

