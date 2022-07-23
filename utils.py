import numpy as np
from keras.models import model_from_json, Model
import time
import cv2
import pickle as pkl

def load_model(model_dir, model_name):
	# load json and create model
	json_file = open(model_dir + model_name + '.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)

	# load weights into new model
	loaded_model.load_weights(model_dir + model_name + '.h5')
	input_shape = loaded_model.layers[0].input_shape[0]
	input_h, input_w = input_shape[1:3]

	return loaded_model, input_h, input_w

def save_model(model, model_name, model_dir, history):
	model_json = model.to_json()

	stamp = str(int(time.time()))
	with open(model_dir+model_name+"_"+stamp+".json", "w") as json_file:
	    json_file.write(model_json)
	# serialize weights to HDF5
	model.save_weights(model_dir+model_name+"_"+stamp+".h5")

	with open(model_dir+model_name+"_"+stamp+".pkl", 'wb') as f:
		pkl.dump(history.history, f)

	print("Saved Model and Weights with stamp:", stamp, ".")


def extractCoords(box, idx, h, w, dh, dw,anchor):
	dy = box[1] 
	dx = box[2] 
	hr = box[3]
	wr = box[4]
	x = idx[1]
	y = idx[0]

	xmid = (x + dx) * dw
	ymid = (y + dy) * dh		

	x1 = np.int(xmid - anchor*wr/2)
	x2 = np.int(xmid + anchor*wr/2)
	y1 = np.int(ymid - anchor*hr/2)
	y2 = np.int(ymid + anchor*hr/2)
	return x1,y1,x2,y2


def iouCompute(boxA, boxB):
	# compute intersection-over-union
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])

	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

	try:
		iou = interArea / float(boxAArea + boxBArea - interArea)
	except:
		iou = 0

	return iou


def clean_boxes(prediction, iou_thr, obj_thr, class_thr, anchor, input_h, input_w, dh, dw):
	# perform non-max-suppression (NMS)

	exists = np.where(prediction[:,:,0] > obj_thr)
	obj = prediction[exists[0], exists[1], 0]
	indexes = np.argsort(obj, axis = None)[::-1]
	valid = np.ones(len(indexes))
	boxes = []

	for i, p in enumerate(indexes):
		if valid[i] == 0:
			continue
		box1 = prediction[exists[0][p], exists[1][p]]
		coord1 = extractCoords(box1, [exists[0][p], exists[1][p]], input_h, input_w, dh, dw, anchor)
		boxes.append([coord1, box1[5:]])

		for j in range(i+1, len(indexes)):
			if valid[j] == 1:
				box2 = prediction[exists[0][indexes[j]], exists[1][indexes[j]]]
				coord2 = extractCoords(box1, [exists[0][indexes[j]], exists[1][indexes[j]]], input_h, input_w, dh, dw, anchor)
				iou = iouCompute(coord1, coord2)
				if iou > iou_thr:
					valid[j] = 0
	return boxes


def scale_boxes(x1,y1,x2,y2, w1, h1, w2, h2):
	x1 = int(x1 * w1/w2)
	y1 = int(y1 * h1/h2)
	x2 = int(x2 * w1/w2)
	y2 = int(y2 * h1/h2)	
	return x1,y1,x2,y2
	

def scale_boxes_r(xr,yr,w,h, display_w, display_h):
	x1 = int((xr-w/2)*display_w)
	x2 = int((xr+w/2)*display_w)
	y1 = int((yr-h/2)*display_h)
	y2 = int((yr+h/2)*display_h)
	return x1,y1,x2,y2


