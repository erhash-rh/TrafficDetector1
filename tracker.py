import numpy as np
import copy
from utils import iouCompute

class Tracker(object):
	def __init__(self, input_h, input_w, det_max = 10, dist_thr = 0.05, lookback = 3, yroi=[0.5,0.9]):
		super().__init__()
		self.objects = []
		self.det_max = det_max
		self.add_det = 1
		self.sub_det = 1
		self.max_id = 0
		self.input_h = input_h
		self.input_w = input_w

		self.yroi = yroi

		self.dist_thr = dist_thr

		self.lookback = lookback

	def add_new_object(self, xr, yr, w, h, class_idx):
		new_object = {'xy': [xr, yr],
					  'wh': [w, h],
					  'id': self.max_id,
					  'class_idx': class_idx,
					  'life': 5,
					  'dxr_arr': [],
					  'dyr_arr': [],
					  'w_arr': [w],
					  'h_arr': [h]}
		self.objects.append(copy.deepcopy(new_object))
		self.max_id += 1

	def track(self, boxes, image):
		new_objects = []
		for obj in self.objects:
			obj['life'] -= self.sub_det
			if obj['life'] > 0:
				new_objects.append(obj)

		self.objects = new_objects

		for box in boxes:
			x1,y1,x2,y2 = box[0]
			class_idx = np.argmax(box[1])

			# test for location in roi
			yr = (y1+y2)/2/self.input_h
			if yr < self.yroi[0] or yr > self.yroi[1]:
				continue

			# make feature vector
			xr = (x1+x2)/2/self.input_w
			w = (x2-x1)/self.input_w
			h = (y2-y1)/self.input_h

			feat_vector = np.asarray([xr, yr])
			
			# compute distance to tracked objects
			distances = [None for _ in range(len(self.objects))]
			for i, obj in enumerate(self.objects):
				distances[i]=np.linalg.norm(feat_vector-np.asarray(obj['xy']))


			if len(self.objects) != 0:
				# get closest object 
				position = np.argmin(distances)

				# if the object is closer than the minimum distance
				if distances[position] < self.dist_thr:

					# get its relative position and delta
					xr_old, yr_old = self.objects[position]['xy']
					dxr = (xr - xr_old)
					dyr = (yr - yr_old)

					# append in the circular buffer
					self.objects[position]['dxr_arr'].append(dxr)
					self.objects[position]['dyr_arr'].append(dyr)
					self.objects[position]['w_arr'].append(w)
					self.objects[position]['h_arr'].append(h)

					for pos in ['dxr_arr', 'dyr_arr', 'w_arr', 'h_arr']:
						if len(self.objects[position][pos]) == self.lookback:
							self.objects[position][pos].pop(0)

					# predict next location delta of the object
					dxrp = np.mean(self.objects[position]['dxr_arr'])
					dyrp = np.mean(self.objects[position]['dyr_arr'])
					wp = np.mean(self.objects[position]['w_arr'])
					hp = np.mean(self.objects[position]['h_arr'])

					# average the prediction with the actual inferred position
					# this extra averagingt reduces oscillations
					xrp = (xr_old + dxrp + xr)/2
					yrp = (yr_old + dyrp + yr)/2

					# record new position and box sizes
					self.objects[position]['xy'] = [xrp, yrp]
					self.objects[position]['wh'] = [wp, hp]
					self.objects[position]['life'] = np.clip(self.objects[position]['life'] + self.add_det, 0, self.det_max)

				# then it means there is a new object
				else:
					self.add_new_object(xr, yr, w, h, class_idx)
			else:
				self.add_new_object(xr, yr, w, h, class_idx)

		return self.objects



class Projector(object):
	def __init__(self, wt, wb, input_h, alpha = np.pi/8, H=100, W=100):
		super().__init__()
		self.wt = wt
		self.wb = wb
		self.alpha = alpha
		self.H = H
		self.W = W
		self.input_h = input_h

		self.projection = np.zeros((H, W))

		self.sineAlpha = np.sin(alpha)
		print("sine alpha", self.sineAlpha)

	def project(self, objects):
		for obj in objects:
			xr, yr = obj[0]
			w, h = obj[1]
			yr = np.clip(yr + h/2, 0, 1)

			wprime = (1 - yr*(1-self.wb))

			xrp = xr*wprime + (1-wprime)/2

			yrp = yr

			xrp = np.clip(xrp, 0, 0.99)
			yrp = np.clip(yrp, 0, 0.99)

			x = int(xrp*self.W)
			y = int(yrp*self.H)

			self.projection[y,x] += 1

		image = self.projection/self.projection.max()*255

		return np.array(image, dtype=np.uint8)

	def reset(self):
		self.projection = np.zeros((H, W))









