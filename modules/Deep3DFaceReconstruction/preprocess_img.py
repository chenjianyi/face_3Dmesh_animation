import numpy as np 
from scipy.io import loadmat,savemat
from PIL import Image

#calculating least sqaures problem
def POS(xp,x):
	npts = xp.shape[1]

	A = np.zeros([2*npts,8])

	A[0:2*npts-1:2,0:3] = x.transpose()
	A[0:2*npts-1:2,3] = 1

	A[1:2*npts:2,4:7] = x.transpose()
	A[1:2*npts:2,7] = 1;

	b = np.reshape(xp.transpose(),[2*npts,1])

	k,_,_,_ = np.linalg.lstsq(A,b)

	R1 = k[0:3]
	R2 = k[4:7]
	sTx = k[3]
	sTy = k[7]
	s = (np.linalg.norm(R1) + np.linalg.norm(R2))/2
	t = np.stack([sTx,sTy],axis = 0)

	return t,s

def process_img(img,lm,t,s):
	w0,h0 = img.size
	img = img.transform(img.size, Image.AFFINE, (1, 0, t[0] - w0/2, 0, 1, h0/2 - t[1]))
	w = (w0/s*102).astype(np.int32)
	h = (h0/s*102).astype(np.int32)
	img = img.resize((w,h),resample = Image.BILINEAR)
	lm = np.stack([lm[:,0] - t[0] + w0/2,lm[:,1] - t[1] + h0/2],axis = 1)/s*102

	# crop the image to 224*224 from image center
	left = (w/2 - 112).astype(np.int32)
	right = left + 224
	up = (h/2 - 112).astype(np.int32)
	below = up + 224

	img = img.crop((left,up,right,below))
	img = np.array(img)
	img = img[:,:,::-1]
	img = np.expand_dims(img,0)
	lm = lm - np.reshape(np.array([(w/2 - 112),(h/2-112)]),[1,2])

	return img,lm


# resize and crop input images before sending to the R-Net
def Preprocess(img,lm,lm3D):

	w0,h0 = img.size

	# change from image plane coordinates to 3D sapce coordinates(X-Y plane)
	lm = np.stack([lm[:,0],h0 - 1 - lm[:,1]], axis = 1)

	# calculate translation and scale factors using 5 facial landmarks and standard landmarks
	t,s = POS(lm.transpose(),lm3D.transpose())

	# processing the image
	img_new,lm_new = process_img(img,lm,t,s)
	lm_new = np.stack([lm_new[:,0],223 - lm_new[:,1]], axis = 1)
	trans_params = np.array([w0,h0,102.0/s,t[0],t[1]])

	return img_new,lm_new,trans_params

import dlib
import cv2
from imutils import face_utils

def get_5_face_landmarks(img, face_landmarks='/mnt/mfs/chenjianyi/project/Deepfake/3DDFA/models/shape_predictor_68_face_landmarks.dat'):
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(face_landmarks)
	
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	rects = detector(gray, 0)
	
	def midpoint(p1, p2):
	    coords = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
	    return [int(x) for x in coords]
	
	if len(rects) > 0:
		for rect in rects:
			x = rect.left()
			y = rect.top()
			w = rect.right()
			h = rect.bottom()
			shape = predictor(gray, rect)
			shape_np = face_utils.shape_to_np(shape).tolist()
			left_eye = midpoint(shape_np[36], shape_np[39])
			right_eye = midpoint(shape_np[42], shape_np[45])
			features = [left_eye, right_eye, shape_np[33], shape_np[48], shape_np[54]]
			#return features
			return np.array(features)
