import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import scipy.io as io
import math

def toAffinity(f):
	T = f[0:2]
	scale = f[2]
	theta = f[3]
	A = np.zeros((3, 3), np.float32)
	A[0, 0] = scale * math.cos(theta)
	A[0, 1] = scale * math.sin(theta) * (-1)
	A[1, 0] = scale * math.sin(theta)
	A[1, 1] = scale * math.cos(theta)
	A[0:2, 2] = T
	A[2, 0] = 0
	A[2, 1] = 0
	A[2, 2] = 1
	return A

def centering(x):
	T = np.zeros((3,3), np.float32)
	T[0, 0] = 1
	T[1, 1] = 1
	T[0:2, 2] = (-1) * np.mean(x[0:2, :], axis=1)
	T[2, :] = [0, 0, 1]
	x = np.matmul(T, x)
	std1 = x[0,:].std(axis=0)
	std2 = x[1,:].std(axis=0)

	# at least one pixel apart to avoid numerical problems
	std1 = np.maximum(std1, 1)
	std2 = np.maximum(std2, 1)

	S = np.array([[1.0/std1, 0, 0], [0, 1.0/std2, 0], [0, 0, 1]], np.float32)
	C = np.matmul(S, T)
	return C

ins_folder = '/home/yimeng/Datasets/Piyush/mavic_images/'

# instance image
im_ins = cv2.imread(ins_folder + 'mavic sequence0000.jpg', 1)
# reference image
im_ref = cv2.imread('/home/yimeng/Datasets/Piyush/base.tif', 1)

#im_ins_gray = cv2.cvtColor(im_ins, cv2.COLOR_BGR2GRAY)
#sift = cv2.xfeatures2d.SIFT_create()
#ins_keypoints = sift.detect(im_ins_gray, None)
#sift = cv2.SIFT()
#ins_kp, ins_des = sift.detectAndCompute(im_ins, None)

#temp_ins = cv2.drawKeypoints(im_ins, ins_kp)

# load keypoints and descriptors from mat files for instance and reference image
ins_mat = io.loadmat('mavic sequence0000_sift.mat')
ref_mat = io.loadmat('base_sift.mat')

ins_kp, ins_des = ins_mat['f'].transpose().astype(np.float32), ins_mat['d'].transpose().astype(np.float32)
ref_kp, ref_des = ref_mat['f'].transpose().astype(np.float32), ref_mat['d'].transpose().astype(np.float32)
''' make the number of features smaller
ins_kp = ins_kp[0:99, :]
ins_des = ins_des[0:99, :]
ref_kp = ref_kp[0:999, :]
ref_des = ref_des[0:999, :]
'''

show_ins = im_ins.copy()

num_kp, _ = ins_kp.shape
#''' draw keypoints as circles
for i in range(num_kp):
	x, y = int(ins_kp[0, i]), int(ins_kp[1, i])
	show_ins = cv2.circle(show_ins, (x, y), 1, (0, 255, 0))

plt.imshow(show_ins)
plt.savefig('show_ins.png', dpi=(400))
#'''

#'''
# FLANN parameters
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary

flann = cv2.FlannBasedMatcher(index_params, search_params)
print 'start knnMatch ............'
matches = flann.knnMatch(ins_des, ref_des, k=2)

np_matches = np.zeros((len(matches), 6), dtype=np.float32)
for i in range(len(matches)):
	m, n = matches[i]
	np_matches[i, 0] = m.distance
	np_matches[i, 1] = m.queryIdx
	np_matches[i, 2] = m.trainIdx
	np_matches[i, 3] = n.distance
	np_matches[i, 4] = n.queryIdx
	np_matches[i, 5] = n.trainIdx

np.save('matches.npy', np_matches)
#'''

np_matches = np.load('matches.npy')

#'''
# ratio test as per Lowe's paper
good = []
for id in range(np_matches.shape[0]):
	#print('m.distance = %f, n.distance = %f\n' % (m.distance, n.distance))
	m_dis, m_queryIdx, _, n_dis, _, _ = np_matches[id, :]
	if m_dis < 0.8 * n_dis:
		good.append(id)

#'''

#''' draw all the keypoints as circles
show_ins = im_ins.copy()
for i in range(ins_kp.shape[0]):
	x, y = int(ins_kp[i, 0]), int(ins_kp[i, 1])
	cv2.circle(show_ins, (x, y), 3, (0, 255, 0), 3)
plt.imshow(show_ins)
plt.savefig('show_ins.jpg', dpi=400)
plt.close()

show_ref = im_ref.copy()
for i in range(ref_kp.shape[0]):
	x, y = int(ref_kp[i, 0]), int(ref_kp[i, 1])
	show_ref = cv2.circle(show_ref, (x, y), 3, (0, 255, 0), 3)
plt.imshow(show_ref)
plt.savefig('show_ref.jpg', dpi=400)
plt.close()

#draw good keypoints as circles
show_ins = im_ins.copy()
for i in range(len(good)):
	ins_id = int(np_matches[good[i], 1])
	x, y = int(ins_kp[ins_id, 0]), int(ins_kp[ins_id, 1])
	cv2.circle(show_ins, (x, y), 20, (0, 255, 0), 20)
plt.imshow(show_ins)
plt.savefig('show_ins_good.jpg', dpi=400)
plt.close()

show_ref = im_ref.copy()
for i in range(len(good)):
	ref_id = int(np_matches[good[i], 2]) #trainIdx
	x, y = int(ref_kp[ref_id, 0]), int(ref_kp[ref_id, 1])
	cv2.circle(show_ref, (x, y), 20, (0, 255, 0), 20)
plt.imshow(show_ref)
plt.savefig('show_ref_good.jpg', dpi=400)
plt.close()
#'''

#''' draw matches
if im_ins.shape[0] >= im_ref.shape[0]:
	combined_height = im_ins.shape[0]
else:
	combined_height = im_ref.shape[0]
combined_width = im_ins.shape[1] + im_ref.shape[1]

im_show = np.zeros((combined_height, combined_width, 3), np.uint8)

im_show[0:im_ins.shape[0], 0:im_ins.shape[1], :] = im_ins
im_show[0:im_ref.shape[0], im_ins.shape[1]:combined_width, :] = im_ref

#draw keypoints as circles and draw matches as lines
for i in range(len(good)):
	ins_id = int(np_matches[good[i], 1]) #queryIdx
	x, y = int(ins_kp[ins_id, 0]), int(ins_kp[ins_id, 1])
	cv2.circle(im_show, (x, y), 20, (0, 255, 0), 20)
for i in range(len(good)):
	ref_id = int(np_matches[good[i], 2]) #trainIdx
	x, y = int(ref_kp[ref_id, 0]) + im_ins.shape[1], int(ref_kp[ref_id, 1])
	cv2.circle(im_show, (x, y), 20, (0, 255, 0), 20)
for i in range(len(good)):
	ins_id = int(np_matches[good[i], 1]) #queryIdx
	x1, y1 = int(ins_kp[ins_id, 0]), int(ins_kp[ins_id, 1])
	ref_id = int(np_matches[good[i], 2]) #trainIdx
	x2, y2 = int(ref_kp[ref_id, 0]) + im_ins.shape[1], int(ref_kp[ref_id, 1])
	cv2.line(im_show, (x1,y1),(x2,y2),(255,0,0),5)

plt.imshow(im_show)
plt.savefig('matches.jpg', dpi=(400))
plt.close()
#'''

#''' geometric verification
ins_kp_good = np.zeros((len(good), 4), np.float32)
ref_kp_good = np.zeros((len(good), 4), np.float32)
for i in range(len(good)):
	ins_id = int(np_matches[good[i], 1]) #queryIdx
	ins_kp_good[i, :] = ins_kp[ins_id, :]
	ref_id = int(np_matches[good[i], 2]) #trainIdx
	ref_kp_good[i, :] = ref_kp[ref_id, :]
ins_kp_good = ins_kp_good.transpose()
ref_kp_good = ref_kp_good.transpose()

opts_tolerance1 = 20
opts_tolerance2 = 15
opts_tolerance3 = 8
opts_minInliers = 6
opts_numRefinementIterations = 5

numMatches = len(good)
inliers = {}
H = {}

x1 = ins_kp_good[0:2, :]
x2 = ref_kp_good[0:2, :]

x1hom = np.zeros((3, x1.shape[1]), np.float32)
x1hom[0:2, :] = x1
x1hom[2, :] = 1
x2hom = np.zeros((3, x2.shape[1]), np.float32)
x2hom[0:2, :] = x2
x2hom[2, :] = 1

for m in range(numMatches):
	for t in range(opts_numRefinementIterations):
		if t == 0:
			A1 = toAffinity(ins_kp_good[:, m])
			A2 = toAffinity(ref_kp_good[:, m])
			H21 = np.matmul(A2, np.linalg.inv(A1))
			x1p = np.matmul(H21[0:2, :], x1hom)
			tol = opts_tolerance1
		elif t <= 3:
			# affinity
			H21 = np.linalg.lstsq(x2[:, inliers[m]].T, x1hom[:, inliers[m]].T).T
			x1p = np.matmul(H21[0:2, :], x1hom)
			H21[2, :] = [0, 0, 1]
			tol = opts_tolerance2
		else:
			# homography
			x1in = x1hom[:, inliers[m]]
			x2in = x2hom[:, inliers[m]]

			S1 = centering(x1in)
			S2 = centering(x2in)
			x1c = np.matmul(S1, x1in)
			x2c = np.matmul(S2, x2in)

			r, c = x1c.shape
			M = np.zeros((r*2, c*3), np.float32)
			M[0:r, 0:c] = x1c
			M[r:2*r, c:2*c] = x1c
			M[0:r, 2*c:3*c] = x1c * (-x2c)[0,:][:,None]
			M[r:2*r, 2*c:3*c] = x1c * (-x2c)[1,:][:,None]

			U, D, _ = np.linalg.svd(M, full_matrices=False)
			H21 = np.reshape(H21[:, -1], (3,3)).T
			H21 = np.linalg.inv(S2).dot(H21).dot(S1)
			H21 = H21 / H21[:,-1][:,None]

			x1phom = H21.dot(x1hom)
			x1p = np.zeros((2, x1phom.shape[1]), np.float32)
			x1p[0,:] = x1phom[0,:] / x1phom[2,:]
			x1p[1,:] = x1phom[1,:] / x1phom[2,:]
			tol = opts_tolerance3

		dist2 = np.sum(np.power((x2 - x1p), 2), axis=0)
		inliers[m] = (dist2 < (tol^2)).nonzero()[0]
		H[m] = H21
		if len(inliers[m]) < opts_minInliers:
			break
		if len(inliers[m]) > 0.7 * numMatches:
			break

scores = []
for k, v in inliers.items():
	scores.append(len(v))
best = scores.index(max(scores))
best_inliers = inliers[best]
H = np.linalg.inv(H[best])

#''' draw matches
if im_ins.shape[0] >= im_ref.shape[0]:
	combined_height = im_ins.shape[0]
else:
	combined_height = im_ref.shape[0]
combined_width = im_ins.shape[1] + im_ref.shape[1]

im_show = np.zeros((combined_height, combined_width, 3), np.uint8)

im_show[0:im_ins.shape[0], 0:im_ins.shape[1], :] = im_ins
im_show[0:im_ref.shape[0], im_ins.shape[1]:combined_width, :] = im_ref

#draw keypoints as circles and draw matches as lines
for i in range(len(best_inliers)):
	ins_id = int(np_matches[best_inliers[i], 1]) #queryIdx
	x, y = int(ins_kp[ins_id, 0]), int(ins_kp[ins_id, 1])
	cv2.circle(im_show, (x, y), 20, (0, 255, 0), 20)
for i in range(len(best_inliers)):
	ref_id = int(np_matches[best_inliers[i], 2]) #trainIdx
	x, y = int(ref_kp[ref_id, 0]) + im_ins.shape[1], int(ref_kp[ref_id, 1])
	cv2.circle(im_show, (x, y), 20, (0, 255, 0), 20)
for i in range(len(best_inliers)):
	ins_id = int(np_matches[best_inliers[i], 1]) #queryIdx
	x1, y1 = int(ins_kp[ins_id, 0]), int(ins_kp[ins_id, 1])
	ref_id = int(np_matches[best_inliers[i], 2]) #trainIdx
	x2, y2 = int(ref_kp[ref_id, 0]) + im_ins.shape[1], int(ref_kp[ref_id, 1])
	cv2.line(im_show, (x1,y1),(x2,y2),(255,0,0),5)

plt.imshow(im_show)
plt.savefig('best_matches.jpg', dpi=(400))
plt.close()
#'''




'''
MIN_MATCH_COUNT = 10

if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([ins_kp[m.queryIdx  int(np_matches[m, 0])].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ref_kp[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()

    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)

    img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

else:
    print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
    matchesMask = None
'''