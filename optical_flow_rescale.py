import cv2
import os


def img_resize(dataset,out_size):
	for root, dirs, files in os.walk(os.path.join(dataset)):
		for file in files:
			abspath = os.path.join(root, file)
			img = cv2.imread(abspath, 1)
			height = img.shape[0]		# 0 height, 1 width
			width = img.shape[1]
			print(abspath)
			if height > width:
				upper_bound = int((height - width) / 2)
				lower_bound = int(upper_bound + width)
				img_center = img[upper_bound:lower_bound, 0:width]
				img_resized = cv2.resize(img_center, (out_size, out_size))
				cv2.imwrite(abspath, img_resized)
			else:
				left_bound = int((width - height) / 2)
				right_bound = int(left_bound + height)
				img_center = img[0:height, left_bound:right_bound]  # 先上下， 后左右
				img_resized = cv2.resize(img_center, (out_size, out_size))
				cv2.imwrite(abspath, img_resized)

img_resize('CASME2FLOW',96)

