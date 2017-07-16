import os
import h5py
import zipfile
import shutil

import numpy as np
from PIL import Image

from pycocotools.coco import COCO 
from calculate_mean import calculate_mean


def main():
	# clean up
	for dir_name in ['train2014', 'val2014', 'test2014', 'test2015', 'annotations']:
		if os.path.isdir(dir_name):
			shutil.rmtree(dir_name)

	dataset_links = ['http://msvocds.blob.core.windows.net/coco2014/train2014.zip',
					'http://msvocds.blob.core.windows.net/coco2014/val2014.zip',
					'http://msvocds.blob.core.windows.net/coco2014/test2014.zip',
					'http://msvocds.blob.core.windows.net/coco2015/test2015.zip',
					'http://msvocds.blob.core.windows.net/annotations-1-0-3/instances_train-val2014.zip',
					'http://msvocds.blob.core.windows.net/annotations-1-0-3/person_keypoints_trainval2014.zip',
					'http://msvocds.blob.core.windows.net/annotations-1-0-3/captions_train-val2014.zip',
					'http://msvocds.blob.core.windows.net/annotations-1-0-4/image_info_test2014.zip',
					'http://msvocds.blob.core.windows.net/annotations-1-0-4/image_info_test2015.zip']
	
	for dataset_link in dataset_links:
		file_name = dataset_link.split('/')[-1]
		# download the zip file if it is not there
		if not os.path.isfile(file_name):
			os.system('wget -t0 -c ' + dataset_link)
		#extract the downloaded file
		with zipfile.ZipFile(file_name) as zf:
			zf.extractall()
	
	with open('cats') as f:
		cats = f.read().split('\n')
	data_types = ['train', 'val']
	
	f = h5py.File('ms_coco.h5', 'w')

	dt_uint8 = h5py.special_dtype(vlen=np.dtype('uint8'))
	dt_str = h5py.special_dtype(vlen=str)
	# write dataset types and categories to the .h5 file
	data_types_h = f.create_dataset('data_types', (len(data_types),), dtype=dt_str)
	for ind_data_type, data_type in enumerate(data_types):
		data_types_h[ind_data_type] = data_type
	cats_h = f.create_dataset('cats', (len(cats),), dtype=dt_str)
	for ind_cat, cat in enumerate(cats):
		cats_h[ind_cat] = cat

	for data_type in data_types:
		# read the annotations file
		ann_path = os.path.join('annotations', 'instances_' + data_type + '2014.json')
		coco = COCO(ann_path)
		# our cat indices go from 0 to 79
		# MS COCO IDs are a bit wonkier, so we have to convert them
		ids_coco = coco.getCatIds()
		image_ids_of_cats = []

		# get list of image ids for each cat
		for ind_cat in range(len(cats)):
			id_coco = ids_coco[ind_cat]
			image_ids_of_cats.append(coco.getImgIds(catIds=id_coco))

		# create one-hot label vectors for each image
		image_ids = []
		labels = []
		for ind_cat in range(len(cats)):
			for image_id in image_ids_of_cats[ind_cat]:
				if image_id in image_ids:
					labels[image_ids.index(image_id)][ind_cat] = 1
				else:
					image_ids.append(image_id)
					labels.append(np.zeros(len(cats), dtype=np.int))
					labels[-1][ind_cat] = 1

		image_h = f.create_dataset(data_type + '_images', (len(image_ids),), dtype=dt_uint8)
		name_h = f.create_dataset(data_type + '_image_names', (len(image_ids),), dtype=dt_str)
		shape_h = f.create_dataset(data_type + '_image_shapes', (len(image_ids), 3), dtype=np.int)
		label_h = f.create_dataset(data_type + '_labels', (len(image_ids), len(cats)), dtype=np.int)
		# read the images and write to the .h5 file
		for ind, image_id in enumerate(image_ids):
			coco_img = coco.loadImgs(image_id)
			image_path = os.path.join(data_type + '2014', coco_img[0]['file_name'])
			image = Image.open(image_path)
			np_image = np.array(image)
			# if the image is grayscale, repeat its channels to make it RGB
			if len(np_image.shape) == 2:
				np_image = np.repeat(np_image[:, :, np.newaxis], 3, axis=2)

			image_h[ind] = np_image.flatten()
			name_h[ind] = coco_img[0]['file_name']
			shape_h[ind] = np_image.shape
			label_h[ind] = labels[ind]
	f.close()
	for dir_name in ['train2014', 'val2014', 'test2014', 'test2015', 'annotations']:
		shutil.rmtree(dir_name)

	calculate_mean()

	# show random images to test
	f = h5py.File('ms_coco.h5', 'r')
	cats_h = f['cats']
	data_types_h = f['data_types']
	while True:
		ind_data_type = np.random.randint(0, len(data_types_h))
		data_type = data_types_h[ind_data_type]

		image_h = f[data_type + '_images']
		name_h = f[data_type + '_image_names']
		shape_h = f[data_type + '_image_shapes']
		label_h = f[data_type + '_labels']

		ind_image = np.random.randint(0, len(image_h))

		np_image = np.reshape(image_h[ind_image], shape_h[ind_image])
		image = Image.fromarray(np_image, 'RGB')
		image.show()

		print('Image type: ' + data_type)
		print('Image name: ' + name_h[ind_image])
		for ind_cat, cat in enumerate(cats_h):
			if label_h[ind_image][ind_cat] == 1:
				print cat
		raw_input("...")


if __name__ == "__main__":
    main()