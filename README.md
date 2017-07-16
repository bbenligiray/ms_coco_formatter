# MS COCO Formatter

A tool to download and format MS COCO dataset

It outputs a .h5 file that contains the following:

* data_types: 'train', 'val' and 'test'
* cats: names of the 80 categories
(replace x with any data type)
* x_images: flattened images (not preprocessed in any way)
* x_shapes: shapes of the images, to reshape the flattened images
* x_names: file names of the images
* x_label: a one-hot integer vector of labels