import numpy as np
from PIL import Image
from pickle import load
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from utils.model import CNNModel, generate_caption_beam_search
import os
import cv2

from config import config

"""
    *Some simple checking
"""
assert type(config['max_length']) is int, 'Please provide an integer value for `max_length` parameter in config.py file'
assert type(config['beam_search_k']) is int, 'Please provide an integer value for `beam_search_k` parameter in config.py file'

def central_crop(image, central_fraction):
	"""Crop the central region of the image.
	Remove the outer parts of an image but retain the central region of the image
	along each dimension. If we specify central_fraction = 0.5, this function
	returns the region marked with "X" in the below diagram.
	   --------
	  |        |
	  |  XXXX  |
	  |  XXXX  |
	  |        |   where "X" is the central 50% of the image.
	   --------
	Args:
	image: 3-D array of shape [height, width, depth]
	central_fraction: float (0, 1], fraction of size to crop
	Raises:
	ValueError: if central_crop_fraction is not within (0, 1].
	Returns:
	3-D array
	"""
	if central_fraction <= 0.0 or central_fraction > 1.0:
		raise ValueError('central_fraction must be within (0, 1]')
	if central_fraction == 1.0:
		return image

	img_shape = image.shape
	depth = img_shape[2]
	fraction_offset = int(1 / ((1 - central_fraction) / 2.0))
	bbox_h_start = int(np.divide(img_shape[0], fraction_offset))
	bbox_w_start = int(np.divide(img_shape[1], fraction_offset))

	bbox_h_size = int(img_shape[0] - bbox_h_start * 2)
	bbox_w_size = int(img_shape[1] - bbox_w_start * 2)

	image = image[bbox_h_start:bbox_h_start+bbox_h_size, bbox_w_start:bbox_w_start+bbox_w_size]
	return image

# Extract features from each image in the directory
def extract_features(filename, model, model_type):
	if model_type == 'inceptionv3':
		from keras.applications.inception_v3 import preprocess_input
		target_size = (299, 299)
	elif model_type == 'inceptionv4':
		from inception_v4 import preprocess_input
		target_size = (299, 299)
  # image = np.asarray(cv2.imread(filename))[:,:,::-1]
  # # Convert the image pixels to a numpy array
  # image = central_crop(image, 0.875)
  # # Reshape data for the model
  # image = cv2.resize(image, (299, 299))
  # # Prepare the image for the CNN Model model
  # image = preprocess_input(image)
  # image = image.reshape(-1,299,299,3)
  # feature = model.predict(image, verbose=0)
	# Loading and resizing image
	image = np.asarray(cv2.imread(filename))[:,:,::-1]
	# Convert the image pixels to a numpy array
	image = central_crop(image, 0.875)
	# Reshape data for the model
	image = cv2.resize(image, (299, 299))
	# Prepare the image for the CNN Model model
	image = preprocess_input(image)
	image = image.reshape(-1,299,299,3)
	features = model.predict(image, verbose=0)
	return features

# Load the tokenizer
tokenizer_path = config['tokenizer_path']
tokenizer = load(open(tokenizer_path, 'rb'))

# Max sequence length (from training)
max_length = config['max_length']

# Load the model
caption_model = load_model(config['model_load_path'])

image_model = CNNModel(config['model_type'])

# Load and prepare the image
for image_file in os.listdir(config['test_data_path']):
	if(image_file.split('--')[0]=='output'):
		continue
	if(image_file.split('.')[1]=='jpg' or image_file.split('.')[1]=='jpeg'):
		print('Generating caption for {}'.format(image_file))
		# Encode image using CNN Model
		image = extract_features(config['test_data_path']+image_file, image_model, config['model_type'])
		# Generate caption using Decoder RNN Model + BEAM search
		generated_caption = generate_caption_beam_search(caption_model, tokenizer, image, max_length, beam_index=config['beam_search_k'])
		# Remove startseq and endseq
		caption = 'Caption: ' + generated_caption.split()[1].capitalize()
		for x in generated_caption.split()[2:len(generated_caption.split())-1]:
		    caption = caption + ' ' + x
		caption += '.'
		# Show image and its caption
		pil_im = Image.open(config['test_data_path']+image_file, 'r')
		fig, ax = plt.subplots(figsize=(8, 8))
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
		_ = ax.imshow(np.asarray(pil_im), interpolation='nearest')
		_ = ax.set_title("BEAM Search with k={}\n{}".format(config['beam_search_k'],caption),fontdict={'fontsize': '20','fontweight' : '40'})
		plt.savefig(config['test_data_path']+'output--'+image_file)