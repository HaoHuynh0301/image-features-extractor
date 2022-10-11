import numpy as np
import os
import cv2
from pickle import dump
import string
from tqdm import tqdm
from utils.model import CNNModel
from keras.preprocessing.image import load_img, img_to_array
from datetime import datetime as dt
from inception_v4 import preprocess_input

# Utility function for pretty printing
def mytime(with_date=False):
	_str = ''
	if with_date:
		_str = str(dt.now().year)+'-'+str(dt.now().month)+'-'+str(dt.now().day)+' '
		_str = _str+str(dt.now().hour)+':'+str(dt.now().minute)+':'+str(dt.now().second)
	else:
		_str = str(dt.now().hour)+':'+str(dt.now().minute)+':'+str(dt.now().second)
	return _str

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

"""
	*This function returns a dictionary of form:
	{
		image_id1 : image_features1,
		image_id2 : image_features2,
		...
	}
"""
def extract_features(path, model_type):
	if model_type == 'inceptionv3':
		from keras.applications.inception_v3 import preprocess_input
		target_size = (299, 299)
	elif model_type == 'inceptionv4':
		from .inception_v4 import preprocess_input
		target_size = (299, 299)
	# Get CNN Model from model.py
	model = CNNModel(model_type)
	features = dict()
	# Extract features from each photo
	for name in tqdm(os.listdir(path)):
		filename = path + name
		image = np.asarray(cv2.imread(filename))[:,:,::-1]
		# Convert the image pixels to a numpy array
		image = central_crop(image, 0.875)
		# Reshape data for the model
		image = cv2.resize(image, (299, 299))
		# Prepare the image for the CNN Model model
		image = preprocess_input(image)
		image = image.reshape(-1,299,299,3)
		feature = model.predict(image, verbose=0)
		# Store encoded features for the image
		image_id = name.split('.')[0]
		features[image_id] = feature
	return features

"""
	*Extract captions for images
	*Glimpse of file:
		1000268201_693b08cb0e.jpg#0	A child in a pink dress is climbing up a set of stairs in an entry way .
		1000268201_693b08cb0e.jpg#1	A girl going into a wooden building .
		1000268201_693b08cb0e.jpg#2	A little girl climbing into a wooden playhouse .
		1000268201_693b08cb0e.jpg#3	A little girl climbing the stairs to her playhouse .
		1000268201_693b08cb0e.jpg#4	A little girl in a pink dress going into a wooden cabin .
"""
def load_captions(filename):
	file = open(filename, 'r')
	doc = file.read()
	file.close()
	"""
	Captions dict is of form:
	{
		image_id1 : [caption1, caption2, etc],
		image_id2 : [caption1, caption2, etc],
		...
	}
	"""
	captions = dict()
	# Process lines by line
	_count = 0
	for line in doc.split('\n'):
		# Split line on white space
		tokens = line.split()
		if len(line) < 2:
			continue
		# Take the first token as the image id, the rest as the caption
		image_id, image_caption = tokens[0], tokens[1:]
		# Extract filename from image id
		image_id = image_id.split('.')[0]
		# Convert caption tokens back to caption string
		image_caption = ' '.join(image_caption)
		# Create the list if needed
		if image_id not in captions:
			captions[image_id] = list()
		# Store caption
		captions[image_id].append(image_caption)
		_count = _count+1
	print('{}: Parsed captions: {}'.format(mytime(),_count))
	return captions

def clean_captions(captions):
	# Prepare translation table for removing punctuation
	table = str.maketrans('', '', string.punctuation)
	for _, caption_list in captions.items():
		for i in range(len(caption_list)):
			caption = caption_list[i]
			# Tokenize i.e. split on white spaces
			caption = caption.split()
			# Convert to lowercase
			caption = [word.lower() for word in caption]
			# Remove punctuation from each token
			caption = [w.translate(table) for w in caption]
			# Remove hanging 's' and 'a'
			caption = [word for word in caption if len(word)>1]
			# Remove tokens with numbers in them
			caption = [word for word in caption if word.isalpha()]
			# Store as string
			caption_list[i] =  ' '.join(caption)

"""
	*Save captions to file, one per line
	*After saving, captions.txt is of form :- `id` `caption`
		Example : 2252123185_487f21e336 stadium full of people watch game
"""
def save_captions(captions, filename):
	lines = list()
	for key, captions_list in captions.items():
		for caption in captions_list:
			lines.append(key + ' ' + caption)
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()

def preprocessData(config):
	print('{}: Using {} model'.format(mytime(),config['model_type'].title()))
	# Extract features from all images
	if os.path.exists(config['model_data_path']+'features_'+str(config['model_type'])+'.pkl'):
		print('{}: Image features already generated at {}'.format(mytime(), config['model_data_path']+'features_'+str(config['model_type'])+'.pkl'))
	else:
		print('{}: Generating image features using '+str(config['model_type'])+' model...'.format(mytime()))
		features = extract_features(config['images_path'], config['model_type'])
		# Save to file
		dump(features, open(config['model_data_path']+'features_'+str(config['model_type'])+'.pkl', 'wb'))
		print('{}: Completed & Saved features for {} images successfully'.format(mytime(),len(features)))
	# Load file containing captions and parse them
	if os.path.exists(config['model_data_path']+'captions.txt'):
		print('{}: Parsed caption file already generated at {}'.format(mytime(), config['model_data_path']+'captions.txt'))
	else:
		print('{}: Parsing captions file...'.format(mytime()))
		captions = load_captions(config['captions_path'])
		# Clean captions
		# Ignore this function because Tokenizer from keras will handle cleaning
		# clean_captions(captions)
		# Save captions
		save_captions(captions, config['model_data_path']+'captions.txt')
		print('{}: Parsed & Saved successfully'.format(mytime()))