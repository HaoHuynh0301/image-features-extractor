"""
history of what has been done
"""

from text_features.extract import extract_features as text_extract_features
from image_features.extract import extract_features as img_extract_features
from meme_classifier_model import MemeClassifierModel
import meme_classifier_model
import tensorflow as tf
import datetime
import seaborn
import numpy as np
from matplotlib import pyplot
import os
from text_features.extract import *
import text_features.extract
from importlib import reload
from image_features.extract import *
import image_features.extract
import pandas
import sys

print('This is not a script to run')
sys.exit(0)
raise Exception('JuST iN cASe')

##### eda -> done


##### image feature extraction -> done

#from importlib import reload


d = pandas.read_json('data/train.jsonl', lines=True)

#image_features.extract = reload(image_features.extract)

output_data = extract_features(d, 'data', 'data/img-features')


output_data.to_csv('data/img-features.csv', index=False)


##### text feature extraction -> done


d = pandas.read_json('data/train.jsonl', lines=True)

text_features.extract = reload(text_features.extract)

output_data = extract_features(d, 'data/text-features')

output_data.to_csv('data/text-features.csv', index=False)

#### save data -> TODO

# ! tar cf features.tar img-features text-features img-features.csv text-features.csv
# ! gzip features.tar


#### eda on the features -> done


text_features_npy_dir = 'data/text-features'
image_features_npy_dir = 'data/img-features'

text_data = pandas.read_csv('data/text-features.csv', index_col='id')
image_data = pandas.read_csv('data/img-features.csv', index_col='id')

nb_items = len(text_data)
item_indexes = text_data.index


image_mins = nb_items * [None, ]
image_maxs = nb_items * [None, ]
text_mins = nb_items * [None, ]
text_maxs = nb_items * [None, ]

text_features_max_shape = [0, ]

i = 0

for item_index in item_indexes:
  text_features = np.load(os.path.join(
      text_features_npy_dir, text_data.loc[item_index].file_name))
  text_mins[i] = text_features.min()
  text_maxs[i] = text_features.max()
  shape = text_features.shape
  for d in range(1):
    if (shape[d] > text_features_max_shape[d]):
      text_features_max_shape[d] = shape[d]
  image_features = np.load(os.path.join(
      image_features_npy_dir, image_data.loc[item_index].file_name))
  image_mins[i] = image_features.min()
  image_maxs[i] = image_features.max()
  i += 1

meta_data = pandas.DataFrame()
meta_data['id'] = item_indexes
meta_data['text_min'] = text_mins
meta_data['text_max'] = text_maxs
meta_data['image_min'] = image_mins
meta_data['image_max'] = image_maxs

meta_data.boxplot(column=['text_min', 'text_max'])
pyplot.show()
meta_data.boxplot(column=['image_min', 'image_max'])
pyplot.show()

seaborn.violinplot(data=meta_data[[
                   'text_min', 'text_max', 'image_max']], inner='quartile', color='white')
pyplot.show()
seaborn.violinplot(data=meta_data[['image_max']],
                   inner='quartile', color='white')
pyplot.show()

print(f'image min-max: {min(image_mins)}, {max(image_maxs)}')
print(f'text min-max: {min(text_mins)}, {max(text_maxs)}')
print(f'text max shape: {text_features_max_shape}')

"""
>>> print(f'image min-max: {min(image_mins)}, {max(image_maxs)}')
image min-max: 0.0, 8.585090637207031
>>> print(f'text min-max: {min(text_mins)}, {max(text_maxs)}')
text min-max: -1.0, 1.0

"""


#### classifier -> done

# nrmalize the features
# concatenate
# dense layer
# softmax with finetunable temperature: need to separate test and eval

# TODO: optmizer - adam or something
# TODO: loss='binary_crossentropy' - output the logits?

# TODO: train pipeline
# TODO: monitor with tensorboard (need to set up logging)


"""
text_features_npy_dir = 'data/text-features'
image_features_npy_dir = 'data/img-features'

text_data = pandas.read_csv('data/text-features.csv', index_col = 'id')
image_data = pandas.read_csv('data/img-features.csv', index_col = 'id')

nb_items = len(text_data)
item_indexes = text_data.index
"""


image_input_size = 2048
text_input_size = 768

train_batch_size = 16
train_nb_epochs = 600

main_log_dir = 'logs'
main_checkpoint_dir = 'checkpoints'


# training pipeline: since its not a lot of data, just load everything in memory thhen data set from tensor slices

text_features_npy_dir = 'data/text-features'
image_features_npy_dir = 'data/img-features'

text_data = pandas.read_csv('data/text-features.csv', index_col='id')
image_data = pandas.read_csv('data/img-features.csv', index_col='id')


def load_features(data, data_dir, feature_size):
  nb_items = len(data)
  item_indexes = data.index
  all_the_data = np.zeros((nb_items, feature_size))
  for i, item_id in enumerate(item_indexes):
    file_path = os.path.join(data_dir, data.loc[item_id]['file_name'])
    all_the_data[i, :] = np.load(file_path)
  return all_the_data


all_the_text_features = load_features(
    text_data, 'data/text-features', text_input_size)
all_the_image_features = load_features(
    image_data, 'data/img-features', image_input_size)

d = pandas.read_json('data/train.jsonl', lines=True)

all_the_labels = d.label.values

# TODO: train/test/eval split
"""
train_text_features = tf.data.Dataset.from_tensor_slices(all_the_text_features)
train_image_features = tf.data.Dataset.from_tensor_slices(all_the_image_features)
train_labels = tf.data.Dataset.from_tensor_slices(all_the_labels)


train_dataset = tf.data.Dataset.zip(( train_image_features, train_text_features))
train_dataset = tf.data.Dataset.zip(( train_dataset, train_labels ))

train_dataset = train_dataset.shuffle(10000).padded_batch(train_batch_size).prefetch(tf.data.experimental.AUTOTUNE)
"""
nb_items = len(d)
all_the_inputs = np.zeros((nb_items, image_input_size + text_input_size))
all_the_inputs[:, : image_input_size] = all_the_image_features
all_the_inputs[:, image_input_size:] = all_the_text_features


# model
meme_classifier_model = reload(meme_classifier_model)

meme_classifier = MemeClassifierModel(image_input_size,
                                      text_input_size,
                                      has_preprocessing_layers=True,
                                      nb_hidden_layers=2,
                                      dropout=0.2)


optimizer = tf.keras.optimizers.Adam()
meme_classifier.compile(optimizer=optimizer,
                        loss=tf.keras.losses.BinaryCrossentropy(
                            from_logits=False),
                        metrics=['accuracy', tf.keras.metrics.AUC(curve='ROC', name='AUC_ROC')])


# checkpoint TODO: useless but need to add in
"""
checkpoint = tf.train.Checkpoint(model = meme_classifier, optimizer = optimizer)
checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep = 5)


if (checkpoint_manager.latest_checkpoint):
  checkpoint.restore(checkpoint_manager.latest_checkpoint)
  print ('Latest checkpoint restored!!')
"""

#
model_name = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
log_dir = os.path.join(main_log_dir, model_name)
checkpoint_dir = os.path.join(main_checkpoint_dir, model_name)

train_callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(
        checkpoint_dir, checkpoint_dir), verbose=1, save_freq='epoch'),
    tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
]


train_nb_epochs = 10
validation_split_size = 0  # 0.1

meme_classifier.fit(x=all_the_inputs,
                    y=all_the_labels,
                    batch_size=train_batch_size,
                    epochs=train_nb_epochs,
                    verbose=1,
                    callbacks=train_callbacks,
                    validation_split=validation_split_size,
                    shuffle=True)


"""
model_name
'20201027-114755'
"""


# prepare submission


model_name = '20201027-114755'
test_data_file = 'data/test_seen.jsonl'

image_input_size = 2048
text_input_size = 768


data = pandas.read_json(test_data_file, lines=True)


image_features_data = img_extract_features(data, 'data', 'data/img-features')
text_features_data = text_extract_features(data, 'data/text-features')


all_the_image_features = load_features(
    image_features_data, 'data/img-features', image_input_size)
all_the_text_features = load_features(
    text_features_data, 'data/text-features', text_input_size)

nb_items = len(data)

all_the_inputs = np.zeros((nb_items, image_input_size + text_input_size))
all_the_inputs[:, : image_input_size] = all_the_image_features
all_the_inputs[:, image_input_size:] = all_the_text_features

probas = np.zeros(nb_items)

#TODO:
# load the model


# TODO: batch eval
for i in range(nb_items):
  probas[i] = meme_classifier(tf.expand_dims(
      all_the_inputs[i, :], 0), training=False)

labels = (probas >= 0.5).astype(int)

results_df = pandas.DataFrame()
results_df['id'] = data.id
results_df['proba'] = probas
results_df['label'] = labels

results_df.to_csv('test_seen_predictions.csv', index=False)
