
import os
import tensorflow as tf
import numpy as np
import pandas
import torch
import transformers



"""
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
tokenizer = transformers.BertTokenizer.from_pretrained(model_name)
bert = transformers.BertModel.from_pretrained(model_name)

x = tokenizer.encode('this is a text', return_tensors = 'pt').to(device)
y = bert(x)

y[0].detach().numpy()

"""

def prepare_batching(nb_items, batch_size):
  nb_batches = nb_items // batch_size
  last_batch_size = nb_items - nb_batches * batch_size
  if(last_batch_size > 0):
    nb_batches += 1
  else:
    last_batch_size = batch_size
  return nb_batches, last_batch_size


def tokenize_texts(device, tokenizer, text_data):
  return text_data.text.apply(lambda text : tokenizer.encode(text, return_tensors = 'pt').to(device))


def batch_tensors(batch_size, batch_index, nb_tensors_to_batch, tensor_series):
  tensor_index_start = batch_size * batch_index
  tensor_index_end = tensor_index_start + nb_tensors_to_batch
  tensor_dims = [ t.shape[-1] for t in tensor_series[tensor_index_start : tensor_index_end] ]
  max_len = max(tensor_dims)
  batch = torch.zeros(nb_tensors_to_batch, max_len, dtype = torch.int)
  for i in range(nb_tensors_to_batch):
    tensor_index = tensor_index_start + i
    batch[i, : tensor_dims[i]] = tensor_series[tensor_index][0,:]
  return batch, tensor_dims


def extract_features(text_data, dest_dir, model_name = 'bert-base-uncased', batch_size = 64, allow_pickle = True):
  
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  
  tokenizer = transformers.BertTokenizer.from_pretrained(model_name)
  tokenized_texts = tokenize_texts(device, tokenizer, text_data)
  
  bert = transformers.BertModel.from_pretrained(model_name)
  
  nb_texts = len(text_data)
  output_ids = nb_texts * [ None, ]
  output_paths = nb_texts * [ None, ] # this is redundant information
  

  for item_index in range(nb_texts):
    features = bert(tokenized_texts[item_index])[1].detach().numpy()[0, :]
    item_id = text_data.loc[item_index].id
    output_ids[item_index] = item_id
    feature_file_name = f'tf-{item_id:05d}.npy'
    output_paths[item_index] = feature_file_name
    np.save(os.path.join(dest_dir, feature_file_name),
            features,
            allow_pickle = allow_pickle,
            fix_imports = False)
    #if(item_index > 10):
    #  break
    
  """
  # batching daoesnt work like that
  
  nb_batches, last_batch_size = prepare_batching(nb_texts, batch_size)
  
  item_index = 0
  
  for batch_index in range(nb_batches):
    
    current_batch_size = batch_size if(batch_index < nb_batches - 1) else last_batch_size
    
    batch, original_lengths = batch_tensors(batch_size, batch_index, current_batch_size, tokenized_texts)
    
    batch_features = bert(batch)[0].detach().numpy()
    
    for index_in_batch in range(current_batch_size):
      item_index = batch_index * batch_size + index_in_batch
      item_id = text_data.loc[item_index].id
      output_ids[item_index] = item_id
      feature_file_name = f'tf-{item_id:05d}.npy'
      output_paths[item_index] = feature_file_name
      np.save(os.path.join(dest_dir, feature_file_name),
              batch_features[index_in_batch, original_lengths[:index_in_batch]],
              allow_pickle = allow_pickle,
              fix_imports = False)
    
  """
    
  
  output_data = pandas.DataFrame()
  output_data['id'] = output_ids
  output_data['file_name'] = output_paths
  
  return output_data


class MemeClassifierModel(tf.keras.Model):
  
  def __init__(self,
               image_features_dimension,
               text_features_dimension,
               return_logits = False,
               dropout = 0.1,
               has_preprocessing_layers = True,
               nb_hidden_layers = 0):
    
    super(MemeClassifierModel, self).__init__()
    
    self.image_input_size = image_features_dimension
    self.text_input_size = text_features_dimension
    self.return_logits = return_logits
    self.split_sizes = [ self.image_input_size, self.text_input_size ]
    self.has_preprocessing_layers = has_preprocessing_layers
    self.nb_hidden_layers = nb_hidden_layers
    self.hidden_size = self.image_input_size + self.text_input_size
    
    if(self.has_preprocessing_layers):
      self.image_feature_layer = tf.keras.layers.Dense(self.image_input_size, activation = 'relu', name = 'image_preprocessing_layer')
      self.text_feature_layer = tf.keras.layers.Dense(self.text_input_size, activation = 'relu', name = 'text_preprocessing_layer')
    
    self.feature_concatenator = tf.keras.layers.Concatenate(name = 'image_text_concat')
    
    self.feature_dropout = tf.keras.layers.Dropout(0.1, name = 'feature_dropout')
    
    self.hidden_layers = []
    for i in range(self.nb_hidden_layers):
      self.hidden_layers.append(tf.keras.layers.Dense(self.hidden_size, activation = 'relu', name = f'hidden_layer_{i}'))
      self.hidden_layers.append(tf.keras.layers.Dropout(0.1, name = f'hidden_layer_dropout_{i}'))
    
    if(not self.return_logits):
      self.classifier_output = tf.keras.layers.Dense(1, activation = 'sigmoid', name = 'classifier')
    else:
      self.classifier_output = tf.keras.layers.Dense(1, name = 'classifier')
  
  def call(self, inputs, training):
    
    if(self.has_preprocessing_layers):
      image_features, text_features = tf.split(inputs, self.split_sizes, axis = 1, name = 'text_image_split')
      x_image = self.image_feature_layer(image_features)
      x_text = self.text_feature_layer(text_features)
      x = self.feature_concatenator([x_image, x_text])
    else:
      x = inputs
    
    if(training):
      x = self.feature_dropout(x)
    
    for i in range(self.nb_hidden_layers):
      x = self.hidden_layers[2 * i](x)
      if(training): # dropout layer
        x = self.hidden_layers[2 * i + 1](x)
    
    output = self.classifier_output(x)
    return output
  
def inceptionBertClassifierModel(MemeClassifierModel, image_input_size, text_input_size):
  meme_classifier = MemeClassifierModel(image_input_size,
                                      text_input_size,
                                      has_preprocessing_layers = True,
                                      nb_hidden_layers = 2,
                                      dropout = 0.2)

  optimizer = tf.keras.optimizers.Adam()
  meme_classifier.compile(optimizer = optimizer,
                          loss = tf.keras.losses.BinaryCrossentropy(from_logits = False),
                          metrics = [ 'accuracy', tf.keras.metrics.AUC(curve = 'ROC', name = 'AUC_ROC') ])











