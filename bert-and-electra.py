import torch
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    print("device_count: {}".format(device_count))
    for device_num in range(device_count):
        print("device {} capability {}".format(
            device_num,
              torch.cuda.get_device_capability(device_num)))
        print("device {} name {}".format(device_num, torch.cuda.get_device_name(device_num)))
else:
    print("no cuda device")


seed_value= 0
os.environ['PYTHONHASHSEED'] = str(seed_value)
np.random.seed(seed_value)
random.seed(seed_value)
tf.random.set_seed(seed_value)

print(tf.__version__)

# predefine hyperparameters and model-type
model_name='bert_base_3'
current_lr= 2e-6
epoch=1
batch=32

train_ds = tf.data.Dataset.from_tensor_slices((train_texts, train_labels)).shuffle(len(train_texts), seed=seed_value).batch(batch)
val_ds = tf.data.Dataset.from_tensor_slices((val_texts, val_labels)).shuffle(len(val_texts), seed=seed_value).batch(batch)

if model_name == 'bert_base' or model_name == 'bert_base_sampled_3':
  model_path ='https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4'
  model_preprocess ='https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'

elif model_name == 'electra_base':
  model_path = 'https://tfhub.dev/google/electra_base/2'
  model_preprocess = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'

tfhub_handle_encoder = model_path
tfhub_handle_preprocess = model_preprocess

def build_model():
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(tfhub_handle_encoder, dtype=tf.float64, trainable=True, name=model_name+'_encoder')
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(0.3, seed=seed_value)(net)
    net = tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed_value), activation='sigmoid', name='classifier')(net)
    model=tf.keras.Model(text_input, net)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=current_lr),
                            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                            metrics=tf.metrics.BinaryAccuracy())
    model.summary()
    return model

from tensorflow.keras.callbacks import ModelCheckpoint

def namefile(model_name):
    filepath= "#models/tensorflow_"+model_name+".hdf5"
    return filepath

for fold in range(1):
    tf.keras.utils.set_random_seed(seed_value)
    os.environ["TF_DISABLE_SEGMENT_REDUCTION_OP_DETERMINISM_EXCEPTIONS"]="True"


    model = build_model()
    filepath= namefile(model_name)

    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]

    print(f'Training model with {tfhub_handle_encoder}')
    history = model.fit(x=train_ds,
                                    validation_data=val_ds,
                                    epochs=epoch,
                                    callbacks=callbacks_list,
                                    shuffle=False)

def load_model(file):
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
    preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)
    encoder = hub.KerasLayer(tfhub_handle_encoder, dtype=tf.float64, trainable=True, name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(0.3, seed=seed_value)(net)
    net = tf.keras.layers.Dense(1, kernel_initializer=tf.keras.initializers.GlorotUniform(seed=seed_value), activation='sigmoid', name='classifier')(net)
    model=tf.keras.Model(text_input, net)
    model.load_weights(file)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=current_lr),
                            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                             metrics=tf.metrics.BinaryAccuracy())

    return model    

from sklearn.metrics import roc_auc_score, balanced_accuracy_score, average_precision_score, f1_score

filepath= namefile(model_name)
model = load_model(filepath)
results = model.predict(test_texts)

roc_score = roc_auc_score(test_labels, results)
apr = average_precision_score(test_labels, results)
apr_rev = average_precision_score(test_labels, results, pos_label=0)
print("roc_scoreL: ", roc_score)
print("avg prec score: ", apr)
print("avg prec score reverse: ", apr_rev)



# total = len(results)
# tp = 0;
# fp = 0;
# tn = 0;
# fn = 0


# x = 0.13
# for i in range(len(results)):
#   if results[i] >= x and test_labels[i] == 1:
#     tp = tp + 1
#   elif results[i] >= x and test_labels[i] == 0:
#     fp = fp + 1
#   elif results[i] < x and test_labels[i] == 1:
#     fn = fn + 1
#   elif results[i] < x and test_labels[i] == 0:
#     tn = tn + 1
