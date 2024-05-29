# Copyright 2017  . All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ==============================================================================
"""Data loading and other utilities.

Use this file to first copy over and pre-process the Omniglot dataset.
Simply call
  python data_utils.py
"""

import pickle as pickle
import logging
import os
import subprocess

import numpy as np
from scipy.misc import imresize
from scipy.misc import imrotate
from scipy.ndimage import imread
from scipy.misc import imsave
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import defaultdict
import gzip
from scipy import ndimage
from itertools import combinations, permutations
import math
import random
import zipfile

MAIN_DIR = r'H:\datasets\omniglot'
MAIN_DIR = ''
#import cv2
REPO_LOCATION = 'https://github.com/brendenlake/omniglot.git'
REPO_DIR = os.path.join(MAIN_DIR, 'omniglot')
DATA_DIR = os.path.join(REPO_DIR, 'python')
TRAIN_DIR = os.path.join(DATA_DIR, 'images_background')
TEST_DIR = os.path.join(DATA_DIR, 'images_evaluation')
ONESHOT_DIR = os.path.join(DATA_DIR, r'one-shot-classification', r'all_runs')
DATA_FILE_FORMAT = os.path.join(MAIN_DIR, '%s_omni.pkl')

TRAIN_ROTATIONS = True  # augment training data with rotations
TEST_ROTATIONS = False  # augment testing data with rotations
IMAGE_ORIGINAL_SIZE = 105

def get_data_with_infos(data_type='train',images_count=20):
  """Get data in form suitable for episodic training.

  Returns:
    Train and test data as dictionaries mapping
    label to list of examples.
  """
#  with tf.gfile.GFile(DATA_FILE_FORMAT % 'train') as f:
  with gzip.open(DATA_FILE_FORMAT % data_type, 'rb') as f:  
    processed_data = pickle.load(f)

  data = defaultdict(list)
  infos = defaultdict(list)
  for data, processed_data in zip([data],
                                  [processed_data]):
    for image, label,info in zip(processed_data['images'],
                            processed_data['labels'],
                            processed_data['info']):
      if len(data[label])>=images_count:
        continue
      data[label].append((1-image/255).astype('float32'))
      infos[label].append(info)

  ok_num_examples = [len(ll) == images_count for _, ll in data.items()]
  assert all(ok_num_examples), 'Bad number of examples in omniglot data.'
  
  print('Number of labels in  data %s: %d,images_count:%d'%(data_type,len(data),images_count))

  return data,infos

def get_data(data_type='train',images_count=20):
  data,infos = get_data_with_infos(data_type,images_count=images_count)
  return data

  
def crawl_directory(directory, augment_with_rotations=False,
                    first_label=0,char_count_per_alphabet=-1,
                    parts_shuffled=False):
  """Crawls data directory and returns stuff."""
  label_idx = first_label
  images = []
  labels = []
  info = []

  # traverse root directory
  alphabet_counts = defaultdict(int)
  angles = [0, 90, 180, 270]
  
  if char_count_per_alphabet == -1:
    char_count_per_alphabet = 20
  
  characters = [ 'character'+str(c).zfill(2) for c in range(1,char_count_per_alphabet+1) ]
  for root, _, files in os.walk(directory):
#    fileflag = 0
    
    if len(files) > 0 :
      alphabet = root.split(os.path.sep)[-2]
      character = root.split(os.path.sep)[-1]
      if character not in characters:
        continue
      
      alphabet_counts[alphabet] += 1
      if char_count_per_alphabet > 0 and alphabet_counts[alphabet] > char_count_per_alphabet:
        continue      
    logging.info('Reading files from %s', root)
        
    for file_name in files:
      full_file_name = os.path.join(root, file_name)
      img1 = imread(full_file_name, flatten=True)


        
      for i, angle in enumerate(angles):
        if not augment_with_rotations and i > 0:
          break
        img = 255-img1
        img = imrotate(img, angle)
        img = 255- img
        images.append(img)
        label = label_idx + i
        labels.append(label)
        info.append(full_file_name)
#          imsave(r'H:\datasets\omniglot\temp\%d_%d.png'%(label,ii),img)
#          ii += 1          
       
#        fileflag = 1
        
#    if fileflag:
#      label_idx += 8 if augment_with_rotations else 1
    if len(files) > 0:  
      label_increment = 1
      if augment_with_rotations:
        label_increment *= len(angles)
        
      label_idx += label_increment
      
  return images, labels, info
 
  
def resize_images(images, new_width, new_height):
  """Resize images to new dimensions."""
  resized_images = np.zeros([images.shape[0], new_width, new_height],
                            dtype=np.float32)

  for i in range(images.shape[0]):
    resized_images[i, :, :] = imresize(images[i, :, :],
                                       [new_width, new_height],
                                       interp='bilinear',
                                       mode=None)
  return resized_images


def write_datafiles(directory, write_file,
                    resize=True, rotate=False,
                    parts_shuffled=False,
                    new_width=0, new_height=0,
                    first_label=0,char_count_per_alphabet=-1):
  """Load and preprocess images from a directory and write them to a file.

  Args:
    directory: Directory of alphabet sub-directories.
    write_file: Filename to write to.
    resize: Whether to resize the images.
    rotate: Whether to augment the dataset with rotations.
    new_width: New resize width.
    new_height: New resize height.
    first_label: Label to start with.

  Returns:
    Number of new labels created.
  """

  # these are the default sizes for Omniglot:
  imgwidth = IMAGE_ORIGINAL_SIZE
  imgheight = IMAGE_ORIGINAL_SIZE

  logging.info('Reading the data.')
  images, labels, info = crawl_directory(directory,
                                         augment_with_rotations=rotate,
                                         parts_shuffled=parts_shuffled,
                                         first_label=first_label,
                                         char_count_per_alphabet=char_count_per_alphabet)

  images_np = np.zeros([len(images), imgwidth, imgheight], dtype=np.uint8)
  labels_np = np.zeros([len(labels)], dtype=np.uint32)
  for i in range(len(images)):
    images_np[i, :, :] = images[i]
    labels_np[i] = labels[i]

  if resize:
    logging.info('Resizing images.')
    resized_images = resize_images(images_np, new_width, new_height)

    logging.info('Writing resized data in float32 format.')
    data = {'images': resized_images,
            'labels': labels_np,
            'info': info}
    with gzip.open(write_file, 'wb') as f:
      pickle.dump(data, f)
  else:
    logging.info('Writing original sized data in boolean format.')
    data = {'images': images_np,
            'labels': labels_np,
            'info': info}
    with gzip.open(write_file, 'wb') as f:
#    with tf.gfile.GFile(write_file, 'w') as f:
      pickle.dump(data, f)

  return len(np.unique(labels_np))


def maybe_download_data():
  """Download Omniglot repo if it does not exist."""
  if os.path.exists(REPO_DIR):
    logging.info('It appears that Git repo already exists.')
  else:
    logging.info('It appears that Git repo does not exist.')
    logging.info('Cloning now.')

    subprocess.check_output('git clone %s' % REPO_LOCATION, shell=True)

  if os.path.exists(TRAIN_DIR):
    logging.info('It appears that train data has already been unzipped.')
  else:
    logging.info('It appears that train data has not been unzipped.')
    logging.info('Unzipping now.')

    subprocess.check_output('unzip %s.zip -d %s' % (TRAIN_DIR, DATA_DIR),
                            shell=True)

  if os.path.exists(TEST_DIR):
    logging.info('It appears that test data has already been unzipped.')
  else:
    logging.info('It appears that test data has not been unzipped.')
    logging.info('Unzipping now.')

    subprocess.check_output('unzip %s.zip -d %s' % (TEST_DIR, DATA_DIR),
                            shell=True)

  if os.path.exists(ONESHOT_DIR):
    logging.info('It appears that test data has already been unzipped.')
  else:
    logging.info('It appears that test data has not been unzipped.')
    logging.info('Unzipping now.')

    subprocess.check_output('unzip %s.zip -d %s' % (ONESHOT_DIR, ONESHOT_DIR),
                            shell=True)
 

def preprocess_omniglot(IMAGE_NEW_SIZE = 28):
  """Download and prepare raw Omniglot data.

  Downloads the data from GitHub if it does not exist.
  Then load the images, augment with rotations if desired.
  Resize the images and write them to a pickle file.
  """
  
  maybe_download_data()

  num_labels=0

  directory = os.path.join(DATA_DIR, 'images_background')
  write_file = DATA_FILE_FORMAT % 'train' #+str(IMAGE_NEW_SIZE)
  num_labels = write_datafiles(
      directory, write_file, resize=True, rotate=TRAIN_ROTATIONS,
      new_width=IMAGE_NEW_SIZE, new_height=IMAGE_NEW_SIZE)
  print('ok %s'%write_file)


  
  directory = os.path.join(DATA_DIR, 'images_evaluation')
  write_file = DATA_FILE_FORMAT % ('test' +str(IMAGE_NEW_SIZE))
  write_datafiles(directory, write_file, resize=True, rotate=TEST_ROTATIONS,
                  new_width=IMAGE_NEW_SIZE, new_height=IMAGE_NEW_SIZE,
                  first_label=num_labels)
  print('ok %s'%write_file)


  directory = os.path.join(DATA_DIR, 'images_background')
  write_file = DATA_FILE_FORMAT % ('tiny1' +str(IMAGE_NEW_SIZE))
  num_labels = write_datafiles(
      directory, write_file, resize=True, rotate=TRAIN_ROTATIONS,
      new_width=IMAGE_NEW_SIZE, new_height=IMAGE_NEW_SIZE,char_count_per_alphabet=1)
  print('ok %s'%write_file)

  directory = os.path.join(DATA_DIR, 'images_background')
  write_file = DATA_FILE_FORMAT % ('tiny2' +str(IMAGE_NEW_SIZE))
  num_labels = write_datafiles(
      directory, write_file, resize=True, rotate=TRAIN_ROTATIONS,
      new_width=IMAGE_NEW_SIZE, new_height=IMAGE_NEW_SIZE,char_count_per_alphabet=2)
  print('ok %s'%write_file)
 
  
  directory = os.path.join(DATA_DIR, 'images_background')
  write_file = DATA_FILE_FORMAT % ( 'tiny5' +str(IMAGE_NEW_SIZE))
  num_labels = write_datafiles(
      directory, write_file, resize=True, rotate=TRAIN_ROTATIONS,
      new_width=IMAGE_NEW_SIZE, new_height=IMAGE_NEW_SIZE,char_count_per_alphabet=5)
  
  
  print('ok %s'%write_file)
  
  
def main(unused_argv):
  logging.basicConfig(level=logging.INFO)
  
  preprocess_omniglot(IMAGE_NEW_SIZE = 28)
#  preprocess_omniglot(IMAGE_NEW_SIZE = 64)
#  get_data()


if __name__ == '__main__':
  tf.app.run()
