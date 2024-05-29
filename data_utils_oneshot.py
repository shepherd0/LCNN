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
import tensorflow as tf

import gzip

#MAIN_DIR = r'H:\datasets\omniglot'
MAIN_DIR = ''
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
IMAGE_NEW_SIZE = 28


def get_data(data_type='oneshot'):
  """Get data in form suitable for episodic training.

  Returns:
    Train and test data as dictionaries mapping
    label to list of examples.
  """
#  with tf.gfile.GFile(DATA_FILE_FORMAT % 'train') as f:
  with gzip.open(DATA_FILE_FORMAT % data_type, 'rb') as f:  
    processed_train_data = pickle.load(f)

  return processed_train_data

def get_label_map(fname_label):
  with open(fname_label) as f:
    content = f.read().splitlines()
  pairs = [line.split() for line in content]
  test_files  = [pair[0].split('/')[-1] for pair in pairs]
  train_files = [pair[1].split('/')[-1] for pair in pairs]
  
  label_map={}
  for k,v in zip(test_files,train_files):
    label_map[k] = v
       
  return label_map

def crawl_directory(directory, augment_with_rotations=False,
                    new_width=105, new_height=105):
  """Crawls data directory and returns stuff."""
  images = []
  labels = []
  info = []

  # traverse root directory
  all_runs={}
  label_map={}
  for root, _, files in os.walk(directory):
    logging.info('Reading files from %s', root)
    fileflag = 0
    for file_name in files:
      if file_name.endswith('.txt') :
        label_map = get_label_map(os.path.join(root, file_name))
        continue
      full_file_name = os.path.join(root, file_name)
      print(full_file_name)
      img = imread(full_file_name, flatten=True)

      image = imrotate(img, 0)
      
      image = imresize(image,[new_width, new_height],interp='bilinear',mode=None)
      
      image = (1-image/255.0).astype('float32')
      
      full_file_name_=full_file_name.split(os.path.sep)

      if full_file_name_[-3] not  in all_runs:
        all_runs[full_file_name_[-3]]={}
        
      if full_file_name_[-2] not  in all_runs[full_file_name_[-3]]:
        all_runs[full_file_name_[-3]][full_file_name_[-2]]={}
      
      if full_file_name_[-2]=='test':
        label_id =full_file_name_[-1]+'_'+ label_map[full_file_name_[-1]]
      else:
        label_id = full_file_name_[-1]
        
      all_runs[full_file_name_[-3]][full_file_name_[-2]][label_id]=image
        
#        images.append()
#        labels.append(label_idx + i)
#        info.append(full_file_name)

  return all_runs


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
                    new_width=IMAGE_NEW_SIZE, new_height=IMAGE_NEW_SIZE,
                    first_label=0):
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
  all_runs = crawl_directory(directory,rotate, new_width, new_height,)


  with gzip.open(write_file, 'wb') as f:
    pickle.dump(all_runs, f)



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

  if os.path.exists(TEST_DIR):
    logging.info('It appears that test data has already been unzipped.')
  else:
    logging.info('It appears that test data has not been unzipped.')
    logging.info('Unzipping now.')

    subprocess.check_output('unzip %s.zip -d %s' % (TEST_DIR, DATA_DIR),
                            shell=True)
    

def preprocess_oneshot_omniglot():
  """Download and prepare raw Omniglot data.

  Downloads the data from GitHub if it does not exist.
  Then load the images, augment with rotations if desired.
  Resize the images and write them to a pickle file.
  """

  maybe_download_data()
  IMAGE_NEW_SIZE =28
  
  directory = ONESHOT_DIR
  write_file = DATA_FILE_FORMAT % 'oneshot'
  write_datafiles(directory, write_file, resize=True, rotate=False,
                  new_width=IMAGE_NEW_SIZE, new_height=IMAGE_NEW_SIZE)
  print('ok %s'%write_file)  
  
def main(unused_argv):
  logging.basicConfig(level=logging.INFO)
#  preprocess_omniglot()
  preprocess_oneshot_omniglot()
#  get_data()


if __name__ == '__main__':
  tf.app.run()
