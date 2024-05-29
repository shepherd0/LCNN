# -*- coding: utf-8 -*-
"""
Created on Thu May 18 17:21:13 2017

@author: DDD
"""
from tensorpack import *
import numpy as np
from scipy import misc
from  datasets import data_utils
from  datasets import data_utils_oneshot 
import random
import pickle as pickle
import gzip
import os
import math
from itertools import chain
from collections import defaultdict


class Omniglot(RNGDataFlow):
    """
    Produces [image, label] in Omniglot dataset,
    image is 105x105 in the range [0,1], label is an int.
    """

    def __init__(self, data_type,episode_length=20,episode_width=20,
                 shot_count=1,bach_size=100,
                 train_size_times=1,images_count=20):
        """
        Args:
            train_or_test (str): either 'train' or 'test'
            shuffle (bool): shuffle the dataset
        """

#        assert train_or_test in ['train', 'test']
#        self.train_or_test = train_or_test
        
        self.episode_length=episode_length
        self.episode_width=episode_width
        self.shot_count=shot_count
        
        self.data = data_utils.get_data(data_type,images_count)
        
        print('number of sample %s classes is %d with rotation augmention (90,180,270)'% (data_type,len(self.data)))
        print('number of sample each class:')
        print([len(v) for v in self.data.values()])
#        self.data = get_range(self.data,0,100)
        self._size =(len(self.data)*images_count) // (self.episode_length+self.shot_count)
        self._size = (self._size // bach_size) * bach_size*train_size_times
        if self._size<bach_size:
          self._size = bach_size
        self.reset_state()
      
    
    def size(self):
        return self._size
#        return 20

    def get_data(self):

        for _ in range(self.size()):
          episode_x,episode_y = self.sample_episode_batch(self.data,
                          episode_length=self.episode_length,
                          episode_width=self.episode_width,shot_count=self.shot_count)
          yield [episode_x, episode_y]
            
    def get_per_pixel_mean(self):
        """
        return a mean image of all (train and test) images of size 32x32x3
        """
        mean = np.mean(self.images, axis=0)
        return mean
    def sample_episode_batch(self, data,episode_length,
                             episode_width, shot_count=1):
      """Generates a random batch for training or validation.
  
      Structures each element of the batch as an 'episode'.
      Each episode contains episode_length examples and
      episode_width distinct labels.
  
      Args:
        data: A dictionary mapping label to list of examples.
        episode_length: Number of examples in each episode.
        episode_width: Distinct number of labels in each episode.
        batch_size: Batch size (number of episodes).
  
      Returns:
        A tuple (x, y) where x is a list of batches of examples
        with size episode_length and y is a list of batches of labels.
      """
#      print('data length is %d'%len(data))
      assert len(data) >= episode_width
      assert episode_length % episode_width  == 0
      sample_count =  episode_length // episode_width 
      
      keys = data.keys()
      
      episode_labels = random.sample(keys, episode_width)
      seen_label = episode_labels[0]
      episode_labels = episode_labels[1:]
      episode_x = [random.sample(data[lab], sample_count) for lab  in episode_labels]
      sample_count =len(episode_x[0])
      seen_x = random.sample(data[seen_label],sample_count+shot_count)
      
      episode_x1 =[]
      for x in episode_x:
        episode_x1.extend(x)
      
      episode_y = [1]*(episode_width-1)
      episode_y.extend([0]*1)
      
      episode_x1.extend(seen_x)
      
      episode_x = np.asarray(episode_x1)
      episode_y =np.asarray(episode_y)
      
#      episode_labels.extend([seen_label])
#      episode_labels = np.repeat(episode_labels,sample_count)
      
      idxs_y = list(range(episode_width))
      self.rng.shuffle(idxs_y)
      idxs_x1 =[list(range(idx*sample_count,(idx+1)*sample_count)) for idx in idxs_y ]
#      print(idxs_x1)
      idxs_x = list(chain(*idxs_x1))
#      print(idxs_x)
      episode_x[0:episode_length] = episode_x[0:episode_length][idxs_x]
      episode_y = episode_y[idxs_y]
#      episode_labels =episode_labels[idxs]
      
#      episode_x = 255.0 -episode_x
      
      episode_x =np.transpose(episode_x,[1,2,0])
      
      return (episode_x,episode_y)

    
class Omniglot_oneshot(RNGDataFlow):
    """
    Produces [image, label] in Omniglot dataset,
    image is 105x105 in the range [0,1], label is an int.
    """

    def __init__(self,data_type='oneshot',batch_size=100,episode_item_file=None,is_test=False):
        """
        Args:
            train_or_test (str): either 'train' or 'test'
            shuffle (bool): shuffle the dataset
        """
        self.is_test = is_test
        
        self.batch_size = batch_size
        
        self.episode_item_file = episode_item_file
        
        all_runs = data_utils_oneshot.get_data(data_type)

        
        self.data = all_runs
        
        self.reset_state()
        
        if is_test:
          if self.episode_item_file == None:
            self.episode_item_ids = []
          else:
            with gzip.open(self.episode_item_file, 'rb') as f:  
              output = pickle.load(f)
              self.episode_item_ids = output['episode_item_ids']
            
        self.batch_start = 0
        
        self.predicts =[]
        
        
        
    def size(self):
        return 400
#        return 20
        
    def get_data(self):

      for run_id in self.data.keys():
        for test_id in self.data[run_id]['test'].keys():
          test_ids=test_id.split('_') 
#            item_id = test_ids[0]
          class_id = test_ids[1]
          episode_x,episode_y,item_ids1 = self.sample_episode_batch(run_id,test_id,class_id)
          
          if self.is_test:
            self.episode_item_ids.append(item_ids1)
          
          yield [episode_x, episode_y]
            
    def sample_episode_batch(self,run_id,test_id,class_id):
        
        episode_x=[]
        episode_y=[]
        item_ids=[]
        for key in self.data[run_id]['training'].keys():
          item_ids.append(run_id+'/test/'+key)
          if key==class_id:
            episode_y.append(0)
          else:
            episode_y.append(1)
          episode_x.append(self.data[run_id]['training'][key])
        episode_x.append(self.data[run_id]['test'][test_id])
        item_ids.append(run_id+'/test/'+test_id)
        episode_x1 = np.asarray(episode_x)
        episode_y1 =np.asarray(episode_y)
        item_ids1 = np.asarray(item_ids)
         
        idxs = list(range(20))
        self.rng.shuffle(idxs)
        episode_x1[0:20] = episode_x1[0:20][idxs]
        item_ids1[0:20] = item_ids1[0:20][idxs]
        episode_y1 = episode_y1[idxs]
    
  #      episode_x = 255.0 -episode_x
        
        episode_x1 =np.transpose(episode_x1,[1,2,0])
        
        return (episode_x1,episode_y1,item_ids1)
      
      
    def get_per_pixel_mean(self):
        """
        return a mean image of all (train and test) images of size 32x32x3
        """
        pass  
  