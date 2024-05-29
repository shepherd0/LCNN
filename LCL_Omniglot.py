 
import numpy as np
import tensorflow as tf
import argparse
import os
import sys
import glob
from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.utils.gpu import get_nr_gpu

from tensorflow.contrib.layers import variance_scaling_initializer
from tensorflow.contrib.layers import xavier_initializer_conv2d
from tensorflow.contrib.layers import xavier_initializer
import datetime 
from datasets.omniglot09 import Omniglot,Omniglot_oneshot

class Model(ModelDesc):

    def __init__(self, n,steps_per_epoch=1,is_test=False):
        super(Model, self).__init__()
        self.n = n
        self.steps_per_epoch = steps_per_epoch
        self.update_vars = []
        self.initialized = False
        self.isfirst_encode1 = {}
        self.isfirst_encode2 = True
        self.isfirst_logits = True
        
        self.episode_length=appconfig['episode_length'] 
        self.episode_width=appconfig['episode_width']
        self.shot_count=appconfig['shot_count']
        self.encode_size=appconfig['encode_size']
        self.sample_count =int(self.episode_length/self.episode_width)
        self.objects_length = self.episode_width - 1
        self.channel_count=appconfig['channel_count']
        self.is_test = is_test

    def _get_inputs(self):
        return [InputDesc(tf.float32,appconfig['input_shape'], 'input'),
                InputDesc(tf.float32, [None,self.episode_width], 'label')]

    def encode1(self,image):
      ctx = get_current_tower_context()
      if ctx.name not in self.isfirst_encode1:
        self.isfirst_encode1[ctx.name] = True
      reuse= True if(not self.isfirst_encode1[ctx.name]) else None     
      reuse =  tf.AUTO_REUSE

      with tf.variable_scope("encode1",reuse= reuse ) as scope:          
        self.isfirst_encode1[ctx.name] = False
        return self.encode(image)

    def encode(self,image):
        image = tf.transpose(image,[1,2,3,0])
        def residual(name, l, increase_dim=False, first=False):
            shape = l.get_shape().as_list()
            in_channel = shape[3]

            if increase_dim:
                out_channel = in_channel * 2
                stride1 = 2
            else:
                out_channel = in_channel
                stride1 = 1

            with tf.variable_scope(name) as scope:
                
                b1 = l if first else BNReLU(l)
                c1 = Conv2D('conv1', b1, out_channel, stride=stride1, nl=BNReLU)
                c2 = Conv2D('conv2', c1, out_channel)
                if increase_dim:
                    l = AvgPooling('pool', l, 2)
                    l = tf.pad(l, [[0, 0], [0, 0], [0, 0], [in_channel // 2, in_channel // 2]])
                
                l = c2 + l
             
                return l

        with argscope(Conv2D, nl=tf.identity, use_bias=False, kernel_shape=3,
                      W_init=variance_scaling_initializer(mode='FAN_OUT',dtype=tf.float32)):
            l = Conv2D('conv0', image, args1.start_width, nl=BNReLU)
            l = residual('res1.0', l, first=True)
            for k in range(1, self.n):
                l = residual('res1.{}'.format(k), l)

            l = residual('res2.0', l, increase_dim=True)
            l2 = l
            for k in range(1, self.n):
                l = residual('res2.{}'.format(k), l)

            l = residual('res3.0', l, increase_dim=True)
            for k in range(1, self.n):
                l = residual('res3.' + str(k), l)

            l = BNReLU('bnlast', l)
                
            l = GlobalAvgPooling('gap', l)

            logits = l
                  
        return logits
      
    def _build_logits_graph(self, episode_images,seen_images):
      if self.channel_count ==1:
        episode_images1 = [ tf.concat([episode_image,seen_images],axis=0) for episode_image in episode_images ]
      else:  
        seen_imagessplit = tf.split(value=seen_images,num_or_size_splits=self.channel_count,axis=0)
        episode_images1 = []
        for episode_image in episode_images:
          episode_imagesplit = tf.split(value=episode_image,num_or_size_splits=self.channel_count,axis=0)
          
          episode_images2 = [tf.concat([episode_image1,seen_image1],axis=0) for (episode_image1,seen_image1) in  zip(episode_imagesplit,seen_imagessplit)]
          episode_images1.extend(episode_images2)

      
      episode_encodes1 = [self.encode1(episode_image) for episode_image in episode_images1 ]


      if self.channel_count > 1:
        episode_encodes1= [episode_encodes1[(i*self.channel_count):((i+1)*self.channel_count)] for i in range(self.episode_width)]
        episode_encodes1 = [tf.concat(episode_encode,axis=1) for episode_encode in episode_encodes1]


      episode_encodes = tf.stack(episode_encodes1) 
      episode_encodes = tf.transpose(episode_encodes,[1,0,2])
#      episode_encodes = tf.Print(
#          input_=episode_encodes,
#          data=[episode_encodes],
#          message=None,
#          first_n=-1,
#          summarize=100000,
#          name=None
#      )      
      episode_encodes = tf.reshape(episode_encodes,[-1,episode_encodes.get_shape()[1].value*episode_encodes.get_shape()[2].value])

      logits = FullyConnected('linear', episode_encodes, out_dim=self.episode_width, nl=tf.identity,
                              W_init=variance_scaling_initializer(mode='FAN_OUT',dtype=tf.float32))

      return logits

      
    def _build_graph(self, inputs):
      
      images,label = inputs
      images = tf.transpose(images,[3,0,1,2])
       
      episode_images = tf.slice(images,[0,0,0,0],[self.episode_length*self.channel_count,-1,-1,-1])
      seen_images = tf.slice(images,[self.episode_length*self.channel_count,0,0,0],[self.shot_count*self.channel_count,-1,-1,-1])
      
#      episode_images = tf.unstack(episode_images,axis=0)
      episode_images = tf.split(value=episode_images,num_or_size_splits=self.episode_length,axis=0)
      seen_images = tf.split(value=seen_images,num_or_size_splits=self.shot_count,axis=0)
       
      logits = []
      for seen_image in seen_images:
        with tf.variable_scope("graph",reuse= True if(not self.isfirst_logits) else None ) as scope:          
          self.isfirst_logits = False
          logit = self._build_logits_graph(episode_images,seen_image)
          logits.append(logit)
      
      logits = tf.add_n(logits)
      
      sigmoid_logits = tf.sigmoid(logits,name='sigmoid_logits')
      
      prob = logits
      
      cost = tf.nn.sigmoid_cross_entropy_with_logits(
          _sentinel=None,
          labels=label,
          logits=logits,
      )
      
      cost = tf.reduce_mean(cost, name='cross_entropy_loss')

      
      _,max_indices = tf.nn.top_k(prob, self.objects_length )
  
      idx = tf.where(tf.not_equal(max_indices, -1))
      idx1 = tf.slice(input_=idx,begin=[0,0],size=[-1,1])
      colvalue =  tf.gather_nd(max_indices, idx)
      maxidx1 = tf.stack(values=[tf.cast(idx1,tf.int32),tf.reshape(colvalue,[-1,1])],axis=1)
      maxidx = tf.squeeze(maxidx1)
      batch_objects_count =  self.objects_length*BATCH_SIZE
      prodicts = tf.scatter_nd(maxidx,tf.constant(1,shape=(batch_objects_count,)) ,[BATCH_SIZE,self.episode_width],'prodicts')
      
#        prodicts = tf.Print(
#            input_=prodicts,
#            data=[maxidx,label,prodicts,prob],
#            message=None,
#            first_n=-1,
#            summarize=100,
#            name=None
#        )
      
      wrong = tf.logical_xor(tf.cast(label,tf.bool),tf.cast(prodicts,tf.bool))
      wrong = tf.cast(wrong,dtype=tf.float32)
      batch_wrong = tf.reduce_sum(wrong,axis=1)

      batch_wrong = tf.cast(tf.greater(batch_wrong,0),dtype=tf.float32,name='incorrect_vector')
      
#      batch_wrong = tf.Print(
#          input_=batch_wrong,
#          data=[batch_wrong],
#          message=None,
#          first_n=-1,
#          summarize=1000000,
#          name=None
#      )
    
      train_error = tf.reduce_mean(batch_wrong,name='train_error')
      add_moving_summary(train_error)

      # weight decay on all W of fc layers
      wd_w = tf.train.exponential_decay(0.0002, get_global_step_var(),
                                        self.steps_per_epoch * appconfig['weight_decay_epoch'], 0.2, True)
      wd_cost = tf.multiply(wd_w, regularize_cost('.*/W', tf.nn.l2_loss)+regularize_cost('.*/linear/b', tf.nn.l2_loss), name='wd_cost')
      add_moving_summary(cost, wd_cost)

      add_param_summary(('.*/W', ['histogram']))   # monitor W
      add_param_summary(('.*/linear/b', ['histogram']))   # monitor W
      
      self.cost = tf.add_n([cost, wd_cost], name='cost')
      
      self.initialized  = True
      
    def _get_optimizer(self):
        lr = get_scalar_var('learning_rate',args1.start_learning_rate, summary=True)
        opt = tf.train.MomentumOptimizer(lr, 0.9)
        return opt

    def cosine_similarity(self,x, y, eps=1e-6):
        normed_x = tf.nn.l2_normalize(x, dim=2)
        normed_y = tf.nn.l2_normalize(y, dim=2)
        cosine_similarity = tf.matmul(normed_x,normed_y,transpose_b=True)
        
        return cosine_similarity 
      
def get_data_Omniglot01(train_or_test,data_type='train',images_count=20):
    isTrain = train_or_test == 'train'
            
    ds = Omniglot(data_type,episode_length=appconfig['episode_length'],
                  episode_width=appconfig['episode_width'],
                  shot_count=appconfig['shot_count'],
                  bach_size=BATCH_SIZE,
                  train_size_times=appconfig['train_size_times'],
                  images_count=images_count,)
    
    ds = BatchData(ds, BATCH_SIZE,(not isTrain))
    return ds
  

def get_data_Omniglot_oneshot(data_type):
    episode_item_file = None
#    
    ds = Omniglot_oneshot(data_type,BATCH_SIZE,appconfig['episode_item_file'],appconfig['is_test'])
    augmentors = []
    
    ds = BatchData(ds, BATCH_SIZE,True)

    return ds,ds

  
  
def set_imagesize(width,height,channel_count=1):
    appconfig['image_height'] = height
    appconfig['image_width'] = width
    appconfig['channel_count'] = channel_count
    appconfig['input_shape']= [None, appconfig['image_height'], appconfig['image_width'],
          (appconfig['episode_length']+appconfig['shot_count'])*appconfig['channel_count']]  
  
def get_data(train_or_test):
  
    if train_or_test =='train':
      if appconfig['train_dataset']=='omniglot':
        dataset_train = get_data_Omniglot01('train','train')

      elif appconfig['train_dataset']=='omniglot_tiny1':
        dataset_train = get_data_Omniglot01('train','tiny128',images_count=20)
      elif appconfig['train_dataset']=='omniglot_tiny1_5':
        dataset_train = get_data_Omniglot01('train','tiny128',images_count=5)        

      elif appconfig['train_dataset']=='omniglot_tiny2':
        dataset_train = get_data_Omniglot01('train','tiny228',images_count=20)
      elif appconfig['train_dataset']=='omniglot_tiny2_5':
        dataset_train = get_data_Omniglot01('train','tiny228',images_count=5)  
      elif appconfig['train_dataset']=='omniglot_tiny2_10':
        dataset_train = get_data_Omniglot01('train','tiny228',images_count=10)  
        
        
      return dataset_train
      
    if train_or_test =='test':
      if appconfig['test_dataset']=='omniglot':
        dataset_test =get_data_Omniglot01('test','test')
      elif appconfig['test_dataset']=='omniglot_oneshot':
        dataset_test,_ = get_data_Omniglot_oneshot('oneshot')
        
      return dataset_test
      
evaluator =None
omniglot_oneshot_data_set = None

def get_log_dir():
    runfiles = __file__.replace('\\','/').split('/')
    log_dir = r'tmp/train/'+args1.run_tag+'/Omniglot_%s'%(runfiles[-1].split('.')[0])

    
    return log_dir
  
def get_config():
    
    log_dir = get_log_dir()
    logger.set_logger_dir(log_dir, action='b')
    
    
    dataset_train = get_data('train')
    dataset_test = get_data('test')  
    
    print('nr_tower:%d'%appconfig['nr_tower'])
    
    steps_per_epoch =64
    
    
    model=Model(n=args1.num_units,steps_per_epoch = steps_per_epoch)
    
    callbacks=[
        ModelSaver(max_to_keep=200,keep_checkpoint_every_n_hours=10),
        InferenceRunner(dataset_test,[ScalarStats('cost'),ScalarStats('cross_entropy_loss'), ClassificationError(summary_name='val_error')]),
        ScheduledHyperParamSetter('learning_rate',
                                  [(0,args1.start_learning_rate), (args1.drop_1, 0.01), (args1.drop_2, 0.001), (args1.drop_3, 0.0002)]),
    ]
      
    return TrainConfig(
        dataflow=dataset_train,
        callbacks=callbacks,
        model=model,
        max_epoch=args1.max_epoch,
        starting_epoch=appconfig['starting_epoch'],
        steps_per_epoch=steps_per_epoch,
        
    )
      

appconfig={
    'train_dataset':'omniglot',
    'test_dataset':'omniglot',
    'image_width':28,
    'image_height':28,
    'memory':True,
    'batch_size':100,
    'episode_length':20, 
    'episode_width':20,
    'shot_count':1,
    'encode_size':32,
    'weight_decay_epoch':400000,
    'train_type':'train',
    'train_size_times':1,
    'episode_item_file':None,
    'test_epoches':1,
    'is_test':False,
    'starting_epoch':1,
    'channel_count':1,
    'predicts_path':None,
    'test_times':1,
    'withnull':False,     
    }
appconfig['input_shape']= [None, appconfig['image_height'], appconfig['image_width'],
          appconfig['episode_length']+appconfig['shot_count']]
args1=None

def parse_args(load=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='train' ,help='task is in:train,test')
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('-n', '--num_units',
                        help='number of units in each stage',
                        type=int, default=5)
    parser.add_argument('--start_learning_rate',default=0.1,type=float, help='start learning_rate')    
    parser.add_argument('--drop_1',default=100,type=int, help='Epoch to drop learning rate to 0.01.') # nargs1='*' in multi mode
    parser.add_argument('--drop_2',default=200,type=int,help='Epoch to drop learning rate to 0.001')
    parser.add_argument('--drop_3',default=300,type=int,help='Epoch to drop learning rate to 0.002')
    parser.add_argument('--load', help='load model',default=load)
    parser.add_argument('--run_tag', help='load model',default='09')
    parser.add_argument('--max_epoch',default=300,type=int,help='max epoch')
    parser.add_argument('--cols',default=2,type=int,help='nums of column')
    parser.add_argument('--start_width',default=16,type=int,help=' ')
    parser.add_argument('--growth_width',default=3,type=int,help=' ')
    parser.add_argument('--debug',help=' ')
    parser.add_argument('--memo',help='memo')
    parser.add_argument('--batch_size',type=int,default=appconfig['batch_size'],help='batch_size')
    parser.add_argument('--train_dataset',default=appconfig['train_dataset'], help='train_dataset must be in: omniglot,omniglot_oneshot_train,omniglot_small1,omniglot_small2')
    parser.add_argument('--test_dataset',default=appconfig['test_dataset'], help='test_dataset must be in: omniglot,omniglot_oneshot,nist19sd')
    parser.add_argument('--test_epoches',default=appconfig['test_epoches'],type=float,help='test_epoches')

    parser.add_argument('--episode_length',default=appconfig['episode_length'],type=int,help='episode_length')
    parser.add_argument('--episode_width',default=appconfig['episode_width'],type=int,help='episode_width')
    parser.add_argument('--shot_count',default=appconfig['shot_count'],type=int,help='shot_count')
    parser.add_argument('--starting_epoch',default=appconfig['starting_epoch'],type=int,help='starting_epoch')
    parser.add_argument('--predicts_path',default=appconfig['predicts_path'],help='predicts_path')
    parser.add_argument('--test_times',default=appconfig['test_times'],type=int,help='test_times')
    
    global args1 
    args1 = parser.parse_args()
    
    appconfig['batch_size'] =args1.batch_size
    global BATCH_SIZE
    BATCH_SIZE = appconfig['batch_size']
    
    if args1.gpu:
      appconfig['nr_tower'] = len(args1.gpu.split(','))
      if args1.task =='train':
        BATCH_SIZE = BATCH_SIZE // appconfig['nr_tower']
    else:
      appconfig['nr_tower'] =1
    
    appconfig['train_dataset'] = args1.train_dataset
    appconfig['test_dataset'] = args1.test_dataset
    appconfig['test_epoches'] = args1.test_epoches
    
    appconfig['episode_length'] = args1.episode_length
    appconfig['episode_width'] = args1.episode_width
    appconfig['shot_count'] = args1.shot_count
    appconfig['starting_epoch'] = args1.starting_epoch
    appconfig['input_shape']= [None, appconfig['image_height'], appconfig['image_width'],
              appconfig['episode_length']+appconfig['shot_count']]
    appconfig['predicts_path'] = args1.predicts_path
    appconfig['test_times'] = args1.test_times
    
    
    return args1

def run():
    logger.info("\n".join(['%s=%s'%(key, value) for key, value in appconfig.items()]))
    if args1.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args1.gpu
#    os.environ["TENSORPACK_DATASET"]=r'/home/xcy/tmp/datasets'

    config = get_config()
    logger.info("\n".join(['%s=%s'%(key, value) for key, value in appconfig.items()]))
    if args1.load:
        config.session_init = SaverRestore(args1.load)
    if args1.gpu:
        config.nr_tower = appconfig['nr_tower']

    nr_gpu = max(get_nr_gpu(), 1)

    launch_train_with_config(config, SyncMultiGPUTrainerParameterServer(nr_gpu,ps_device='cpu'))
    
    

    return values

def test1():

    file_names=glob.glob(os.path.join(args1.load,'model-*.index'))
    file_names = [file_name.rstrip('.index') for file_name in file_names]
    file_names = ';'.join(file_names)
    args1.load = file_names
    test()


def test():
    MATPLOTLIB_AVAIBLABLE = False
    from scipy.misc import imsave
    try:
        import matplotlib
        from matplotlib import offsetbox
        import matplotlib.pyplot as plt
        import json
        MATPLOTLIB_AVAIBLABLE = True
    except ImportError:
        MATPLOTLIB_AVAIBLABLE = False  
    
    if not MATPLOTLIB_AVAIBLABLE:
        logger.error("visualize requires matplotlib package ...")
        return

    if args1.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args1.gpu
    
    if args1.load == None:
      args1.load =  get_log_dir()
      
    logger.info(";".join(['%s=%s'%(key, value) for key, value in appconfig.items()]))
        
    def save_predicts(offset,images, label,prodicts,path):
      images=np.transpose(images,[0,3,1,2])
      for episodeindex,episode in enumerate(images):
         iserror = 0
         for imageindex,image in enumerate(episode):
           imagefile=str(offset).zfill(3)+'_'+str(episodeindex).zfill(3)+'_'+str(imageindex).zfill(3)+'_'
           imagefile +='p_'
           if imageindex < label.shape[1]:
             iserror +=abs(label[episodeindex][imageindex]-prodicts[episodeindex][imageindex])
             imagefile +=str(label[episodeindex][imageindex])+'_'+str(prodicts[episodeindex][imageindex])
           else:
             if iserror == 0:
               imagefile +='2_2'
             else:
               imagefile +='3_3'
           imagefile +='.png'

           imagefile=os.path.join(path,imagefile)
           imsave(imagefile,image)
           
    model_paths = args1.load.split(';')
    model_path = model_paths[0]
    
    fileName = os.path.basename(model_path)
    min_stat = None
    if not fileName.startswith('model'):
      stats = json.load(open(os.path.join(model_path,'stat.json')))
      
      min_val_error = sys.maxsize
      
      for stat in stats:
        if stat['val_error'] <= min_val_error:
          min_val_error = stat['val_error']
          min_stat = stat
          
      if min_stat != None:
        model_path = os.path.join(model_path,'model-'+str(min_stat['global_step']))
        logger.info('model_path:%s'%model_path)
        logger.info(min_stat)
      model_paths = [model_path]
        
    model_pathms = {}
    for model_path in model_paths:
      steps = int(model_path.partition('model-')[2])
      model_pathms[steps] = model_path
    model_pathms= sorted(model_pathms.items(), key=lambda d:-d[0])
    
    dataset_test = get_data('test')  
    
    test_results ={}
    for steps,model_path in model_pathms:
      logger.info(model_path)
        
      model=Model(n=args1.num_units)
      predictConfig = PredictConfig(
          session_init=SaverRestore(model_path),
          model=model,
          input_names=['input','label'],
          output_names=['train_error','prodicts','sigmoid_logits'])
      pred = OfflinePredictor(predictConfig)
      
      test_result = {}
      test_result['accs']=[]
#      avg_errors = []
      dataset_test.reset_state()
      for t in range(appconfig['test_times']):
        errors = []
        predicts_path = None
        if appconfig['predicts_path']:
          predicts_path = os.path.join(appconfig['predicts_path'],datetime.datetime.now().now().strftime("%Y%m%d%H%M%S"))
          os.makedirs(predicts_path)
          logger.info('predicts images output path:%s'%(predicts_path))        
          
        for offset, dp in enumerate(dataset_test.get_data()):
            digit, label = dp
            prediction = pred([digit, label])
            error = (1-float(prediction[0]))*100
            prodicts = prediction[1]
            if predicts_path:
              save_predicts(offset,digit,label,prodicts,predicts_path)
            errors.append(error)
            logger.info("%d is %f"%(offset,error))
        avg_error = np.average(errors)
        logger.info('time:%d,%f'%(t,avg_error))
        test_result['accs'].append(avg_error)
      test_result['avg_acc'] = np.average(test_result['accs'])
      test_result['max_acc'] = np.max(test_result['accs'])
      test_result['min_acc'] = np.min(test_result['accs'])
      test_result['var_acc'] = np.var(test_result['accs'])
      if predicts_path:
        test_result['predicts_path'] = predicts_path
      if min_stat:
        test_result['min_stat'] = min_stat
        
      logger.info('avg_acc:%f,max_acc:%f,min_acc:%f,var_acc:%f'%( test_result['avg_acc'],test_result['max_acc'],
                                              test_result['min_acc'],test_result['var_acc'])) 
      test_results[steps] = test_result
      
    test_results1 ={}
    testfile=os.path.join(os.path.dirname(model_path),'test.json')
    if os.path.exists(testfile):
      test_results1 = json.load(open(testfile,'r'))
    test_id = '%s_S%d'%(appconfig['test_dataset'],appconfig['shot_count'])
    test_results1[test_id] = test_results
    json.dump(test_results1,open(testfile,'w'),indent=2,sort_keys=True)
    print('ok,results,write to file:%s'%testfile)
      
    return test_results
  
def prepare_data():
  import datasets.data_utils as data_utils
  
  data_utils.main(None)
  
  import datasets.data_utils_oneshot as data_utils_oneshot
  
  data_utils_oneshot.main(None)
  
  
if __name__ == '__main__':
  parse_args()
  
  if args1.task == 'train':
    run()
    # --task=train --train_dataset=omniglot_tiny2_5 --test_dataset=omniglot_oneshot  -n=1 --memo=101--test_epoches=64--batch_size=40  --drop_1=700 --drop_2=800 --drop_3=900 --max_epoch=900   
    
  elif args1.task == 'test':
    test()
  elif args1.task == 'prepare_data':
    # --task=prepare_data 
    prepare_data()
            