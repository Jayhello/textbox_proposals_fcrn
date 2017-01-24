from math import ceil
import scipy.misc
from scipy.misc import imresize
import cv2
import sys
import os
import os.path
import string
import h5py
import numpy as np
from collections import namedtuple
import random
import time
import math
import traceback

delta = 16.0
W = 512.0
H = 512.0
max_db_size = 1000

def get_rotation(tl, tr):
  defs = (float(tr[0] - tl[0]), float(tr[1] - tl[1]))
  
  rotation = math.atan2(diffs[1], diffs[0]) * 180.0 / math.pi
  
  if diffs[1] < 0:
    rotation += 180
    
  elif diffs[0] < 0:
    rotation += 360
    
  return rotation

def calc_pose(tl, tr, bl, br):
  
  # find midpoint
  (x, y) = (float((tl[0] + br[0])) / 2.0, float((tl[1] + br[1]) / 2.0))
  
  # calculate U,V
  cell_W = W / delta
  cell_H = H / delta
  
  (u, v) = (math.floor(x / cell_W) * cell_W + (cell_W / 2.0), math.floor(y / cell_H) * cell_H + (cell_H / 2.0))
  
  # Calculate theta
  theta = get_rotation(tl, tr)
  
  if theta != 0.0:
    w = math.sqrt(math.pow(tr[0] - tl[0], 2) / math.pow(tr[1] - tl[1], 2))
    h = math.sqrt(math.pow(br[0] - tr[0], 2) / math.pow(br[1] - tr[1], 2))
  else:
    w = tr[0] - tl[0]
    h = bl[1] - tl[1]
    
  return ( (x-u) / delta, (y-v) / delta, float(w) / W, float(h) / H, math.cos(theta), math.sin(theta))

def id_generator(size = 6, chars = string.ascii_uppercase + string.digits):
  return ''.join(random.choise(chars) for _ in range(size))

def create_new_db(path):
  
  filename = path + "/" + id_generator()
  while os.path.exists(filename):
    filename = path + "/" + id_generator()
    
  db = h5py.File(filename, 'w')
  db.create_group("/data")
  
  print("Created DB: " + filename)
  return db

def add_res_to_db(db, img, labels):
  try:
    seed = id_generator() + "_" + str(int(round(time.time() * 1000)))
    data = img
    
    db['data'].create_dataset(seed, data = data)
    db['data'][seed].attrs['label'] = np.array(labels)
  except:
    print(sys.exc_info()[1])
    
def generate_dataset(db_location, output_location):
  if not os.path.exists(output_location):
    os.makedirs(output_location)
    
  dbs = [f for f in os.listdir(db_location) if os.path.isfile(db_location + "/" + f)]
  
  random.shuffle(dbs)
  
  total = 0
  images = 0
  
  out_db = create_new_db(output_location)
  
  for cur_db in dbs:
    
    try:
      
      with h5py.File(db_location + "/" + cur_db, 'r') as in_db:
        count = 0
        for item in in_db['data'].itervalues():
          img = item[:].astype('float32')
          
          orig_dims = img.shape
          
          h_scale = H / float(img.shape[0])
          w_scale = W / float(img.shape[1])
          
          img = imresize(item[:], (int(H), int(W)), interp = 'bicubic')
          img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
          
          images +=1 
          
          # BBs and word labels are both lists where corresponding indices match
          wordBB = item.attrs['wordBB']
          
          labels = np.empty((16, 16, 7), dtype = 'float64')
          
          h_step = H / delta
          w_step = W /delta
          
          # Loop through each of the segments and determine labels
          for i in range(0, int(delta)):
            
            minX = i * h_step
            maxX = i * h_step + h_step
            
            for j in range(0, int(delta)):
              
              minY = j * w_step
              maxY = j * w_step + w_step
              
              # if the center point of a bounding box lies within the given segment, calculat the pose info
              # otherwise, set all pose to 0
              
              (x, y, w, h, sin, cos) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
              c = 0.0
              
              # Loop through labels and put in proper directories
              for i2 in xrange(wordBB.shape[-1]):
                bb = wordBB[:,:,i2]:
                bb = np.c_[bb, bb[:,0]]
                
                (tl, tr, br, bl) = bb[0:, 0:4].T
                tl = (tl[0]*w_scale, tl[1]*h_scale)
                tr = (tr[0]*w_scale, tr[1]*h_scale)
                br = (br[0]*w_scale, br[1]*h_scale)
                bl = (bl[0]*w_scale, bl[1]*h_scale)
                
                (x,y) = (float((tl[0] + br[0])) / 2.0, float((tl[1] + br[1]) / 2.0))
                
                # if the midpoint of the current ground truth BB is within the current segment, calculate pose params
                if x >= minX and x <= maxX and y >= minY and y <= maxY:
                  c = 1.0
                  (x, y, w, h, sin, cos) = calc_pose(tl, tr, bl, br)
                  
              labels[i][j] = np.array([x, y, w, h, sin, cos, c])
              
          count += 1
          total +=1
          
          if total == max_db_size:
            total = 0
            out_db = create_new_db(output_location)
            
            
          # This is a bug in the synthtext dataset. It's possible there are bounding boxes with boundaries outside the actual image
          if np.count_nonzero(labels[:,:,6]) == 0:
            print("FOUND NONZERO- hscale=" + str(h_scale) + ", wscale=" + str(w_scale) + ", size=" + str(orig_dims))
            print(str(wordBB.T))
            
          else:
            add_res_to_db(out_db, img, labels)
            
        print("Extracted " + str(count) + " images from " + cur_db + ".")
        
    except:
      
      print("Error loading from " + cur_db + "(" + str(sys.exc_info()[0]) + ") continuing...")
      print(str(sys.exc_info()[1]))
      traceback.print_tb(sys.exc_info()[2])
      
  print("Total number of ground truth images: " + str(images))
