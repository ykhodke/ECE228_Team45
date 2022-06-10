from matplotlib import transforms
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pydicom as dicom
from PIL import Image
import re
import os
import glob
import shutil, sys
import torchvision
import torch

class_dict = {
  "MALIGNANT"               : 0,
  "BENIGN"                  : 1,
  "BENIGN_WITHOUT_CALLBACK" : 2
}

dataset_dir   = '/home/ykhodke/ECE228/project_final/dataset/images/'
data_png_dir = '/home/ykhodke/ECE228/project_final/dataset/images_rpt/'

test_csv = '/home/ykhodke/ECE228/project_final/dataset/csv/mass_case_description_test_set.csv'
train_csv  = '/home/ykhodke/ECE228/project_final/dataset/csv/mass_case_description_train_set.csv'

def generate_pngs_from_dicom(splt ,data_csv):

  df = pd.read_csv(data_csv)

  df.columns = [c.replace(' ', '_') for c in df.columns]

  print(df.columns)


  np_cropped_image_path_list = df.image_file_path.apply(lambda x: dataset_dir+x).to_numpy()
  np_roimask_image_path_list = df.ROI_mask_file_path.apply(lambda x: dataset_dir+x).to_numpy()
  cropped_image_classification = df.pathology.apply(lambda x: class_dict[x]).to_numpy()

  j = 0
  k = 0

  for i, fp in enumerate(np_cropped_image_path_list):  
    #checking for the existence if the image in the dataset downloaded >> apparently the file naming is different from the csv
    #so extract the directory path
    path = re.search('(.*\/)', fp).group(0)

    #skip all computations below if the directory doesn't exist
    if( not(os.path.exists(path)) ):
      k += 1
      continue

    #see what the directory containts are
    dcm_images = glob.glob(path+'*.dcm')

    for indx, image in enumerate(dcm_images):
      ds = dicom.dcmread(image)
      ti = Image.fromarray(ds.pixel_array)
      ti.save('{}_pid_{}_path_{}_{}.png'.format(splt, i, cropped_image_classification[i], indx))
      j += 1
      break
    break
  
  for i, fp in enumerate(np_roimask_image_path_list):  
    #checking for the existence if the image in the dataset downloaded >> apparently the file naming is different from the csv
    #so extract the directory path
    path = re.search('(.*\/)', fp).group(0)

    #skip all computations below if the directory doesn't exist
    if( not(os.path.exists(path)) ):
      k += 1
      continue

    #see what the directory containts are
    dcm_images = glob.glob(path+'*.dcm')

    for indx, image in enumerate(dcm_images):
      ds = dicom.dcmread(image)
      ti = Image.fromarray(ds.pixel_array)
      ti.save('roi_{}_pid_{}_path_{}_{}.png'.format(splt, i, cropped_image_classification[i], indx))
      j += 1
    break

  print("images_convertetd {}, lost {}".format(j, k))

  
if __name__ == "__main__":

  generate_pngs_from_dicom('train', train_csv)