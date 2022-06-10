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
data_png_dir = '/home/ykhodke/ECE228/project_final/dataset/images_png/'

test_csv = '/home/ykhodke/ECE228/project_final/dataset/csv/mass_case_description_test_set.csv'
train_csv  = '/home/ykhodke/ECE228/project_final/dataset/csv/mass_case_description_train_set.csv'

def generate_pngs_from_dicom(splt ,data_csv):

  df = pd.read_csv(data_csv)

  df.columns = [c.replace(' ', '_') for c in df.columns]


  np_cropped_image_path_list = df.image_file_path.apply(lambda x: dataset_dir+x).to_numpy()
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
      ti.save('dataset/images_png/{}_pid_{}_path_{}_{}.png'.format(splt, i, cropped_image_classification[i], indx))
      j += 1


  print("images_convertetd {}, lost {}".format(j, k))


def resize_data(png_dir):
  #analyze the size of the training and test images
  w = np.array([])
  h = np.array([])
  apr = np.array([])

  png_files = png_dir+'/*.png'
  images_path = glob.glob(png_files)
  for pth in images_path:
    img = Image.open(pth)
    w = np.append(w, img.width)
    h = np.append(h, img.height)
    apr = np.append(apr, img.width/img.height)


  fig, (ax1, ax2, ax3) = plt.subplots(1,3)
  ax1.hist(w)
  ax2.hist(h)
  ax3.hist(apr)
  plt.savefig('dim_distr.png')

  print(np.mean(w), np.std(w), np.mean(h), np.std(h), np.mean(apr), np.std(apr))
  #for cropped images: 371.1668632075472 132.00738298060054; 368.29009433962267 129.88504821931775; 1.022080115448999  0.1733865877826163
  #for full images   : 3149.212853773585 551.5515188306557 ; 5293.366155660377  674.8200168925788 ; 0.5957011709560329 0.07587728048683441


  #naively resizing to 370px * 370px
  # for pth in images_path:
  #   img = Image.open(pth)
  #   img_resize = img.resize((370, 370))
  #   img_resize.save(png_dir+'/resize/'+os.path.basename(pth))

def normalize_image(png_dir, png_nrml_dir):
  png_files = png_dir+'/*.png'
  images_path = glob.glob(png_files)

  normalized_png_dir = '/home/ykhodke/ECE228/project_final/dataset/images_png_nrmlzd/'

  transform = torchvision.transforms.CenterCrop(1600)
  tnsr_2_PILimg = torchvision.transforms.ToPILImage()

  i = 0

  for pth in images_path:
    print(i)
    img = Image.open(pth)
    img = img.resize((370, 370))
    img.save(png_nrml_dir+os.path.basename(pth))
    i = i + 1

if __name__ == "__main__":

  # generate_pngs_from_dicom('train', train_csv)
  # generate_pngs_from_dicom('test', test_csv)

  # resize_data('dataset/images_png')

  normalize_image('dataset/images_png', '/home/ykhodke/ECE228/project_final/dataset/images_png_nrmlzd/')


