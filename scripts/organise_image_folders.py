"""
copy all images into one folder ('../../dataset/images/all') because images are saved in multiple folders right now
"""

import os
import shutil

val_folds = ['M0203', 'M0205', 'M0208', 'M0209', 'M0403', 'M0601', 'M0602', 'M0606', 'M0701', 'M0801', 'M0802',
              'M1001', 'M1004', 'M1007', 'M1009', 'M1101', 'M1301', 'M1302', 'M1303', 'M1401']
old_dir = "/data/lg/UAV-benchmark-M/"
train_dir = "/data/lg/UAVDT/train/images"
val_dir = "/data/lg/UAVDT/val/images"

if not os.path.exists(train_dir):
    os.makedirs(train_dir)

if not os.path.exists(val_dir):
    os.makedirs(val_dir)


folder_names = os.listdir(old_dir)

for folder in folder_names:
    folder_path = old_dir + '/' + folder  # '../../UAV-benchmark-M/M0101'
    img_filename_ls = os.listdir(folder_path)  # 'img000061.jpg'
    if folder in val_folds:
        for img_filename in img_filename_ls:
            # '../../UAV-benchmark-M/M0403/img000061.jpg'
            old_img_path = old_dir + '/' + folder + '/' + img_filename
            # ../../dataset/images/all/M0403_000061.jpg
            output_img_path = val_dir + '/' + folder + '_' + img_filename[-10:]
            # copy images from old path tp new path
            shutil.copyfile(old_img_path, output_img_path)
    else:
        for img_filename in img_filename_ls:
            # '../../UAV-benchmark-M/M0403/img000061.jpg'
            old_img_path = old_dir + '/' + folder + '/' + img_filename
            # ../../dataset/images/all/M0403_000061.jpg
            output_img_path = train_dir + '/' + folder + '_' + img_filename[-10:]
            # copy images from old path tp new path
            shutil.copyfile(old_img_path, output_img_path)

    print('image folder copy finished: ', folder)
print('all images has been copied ')



