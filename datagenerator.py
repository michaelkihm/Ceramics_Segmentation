import numpy as np
import tensorflow as tf
import os
import math
import cv2 as cv
import random

from typing import Tuple, List


class DataGenerator(tf.keras.utils.Sequence):
    """
    Class to read and to preprocess image data for Segmentation Model
    @param image_path path to image data
    @param mask_path path to segmentations masks. \n Should be one channels image where the pixel value represents the class label
    @param classes number of classes 
    @image_size each image and each mask will be scaled to the given size
    @batch_size batch size
    @shuffle shuffle data or not
    @file_types tuple of excepted file types
    """

    def __init__(self, image_path:str,mask_path:str, classes:int = 2, image_size:Tuple=(200, 200), batch_size:int=128, shuffle:bool=True, file_types:tuple=('jpg', 'png')):
        self._image_path = image_path
        self._mask_path = mask_path
        self._image_size = image_size
        self._batch_size = batch_size
        self._shuffle = shuffle
        self._file_types = file_types
        self._classes = classes

        self._images, self._masks = self._create_dataset()
       

    def __len__(self):
        """
        @brief Returns number of batches
        @remark has to be implemented
        """
        return math.ceil(len(self._images) / self._batch_size)

    def __getitem__(self, idx):
        """
        @brief  reads batch of images and preprocesses segmentation masks images.
                Furthermore, scales images. 
        """

        batch_images = self._images[idx * self._batch_size:(idx + 1) * self._batch_size]
        batch_masks =  self._masks[idx * self._batch_size:(idx + 1) * self._batch_size]

        assert len(batch_images) == len(batch_masks)

        images = []
        masks = []

        
        #Read image data
        [images.append(self._read_and_resize_image(file_name)) for file_name in batch_images] 

        #Read mask data
        for mask in batch_masks:
            img = cv.resize(cv.imread(mask,cv.IMREAD_GRAYSCALE),self._image_size)
            mask = np.zeros(shape=(*(img.shape),self._classes))
            for i in range(self._classes):
                mask[:,:,i] = (img == i)
            masks.append(mask)

        assert len(images) == len(masks), "size of image batch doesnt match size of mask list"
        
        return np.array(images).astype(np.float)/255.0, np.array(masks)


    def _create_dataset(self) ->List:
        """
        @brief  lists all files in given image_path and mask_path. Shuffles list of filenames and assures
                that masks belong to corresponding images
        """
        x = ["{p}/{f}".format(p=self._image_path, f=file)
                      for file in os.listdir(self._image_path) if file.lower().endswith(self._file_types)]

        masks = ["{p}/{f}".format(p=self._mask_path, f=file)
                      for file in os.listdir(self._mask_path) if file.lower().endswith(self._file_types)]
        
        assert len(x) == len(masks), "Number of images and masks do not match"

        #sort
        x.sort()
        masks.sort()

        #shuffle
        if self._shuffle:
            random.shuffle(x, lambda: .5)
            random.shuffle(masks, lambda: .5)

        assert len(np.unique([os.path.basename(x[i]) == os.path.basename(masks[i]) for i in range(len(masks))])) == 1 ,"Mask basenames not like image basenames"
            
        return x, masks


    def _read_and_resize_image(self, file_name):
        """
        @brief reads and resizes an image
        """
        return cv.resize(cv.cvtColor(cv.imread(file_name, cv.IMREAD_COLOR),cv.COLOR_BGR2RGB), self._image_size)





    



    



