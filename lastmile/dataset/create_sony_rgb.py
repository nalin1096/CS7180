#!/usr/bin/env python
# coding: utf-8

# In[6]:


import rawpy
import numpy as np
import cv2


# In[24]:


'''
Make a folder in the same directory as Sony dataset folder, with the following directory structure

Sony_RGB
    L Sony
        L long
        L short
            
You can change the folder name from Sony_RGB to something else if you want.
The txt file will be created in the Sony_RGB folder (similar to how it is in the original sony folder)
'''

def get_images():
    new_folder = './Sony_RGB/'  #You can chnage the name if you want
    
    file = open('./dataset/Sony_train_list.txt').read()
    pairs = file.split('\n')
    np.random.shuffle(pairs)
#     pairs = pairs[:5]
    
    image_paths = dict()
    image_paths['dark'] = []
    image_paths['bright'] = []
    
    image_count = 1
    for idx, pair in enumerate(pairs):
        if pair.split()[1] not in image_paths['bright']:
            dark_path = pair.split()[0]
            dark_image = rawpy.imread('./dataset/'+dark_path).postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=8)
            dark_image = cv2.cvtColor(dark_image, cv2.COLOR_RGB2BGR)
            image_paths['dark'].append(dark_path)
            new_dark_path = dark_path[:-3]+'png'
            cv2.imwrite(new_folder + new_dark_path, dark_image)

            bright_path = pair.split()[1]
            bright_image = rawpy.imread('./dataset/'+bright_path).postprocess(use_camera_wb=True, output_bps=8)
            bright_image = cv2.cvtColor(bright_image, cv2.COLOR_RGB2BGR)
            image_paths['bright'].append(bright_path)
            new_bright_path = bright_path[:-3]+'png'
            cv2.imwrite(new_folder + new_bright_path, bright_image)
    
            with open(new_folder + 'Sony_train_list.txt', 'a+') as file:
                file.write(new_dark_path+' '+new_bright_path+'\n')

            print('Image Set: {} \t {} \t {}'.format(image_count, dark_path, bright_path))
            image_count+=1
    
    return image_paths


# In[26]:


paths = get_images()


# In[ ]:




