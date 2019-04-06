import urllib.request as urllib
import pandas as pd 
import rawpy 
import cv2

df = pd.read_csv('./RAISE_1K.csv')
urls = df['NEF'].values

num_images = 1
image_count = 1
for url in urls:
    if image_count>num_images:
        break
    image_str = './raise_dataset/raw/' + str(image_count) + '.nef'
    print(image_str)
    img = urllib.urlopen(url).read()
    f = open(image_str,'wb')
    f.write(img)
    f.close()
    rgb_img = rawpy.imread(image_str).postprocess(use_camera_wb=True, output_bps=8)
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite('./raise_dataset/rgb/' + str(image_count) + '.png', rgb_img)
    image_count+=1
