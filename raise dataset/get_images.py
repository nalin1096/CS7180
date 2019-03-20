import urllib.request as urllib
import pandas as pd 

df = pd.read_csv('./RAISE_1K.csv')
urls = df['NEF'].values

num_images = 100
image_count = 1
for url in urls:
    if image_count>100:
        break
    image_str = './raise_dataset/' + str(image_count) + '.nef'
    print(image_str)
    f = open(image_str,'wb')
    f.write(urllib.urlopen(url).read())
    f.close()
    image_count+=1
