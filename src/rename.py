import os

imgs = os.listdir('../photos')
for i, img in enumerate(imgs):
    os.rename(f'../photos/{img}', f'../photos/{i+1}.jpg')
