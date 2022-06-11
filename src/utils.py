import re
from glob import glob
from PIL import Image
import os 
from pathlib import Path

# collecting images
p = Path('.')
images = list(p.glob('../input/images/*.jpg'))

# Splitting images into train, valid and test set
train_images = images[:int(7390*0.8)]
valid_images = images[int(7390*0.8):int(7390*0.9)]
test_images = images[int(7390*0.9):]


# to extract label from image path
def find_label(img):
    m = re.search(r'(.+)_\d+.jpg$', img.name)
    return m.group(1)