from PIL import Image
import numpy as np
import os
from os import listdir, path
import re

if not os.path.exists('training/fill50k/source_colors'):
    os.makedirs('training/fill50k/source_colors')

def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    float regex comes from https://stackoverflow.com/a/12643073/190597
    '''
    return [atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text)]


for img_name in sorted(listdir('training/fill50k/target/'), key=natural_keys):
    image_path = path.join('training/fill50k/target/', img_name)
    img = Image.open(image_path)
    colors = img.convert('RGB').getcolors(maxcolors=5)

    if len(colors) == 2:
        # color1 = np.full((1, colors[0][0], 3), colors[0][1], np.uint8)
        # color2 = np.full((1, colors[1][0], 3), colors[1][1], np.uint8)
        # color_img = np.concatenate((color1, color2), axis=1).reshape((512, 512, 3))
        # color_img = Image.fromarray(color_img)
        # color_img.save(path.join('training/fill50k/source_colors/', img_name))
        pass
    elif len(colors) == 1:
        color1 = np.full((1, colors[0][0], 3), colors[0][1], np.uint8)
        color_img = color1.reshape((512, 512, 3))
        color_img = Image.fromarray(color_img)
        color_img.save(path.join('training/fill50k/source_colors/', img_name))
    else:
        print(f'{img_name} {str(len(colors))}\n')

print('Done')