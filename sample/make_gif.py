import os
from glob import glob
import cv2
import imageio

def sort_by_img_num(name: str):
    img_name = os.path.basename(name).split(".")[0]
    num = img_name.split("_")[-1]
    return int(num)

image_paths = glob("./*.png")
image_paths = sorted(image_paths, key=sort_by_img_num)
images = [cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB) for image_path in image_paths]
images += [images[-1]]*10
imageio.mimsave('animation.gif', images, fps=7, loop=0)

