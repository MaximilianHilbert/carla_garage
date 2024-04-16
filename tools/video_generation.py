from PIL import Image
import os
from PIL import Image, ImageDraw, ImageFont
import cv2
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
bcso_path="/home/maximilian/Master/carla_garage/vis/bcso"
bcoh_path="/home/maximilian/Master/carla_garage/vis/bcoh"
arp_path="/home/maximilian/Master/carla_garage/vis/arp"
def sort_key(s):
    s=s.strip(".jpg")
    return int(s)
bcso_images=sorted(os.listdir(bcso_path), key=sort_key)
bcoh_images=sorted(os.listdir(bcoh_path), key=sort_key)
arp_images=sorted(os.listdir(arp_path), key=sort_key)
for i in tqdm(range(len(bcso_images))):
    bcso_image=np.array(Image.open(os.path.join(bcso_path,bcso_images[i])))
    bcoh_image=np.array(Image.open(os.path.join(bcoh_path,bcoh_images[i])))
    arp_image=np.array(Image.open(os.path.join(arp_path,arp_images[i])))

    image=np.hstack([bcso_image, bcoh_image, arp_image])
    image=Image.fromarray(image)

    image=np.array(image.resize((1920,1080)))
    if i==0:
        fps = 5
        h,w,_=image.shape
        video_writer = cv2.VideoWriter("/home/maximilian/Master/carla_garage/comparison_one_curve_30.avi", 0,fps, (w, h))
    video_writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
video_writer.release()