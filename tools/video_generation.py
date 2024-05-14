from PIL import Image
import os
import cv2
from tqdm import tqdm
import numpy as np
def sort_key(s):
    s=s.strip(".jpg")
    return int(s)
def generate_video(path):

    images=sorted(os.listdir(path), key=sort_key)

    for i in tqdm(range(len(images))):
        image_object=np.array(Image.open(os.path.join(path,images[i])))
        image=Image.fromarray(image_object)

        #image=np.array(image.resize((1920,1080)))
        if i==0:
            fps = 5
            h,w,_=image.shape
            video_writer = cv2.VideoWriter(os.path.join(path, "collision.avi"), 0,fps, (w, h))
        video_writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    video_writer.release()

def generate_video_stacked(baselines,root_path):
    images_dict={}
    final_shape=(1920,1080)
    fps = 5
    w,h=final_shape
    video_writer = cv2.VideoWriter(os.path.join(root_path, "comparison_video.avi"), cv2.VideoWriter_fourcc(*'XVID'),fps, (w, h))
    
    for baseline in baselines:
        full_path=os.path.join(root_path, "open_loop",baseline)
        list_of_scenarios=os.listdir(full_path)
        scenario_folder=list_of_scenarios[0]
        subfolder=os.path.join(full_path, scenario_folder)
        list_of_subfolders=os.listdir(subfolder)
        rgb_folder=list_of_subfolders[0]
        final_path=os.path.join(subfolder, rgb_folder)
        images=sorted(os.listdir(final_path), key=sort_key)
        images_dict.update({baseline: [os.path.join(final_path, image_name) for image_name in images]})
    for stacked in zip(*images_dict.values()):
        loaded_image=np.hstack([np.array(Image.open(image_path)) for image_path in stacked])
        image=Image.fromarray(loaded_image)
        image=np.array(image.resize(final_shape))
        video_writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    video_writer.release()
