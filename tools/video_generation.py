from PIL import Image
import os
import cv2
from tqdm import tqdm
import numpy as np
def sort_key(s):
    s=s.strip(".jpg")
    return float(s)
def generate_video(path, save_path):
    images=sorted(os.listdir(path), key=sort_key)
    for i in tqdm(range(len(images))):
        image_object=np.array(Image.open(os.path.join(path,images[i])))
        image=Image.fromarray(image_object)
        image=np.array(image)
        if i==0:
            fps = 20
            final_shape=((1920, 1080))
            h,w,_=image.shape
            os.makedirs(os.path.dirname(save_path),exist_ok=True)
            video_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'XVID'),fps, (w, h))
        video_writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    video_writer.release()

def generate_video_stacked():
    images_dict={}
    final_shape=(1920,1080)
    fps = 5
    w,h=final_shape
    
    
    for root, dirs, files in os.walk(os.path.join(os.environ.get("WORK_DIR"), "visualisation", "open_loop")):
        if files:
            if files[0].endswith(".jpg"):
                os.makedirs(os.path.dirname(os.path.join(os.environ.get("WORK_DIR"), "visualisation", "videos")),exist_ok=True)
                video_writer = cv2.VideoWriter(os.path.join(os.environ.get("WORK_DIR"), "visualisation", "videos", os.path.basename(os.path.dirname(root))+"_comparison_video.avi"), cv2.VideoWriter_fourcc(*'XVID'),fps, (w, h))
                images=sorted(os.listdir(root), key=sort_key)
                images_dict.update({os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(root)))):
                                    {os.path.basename(os.path.dirname(root)): [os.path.join(root, image_name) for image_name in images]
                                    }})
    collected_data={}
    for baseline, scenarios in images_dict.items():
        for scenario, values in scenarios.items():
            if scenario not in collected_data:
                collected_data[scenario] = []
            collected_data[scenario].append(values)
    for scenario, images in collected_data.items():
        for stacked in zip(*images):
            loaded_image=np.hstack([np.array(Image.open(image_path)) for image_path in stacked])
            image=Image.fromarray(loaded_image)
            image=np.array(image.resize(final_shape))
            video_writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        video_writer.release()
