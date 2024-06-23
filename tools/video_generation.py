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
                images=sorted(os.listdir(root), key=sort_key)
                scenario=os.path.basename(os.path.dirname(root))
                list_of_images=[os.path.join(root, image_name) for image_name in images]
                if scenario in images_dict.keys():
                    images_dict[scenario].append(
                                        list_of_images
                                        )
                else:
                    images_dict[scenario]=[list_of_images]
    for scenario, images in images_dict.items():
        os.makedirs(os.path.join(os.environ.get("WORK_DIR"), "visualisation", "videos"),exist_ok=True)
        video_writer = cv2.VideoWriter(os.path.join(os.environ.get("WORK_DIR"), "visualisation", "videos", scenario+"_comparison_video.avi"), cv2.VideoWriter_fourcc(*'XVID'),fps, (w, h))
        for stacked in zip(*images):
            loaded_image=np.hstack([np.array(Image.open(img)) for img in stacked])
            image=Image.fromarray(loaded_image)
            image=np.array(image.resize(final_shape))
            video_writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        video_writer.release()
