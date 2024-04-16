from PIL import Image
from tqdm import tqdm
import os
for baseline in ["bcso", "bcoh", "arp"]:
    path=os.path.join("/home/maximilian-hilbert/carla_garage/vis", baseline)
    path_new=os.path.join("/home/maximilian/Master/carla_garage/vis", "new", baseline)
    os.makedirs(path_new, exist_ok=True)
    for image_name in tqdm(os.listdir(path)):
        image = Image.open(os.path.join(path, image_name))
        rgb_im = image.convert('RGB')
        rgb_im.save(os.path.join(path_new, image_name.replace(".png", ".jpg")), quality=95)

        # Close the image
        image.close()