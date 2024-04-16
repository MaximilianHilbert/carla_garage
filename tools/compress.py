from PIL import Image
import os
for baseline in ["bcso", "bcoh", "arp"]:
    path=os.path.join("/home/maximilian-hilbert/carla_garage/vis", baseline)
    for image_name in os.listdir(path):
        image = Image.open(os.path.join(path, image_name))


        compressed_image = image.convert("RGB").save("output_compressed.png", optimize=True, quality=9)

        # Close the image
        image.close()