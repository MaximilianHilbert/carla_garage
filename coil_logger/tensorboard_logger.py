

# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
from io import BytesIO


class Logger(object):

    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        #from datetime import datetime
        #now = datetime.now()
        #log_dir = log_dir + now.strftime("%Y%m%d-%H%M%S")
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        self.writer.add_scalar(tag, value, global_step=step)
    def image_summary(self, tag, images, step):
        """Log a list of images."""
        
        img_summaries = []
        for i, img in enumerate(images):
            # Convert the image to a byte buffer
            print(img.shape)
            print(img)
            
            print(type(img))
            buffer = BytesIO()
            pil_image = Image.fromarray((img * 255).astype(np.uint8).transpose(1,2,0))
            print(pil_image)
            pil_image.save(buffer, format="png")

            # Convert the image to a TensorFlow tensor
            img_tensor = tf.image.decode_image(buffer.getvalue(), channels=img.shape[0])
            img_tensor = tf.expand_dims(img_tensor, axis=0)  # Add batch dimension

            # Append image summary to the list
            img_summaries.append(tf.summary.image('%s/%d' % (tag, i), img_tensor, step=step))

        # Create and write Summary
        print(img_summaries)
        with self.writer.as_default():
            for img_summary in img_summaries:
                tf.summary.image('image', img_summary, step=step)
                self.writer.flush()  # Optionally, flush the writer

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""
        
        # Create a histogram using TensorFlow
        hist, bin_edges = np.histogram(values, bins=bins)
        hist = hist.astype(float)

        # Create and write Summary
        with self.writer.as_default():
            tf.summary.histogram(name=tag, data=values, step=step, buckets=bins)

        self.writer.flush()
