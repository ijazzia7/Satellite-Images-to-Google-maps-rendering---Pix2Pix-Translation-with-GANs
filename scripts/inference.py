import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from PIL import Image

from train import generate_fake_samples

# ---------------------------
# Load and Preprocess Image
# ---------------------------
def preprocess_image(path, size=(256, 256)):
    img = load_img(path, target_size=size)
    img = img_to_array(img)
    img = (img - 127.5) / 127.5
    return np.expand_dims(img, axis=0)

# ---------------------------
# Run Inference
# ---------------------------
if __name__ == "__main__":
    generator = load_model("model_054800.keras")
    img = preprocess_image("/kaggle/input/images/sample.png")
    fake, _ = generate_fake_samples(generator, img, 16)
    fake = (fake[0] + 1) / 2.0  # rescale back to [0,1]
    plt.imshow(fake)
    plt.axis("off")
    plt.show()
