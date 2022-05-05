import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from keras.preprocessing import image

model = load_model("bestmodel.h5")

class_type = {0:'Covid',  1 : 'Normal'}

def get_img_array(img_path):
  """
  Input : Takes in image path as input
  Output : Gives out Pre-Processed image
  """
  path = img_path
  img = image.load_img(path, target_size=(224,224,3))
  img = image.img_to_array(img)/255
  img = np.expand_dims(img , axis= 0 )

  return img

  # path for that new image. ( you can take it either from google or any other scource)

path = "all_images/COVID-2051.png"       # you can add any image path

#predictions: path:- provide any image from google or provide image from all image folder
img = get_img_array(path)

res = class_type[np.argmax(model.predict(img))]
print(f"The given X-Ray image is of type = {res}")
print()
print(f"The chances of image being Covid is : {model.predict(img)[0][0]*100} percent")
print()
print(f"The chances of image being Normal is : {model.predict(img)[0][1]*100} percent")

# to display the image
plt.imshow(img[0], cmap = "gray")
plt.title("input image")
plt.show()
