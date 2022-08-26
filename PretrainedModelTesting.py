from keras.applications.xception import Xception
from keras.applications.xception import decode_predictions
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

model = Xception(weights="imagenet")

img_path = "elephant.jpeg"
img = image.load_img(img_path, target_size=(299, 299))
plt.imshow(img)

img = np.asarray(img)
print(img.shape)
img = np.expand_dims(img, axis=0)
preds = model.predict(img)
print(preds)

# Decoding image layout to find the output/final prediction
print('Predicted:', decode_predictions(preds, top=3)[0])
