import numpy
import skimage.io, skimage.color, skimage.feature
import os
import pickle

fruits = ["apple", "lemon", "mango", "raspberry"]
#164+166+166+166=662
dataset_features = numpy.zeros(shape=(662, 360))
outputs = numpy.zeros(shape=(662))

idx = 0
class_label = 0
for fruit_dir in fruits:
    curr_dir = os.path.join(os.path.sep + "Datasets/Fruits-360/Test", fruit_dir)
    all_imgs = os.listdir(os.getcwd()+curr_dir)
    for img_file in all_imgs:
        if img_file.endswith(".jpg"): # Ensures reading only JPG files.
            fruit_data = skimage.io.imread(fname=os.path.sep.join([os.getcwd(), curr_dir , img_file]), as_gray=False)
            fruit_data_hsv = skimage.color.rgb2hsv(rgb=fruit_data)
            hist = numpy.histogram(a=fruit_data_hsv[:, :, 0], bins=360)
            dataset_features[idx, :] = hist[0]
            outputs[idx] = class_label
            idx = idx + 1
    class_label = class_label + 1

with open("Datasets/test_set_features.pkl", "wb") as f:
    pickle.dump(dataset_features, f)

with open("Datasets/test_set_labels.pkl", "wb") as f:
    pickle.dump(outputs, f)
