import keras
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from matplotlib import pyplot


datagen = ImageDataGenerator(rotation_range=10)
img = load_img('tumor_
