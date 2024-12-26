

from tensorflow import keras
from keras.applications.resnet50 import ResNet50
from keras.utils.vis_utils import plot_model
from glob import glob
from keras.models import Model
from keras.layers import Flatten,Dense,Dropout,Softmax
from keras.optimizers import Adam

"""# DATA ENTRY"""

Image_size = [224,224]
valid_path = "/content/drive/MyDrive/Car/Datasets/Test"
train_path = "/content/drive/MyDrive/Car/Datasets/Train"

"""#CREATING THE MODEL AND ASSIGNING THE WEIGHTS"""

resnet = ResNet50(include_top=False , input_shape=Image_size+[3],weights='imagenet')

plot_model(resnet)

for layer in resnet.layers:
  layer.trainable = False

"""# TO GET THE NUMBER OF CLASSES IN OUTPUT"""

folders = glob("/content/drive/MyDrive/Car/Datasets/Train/*")
folders

"""#Adding Extra Layers"""

x = Flatten()(resnet.output)

"""# Creating output layer"""

prediction = Dense(len(folders),activation='softmax')(x)

"""# Creating Model"""

model = Model(inputs = resnet.input , outputs = prediction)

plot_model(model)

model.compile(loss = 'categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory("/content/drive/MyDrive/Car/Datasets/Train",target_size=(224,224),batch_size=32,class_mode='categorical')

test_set = train_datagen.flow_from_directory("/content/drive/MyDrive/Car/Datasets/Test",target_size=(224,224),batch_size=32,class_mode='categorical')

r = model.fit_generator(
  training_set,
  validation_data=test_set,
  epochs=50,
  steps_per_epoch=len(training_set),
  validation_steps=len(test_set)
)

import matplotlib.pyplot as plt
plt.plot(r.history['loss'], label='train loss')
plt.plot(r.history['val_loss'], label='val loss')
plt.legend()
plt.show()
plt.savefig('LossVal_loss')

# plot the accuracy
plt.plot(r.history['accuracy'], label='train acc')
plt.plot(r.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.savefig('AccVal_acc')

from tensorflow.keras.models import load_model

model.save('model_resnet50.h5')

y_pred = model.predict(test_set)

y_pred

y_pred = np.argmax(y_pred, axis=1)

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model=load_model('model_resnet50.h5')

img=image.load_img('/content/drive/MyDrive/Car/Datasets/Test/audi/23.jpg',target_size=(224,224))

x=image.img_to_array(img)

x =x /255

x=np.expand_dims(x,axis=0)
img_data=preprocess_input(x)
img_data.shape

preds = model.predict(x)
preds=np.argmax(preds, axis=1)
if preds==1:
  preds="The Car IS an Audi"
elif preds==2:
  preds="The Car is a Lamborghini"
else:
  preds="The Car is a Mercedes"
print(preds)