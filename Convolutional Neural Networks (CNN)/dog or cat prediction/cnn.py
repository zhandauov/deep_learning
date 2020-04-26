from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout

classifier = Sequential()
input_size = (128, 128)
classifier.add(Conv2D(32, (3, 3), input_shape = (*input_size, 3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Flatten())

classifier.add(Dense(units = 64, activation = 'relu'))
classifier.add(Dense(units = 12, activation = 'relu'))
classifier.add(Dropout(0.4))    
classifier.add(Dense(units = 1, activation = 'sigmoid'))

classifier.compile(optimizer = 'adadelta', loss = 'binary_crossentropy', metrics = ['accuracy'])


from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = input_size,
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = input_size,
                                            batch_size = 32,
                                            class_mode = 'binary')

batchsize = 32
classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 5,
                         validation_data = test_set,
                         validation_steps = 2000)

classifier.save_weights("cat_dog_weights.h5")
print('saved model to cat_dog_weights.h5')


history = {}

import numpy as np
from keras.preprocessing import image

photo = 'milash'
test_image = image.load_img('dataset/single_prediction/%s.jpg' % photo, target_size = (64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
answer1 = classifier.predict(test_image)
if answer1[0][0] == 0:
    print('cat')
    history[photo] = 'cat'
else:
    print('dog')
    history[photo] = 'dog'

history

training_set.class_indices 


classifier.load_weights('cat_dog_weights.h5')
print('loaded model weights')

