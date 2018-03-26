# Convolutional Neural Network

# 1. Building the CNN

# Importing the Keras libraries packages 
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator

# Initializing the CNN
classifier = Sequential()

# Step 1 : Convolution
# Creating a comvolution layer composed of all feature maps
#classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))
classifier.add(Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)))

# Step 2 : Max Pooling 
# Used to obtain reduced feature maps called the cooling layer 
classifier.add(MaxPooling2D(pool_size = (2,2))) 

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)))
classifier.add(MaxPooling2D(pool_size = (2,2))) 


# Step 3 : Flattening
classifier.add(Flatten())

# Step 4 : Full Connection
classifier.add(Dense(units = 128, activation = 'relu'))
# Output layer
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy' , metrics = ['accuracy'])

# Part 2 - Fitting the CNN to images 
# Application of image augmentation which refers to enriching our training set 
# without adding more images and therefore that allows to get good performance results
# with little or no overfitting
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(training_set,
                        steps_per_epoch = (8000/32),
                        epochs = 25,
                        validation_data = test_set,
                        validation_steps = (2000/32))



