import csv
import cv2
import numpy as np
import matplotlib.image as mpimg
from sklearn.utils import shuffle

samples = []
datadir = 'Train_Data/'
dataPath = datadir + 'driving_log.csv'

def getSampleImagesAndMeasurements(dataPath):
    with open(dataPath) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
    return samples


samples = getSampleImagesAndMeasurements(dataPath)

# Remove the first line, which contains the description of each data column
samples = samples[1:]

print("samples length: ",len(samples))

# Split data into training and validation data
# 80% of the data will be training, 20% for validation.
from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

print('Train samples: {}'.format(len(train_samples)))
print('Validation samples: {}'.format(len(validation_samples)))


def generator(samples, batch_size=32):
    num_samples = len(samples)
    
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            measurements = []
            for batch_sample in batch_samples:
                # Extract filenames
              filename_center = batch_sample[0].split('/')[-1]
              filename_left = batch_sample[1].split('/')[-1]
              filename_right = batch_sample[2].split('/')[-1]
              
              # Construct the path
              path_center = datadir + 'IMG/' + filename_center
              path_left = datadir + 'IMG/' + filename_left
              path_right = datadir + 'IMG/' + filename_right
              
              #Read the left, center, right images
              image_center = mpimg.imread( path_center )
              image_left = mpimg.imread( path_left )
              image_right = mpimg.imread( path_right )
              
              images.append(image_center)
              images.append(image_left)
              images.append(image_right)
              # Flipping
              images.append(cv2.flip(image_center,1))
              
              #measuring the angles and adjusting left and right angles
              correction = 0.125
              measurement_center=float(batch_sample[3])
              measurement_left = measurement_center + correction
              measurement_right = measurement_center - correction

              measurements.append(measurement_center)
              measurements.append(measurement_left)
              measurements.append(measurement_right)
              measurements.append(measurement_center * -1.0)
                
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(measurements)
            yield shuffle(X_train, y_train)


from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Cropping2D, Dropout

#Preprocess the model layers
# 1. Normalize
# 2. Cropping

def preProcessingLayers():
    model = Sequential()
    # Normalizing
    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160,320,3)))
    # cropping the images
    model.add(Cropping2D(cropping=((50,20), (0,0))))
    return model

def createModel():
    model = preProcessingLayers()
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(0.5)) 
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    return model

#  Generators for training and validation data,
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# Model creation
model = createModel()

# Compiling and training the model
model.compile(loss='mse', optimizer='adam')

history_object = model.fit_generator(train_generator, samples_per_epoch= \
                                     len(train_samples), validation_data=validation_generator, \
                                     nb_val_samples=len(validation_samples), nb_epoch=3, verbose=1)

model.save('model.h5')
print(history_object.history.keys())
print('Loss')
print(history_object.history['loss'])
print('Validation Loss')
print(history_object.history['val_loss'])

