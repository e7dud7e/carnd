import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers import Lambda, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D


def main():
    # my code here
    """
    Get the data
    """

    lines = []
    with open('./data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

    #the first line is the header, so remove that
    lines = lines[1:]
    images = [] #input X
    measurements = [] #output y
    #there are about 8000 rows, so I'll use the first 2000 to deal with memory issues
    for line in lines[0:2000]:
        """
        line[0] has original file path of center image; line[1] is left image, line[2] is right image
        """
        center_image = cv2.imread(update_path(line[0]))
        left_image = cv2.imread(update_path(line[1]))
        right_image = cv2.imread(update_path(line[2]))

        steering_offset = 0.2
        center_steering = float(line[3])
        left_steering = center_steering + steering_offset
        right_steering = center_steering - steering_offset
    
        images.extend([center_image, left_image, right_image])   
        measurements.extend([center_steering,left_steering,right_steering])

    augmented_images, augmented_measurements = [], []
    for image, measurement in zip(images, measurements):
        augmented_images.append(image)
        augmented_measurements.append(measurement)
        augmented_images.append(cv2.flip(image,1))
        augmented_measurements.append(measurement * -1.0)

    """
    with full data set, memory overflow.  For now, just use less of the data, will
    work on generators later

    """
    X_train = np.array(augmented_images, dtype=np.float32)
    y_train = np.array(augmented_measurements)

    """
    Build model
    """
    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
    model.add(Convolution2D(6,5,5,activation="relu"))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6,5,5,activation="relu"))
    model.add(MaxPooling2D())
    model.add(Convolution2D(16,5,5,activation="relu"))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Dense(84))
    model.add(Dense(1))

    """
    #loss function is mse for regression network
    #whereas cross enropy is for classification

    verbose=1: progres bar
    verbose=2: 1 log line per epoch
    """

    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, 
            y_train, 
            batch_size=32,
            nb_epoch=2,
            verbose=2,
            validation_split=0.2,
            shuffle=True)

    model.save('model.h5')

def update_path(source_path):
    filename = source_path.split('/')[-1]
    return './data/IMG/' + filename



if __name__ == "__main__":
    main()