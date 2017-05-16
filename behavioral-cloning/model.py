import csv
import pickle
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers import Lambda, Activation, Cropping2D
from keras.layers.core import Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.models import Model

import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt

"""
Note, when running on aws machine, need to declare
matplotlib.use('Agg') before importing matplotlib.pyplot
"""

def main():
    """
    Constants
    """
    BATCH_SIZE=32
    EPOCHS=4

    """
    Get the data
    """
    
    all_files = ['./data/driving_log.csv',
                 './data_normal/driving_log.csv',
                 './data_off_road_02/driving_log.csv',
                 './data_off_road_01/driving_log.csv',
                 './data_reverse/driving_log.csv',
                 './data_advanced_normal/driving_log.csv']
    #use data from my normal driving plus recovering from edges of the road (off_road_02)
    files = all_files[1:3]
    samples = get_samples(files)
    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    # compile and train the model using the generator function
    train_generator = generator(train_samples, batch_size=BATCH_SIZE)
    validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)
   
    """
    Build model; basd on NVIDIA's model
    """
    model = Sequential()
    model.add(Cropping2D(cropping=((70,20), (0,0)), input_shape=(160,320,3)))
    model.add(Lambda(lambda x: x / 255.0 - 0.5))
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation="relu"))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation="relu"))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation="relu"))
    model.add(Convolution2D(64,3,3, activation="relu"))
    model.add(Convolution2D(64,3,3, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    """
    #loss function is mse for regression network
    #whereas cross enropy is for classification

    verbose=1: progres bar
    verbose=2: 1 log line per epoch
    """

    model.compile(loss='mse', optimizer='adam')
    history_object = model.fit_generator(train_generator, 
                                        samples_per_epoch = (len(train_samples) // BATCH_SIZE) * BATCH_SIZE, 
                                        validation_data = validation_generator,
                                        nb_val_samples = len(validation_samples), 
                                        nb_epoch = EPOCHS,
                                        verbose = 1)

    model.save('model.h5')

    ### save the history_object, and plot losses on local machine
    """
    Probably can't pickle the history_object
        with open('history_object.pickle', 'wb') as f:
        pickle.dump([history_object, files, EPOCHS], f)
    """
    
    ### print the keys contained in the history object (do this locally)
    """
    
    """
    ### plot the training and validation loss for each epoch
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    files_str = "|".join([file_path.split('/')[-2] for file_path in files])
    plt.title("MSE Loss " + str(EPOCHS) + " epochs" + " files used:\n" + files_str)
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.savefig("history_loss.png", format="png")

#end main()


def get_samples(files):
    samples_all = []
    for file_name in files:
        with open(file_name) as csvfile:
            samples = []
            reader = csv.reader(csvfile)
            for line in reader:
                samples.append(line)
        #the first line is the header, so remove that
        samples = samples[1:]
        samples_all.extend(samples)
    return samples_all


def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images, angles = [], []
            for batch_sample in batch_samples:
                center_path = update_path(batch_sample[0])
                center_image = cv2.imread(update_path(batch_sample[0]))
                left_image = cv2.imread(update_path(batch_sample[1]))
                right_image = cv2.imread(update_path(batch_sample[2]))

                if center_image is None:
                    print("no image found " + center_path)
                steering_offset = 0.3
                center_steering = float(batch_sample[3])
                left_steering = center_steering + steering_offset
                right_steering = center_steering - steering_offset
        
                images.extend([center_image, left_image, right_image])   
                angles.extend([center_steering,left_steering,right_steering])

            augmented_images, augmented_angles = [], []
            for image, angle in zip(images, angles):
                augmented_images.append(image)
                augmented_angles.append(angle)
                augmented_images.append(cv2.flip(image,1))
                augmented_angles.append(angle * -1.0)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield shuffle(X_train, y_train)
        #end for offset in range(...)
    #end while 1:
#end def generator()


def update_path(source_path):
    if(len(source_path)<=10):
        print("short source path:" + source_path)
    source_paths_split = source_path.split('/')
    if len(source_paths_split) <= 2:
        subfolder='data'
    else:
        subfolder = source_paths_split[-3].strip()
    filename = source_paths_split[-1].strip()
    new_path =  './' + subfolder + '/IMG/' + filename
    return new_path
#end def update_path

if __name__ == "__main__":
    main()
