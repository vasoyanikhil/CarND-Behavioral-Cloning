import os
import csv
import math
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import sklearn
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten
from keras.layers import Lambda,Cropping2D
from keras.layers.convolutional import Convolution2D
from zipfile import ZipFile 
#os.remove("data")
def unZip_file(file_name):
   with ZipFile(file_name, 'r') as zip:
       zip.printdir()
       print('Extracting all the files now...')
       zip.extractall()
       print('Done!') 
#unZip_file('/home/workspace/CarND-Behavioral-Cloning-P3/data.zip')

samples = []
#with open('/opt/data/driving_log.csv') as csvfile:
with open('./data/data/driving_log.csv') as csvfile:    
    reader = csv.reader(csvfile)
    #next(reader, None)
    for line in reader:
        samples.append(line)
print(len(samples))

train_samples, validation_samples = train_test_split(samples, test_size=0.2)



def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                #name = '/opt/data/IMG/'+batch_sample[0].split('/')[-1]
                #name = './data/data/IMG/'+batch_sample[0].split('\\')[-1]
                #name_1 = './data/data/IMG/'+batch_sample[1].split('\\')[-1]
                #name_2 = './data/data/IMG/'+batch_sample[2].split('\\')[-1]
                name = './data/data/IMG/'+batch_sample[0].split('\\')[-1]
                name_1 = './data/data/IMG/'+batch_sample[1].split('\\')[-1]
                name_2 = './data/data/IMG/'+batch_sample[2].split('\\')[-1]
                #print(name)
                #print(batch_sample[0])
                center_image = cv2.imread(name)
                letf_image= cv2.imread(name_1)
                right_image= cv2.imread(name_2)
                #print(batch_sample[3])
                center_image=cv2.cvtColor(center_image,cv2.COLOR_BGR2RGB)
                center_angle = float(batch_sample[3])
                
                letf_image=cv2.cvtColor(letf_image,cv2.COLOR_BGR2RGB)
                letf_angle = float(batch_sample[3])+ 0.2
                
                right_image=cv2.cvtColor(right_image,cv2.COLOR_BGR2RGB)
                right_angle = float(batch_sample[3])- 0.2
               
                images.append(center_image)
                images.append(letf_image)
                images.append(right_image)
                
                angles.append(center_angle)
                angles.append(letf_angle)
                angles.append(right_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Set our batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

ch, row, col = 3, 160, 320  # Trimmed image format

model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 

model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(row, col, ch)))

model.add(Lambda(lambda x: x/127.5 - 1.,
        input_shape=(row, col, ch)))


model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))

model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))

model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))

model.add(Convolution2D(64,3,3,activation="relu"))

model.add(Convolution2D(64,3,3,activation="relu"))
          
model.add(Flatten())
          
model.add(Dense(100))
          
model.add(Dense(50))  

model.add(Dense(10))
          
model.add(Dense(1))
          
model.compile(loss='mse', optimizer='adam')
       
model.summary()

model.fit_generator(train_generator,steps_per_epoch=math.ceil(len(train_samples)/batch_size),validation_data=validation_generator,validation_steps=math.ceil(len(validation_samples)/batch_size),epochs=5, verbose=1)


model.save('model.h5')
