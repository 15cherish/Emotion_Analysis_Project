import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir = 'data/train'
val_dir = 'data/test'
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen=ImageDataGenerator(rescale=1./255)
train_generator= train_datagen.flow_from_directory(
    train_dir,
    target_size=(48,48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(48,48),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical')


emotion_model = Sequential()
emotion_model.add(Conv2D(32, kernel_size=(3,3),activation='relu',input_shape=(48,48,1)))
emotion_model.add(Conv2D(64, kernel_size=(3,3),activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2,2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Conv2D(128, kernel_size=(3,3),activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2,2)))
emotion_model.add(Conv2D(128, kernel_size=(3,3),activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2,2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Flatten())
emotion_model.add(Dense(1024,activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7,activation='softmax'))


emotion_model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001,decay=1e-6),metrics=['accuracy'])
emotion_model_info=emotion_model.fit_generator(
    train_generator,
    steps_per_epoch = 28709 // 64,
    epochs = 5,
    validation_data = validation_generator,
    validation_steps= 7178 // 64)
emotion_model.save('model.h5')
emotion_model.load_weights('model.h5')

# cv2.ocl.setUseOpenCL(False)
# global cap
# show_text = [0]
# #global frame_number

# emotion_dict = {0: " Angry ", 1: " Disgusted ",2: " Fearful ",3: " Happy ",4: " Neutral ",5: " Sad ",6: " Surpriced "}
# cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture(r'C:\Users\Cherish Mahajan\Videos\Captures\Camer.mp4')
# if not cap.isOpened():
#     print("Can't open the camera")
# global frame_number 
# frame_number = 0
# length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# frame_number += 1
# if frame_number >= length :
#     exit()
# cap.set(1,frame_number)
# flag1, frame1 = cap.read()
# frame1 = cv2.resize(frame1, (600,500))
# bounding_box=cv2.CascadeClassifier('C:/Users/Cherish Mahajan/anaconda3/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
# gray_frame=cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
# num_faces = bounding_box.detectMultiScale(gray_frame, scaleFactor= 1.3, minNeighbors=5)
# for (x,y,w,h) in num_faces:
#     cv2.rectangle(frame1,(x,y-50),(x+w, y+h+10),(255,0,0),2)
#     roi_gray_frame = gray_frame[y:y +h, x:x + w]
#     cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame,(48,48)),-1),0)
#     prediction = emotion_model.predict(cropped_img)
#     maxindex = int(np.argmax(prediction))
#     cv2.putText(frame1, emotion_dict[maxindex],(x+20, y-60),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)
#     show_text[0]=maxindex