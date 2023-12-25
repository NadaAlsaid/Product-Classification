import cv2
import argparse
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score

sns.set(style="whitegrid")
import os
import glob as gb
import cv2
import keras
import tensorflow as tf


def getFiles_train( path):
    imlist = {}
    count = 0
    x_train = []
    y_train = []
    for i in range(20):
        each = os.listdir(path + '/' + str(i + 1) + '/Train')
        print(" #### Reading image category ", each, " ##### ")
        imlist[i + 1] = []
        for imagefile in each:
            print("Reading file ", imagefile)
            im = cv2.imread(path + '/' + str(i + 1) + '/Train/' + imagefile)
            kernel2 = np.ones((5, 5), np.float32) / 25
            # Applying the filter
            im = cv2.filter2D(src=im, ddepth=-1, kernel=kernel2)
            desired_width = 100
            desired_height = 100
            im = cv2.resize(im, (desired_width, desired_height))
            x_train.append(im)
            y_train.append(i)
            imlist[i + 1].append(im)
            count += 1
    return [count,  x_train, y_train]

# def read_func(path):
#     x_train = []
#     y_train = []
#     i=0
#     s=100
#     for i in range(20):
#         each = os.listdir(path + '/' + str(i + 1) + '/Train')
#         print(" #### Reading image category ", each, " ##### ")
#         for file in each:
#             image = cv2.imread(path + '/' + str(i + 1) + '/Train/' +file)
#             image_array = cv2.resize(image, (s, s))
#             x_train.append(list(image_array))
#             y_train.append(i)
#
#         i+=1
#     return x_train , y_train
def getFiles_test( path):
    imlist = {}
    count = 0
    x_test = []
    y_test = []
    for i in range(20):
        each = os.listdir(path + '/' + str(i + 1) + '/Validation')
        print(" #### Reading image category ", each, " ##### ")
        imlist[i + 1] = []
        for imagefile in each:
            print("Reading file ", imagefile)
            im = cv2.imread(path + '/' + str(i + 1) + '/Validation/' + imagefile)
            kernel2 = np.ones((5, 5), np.float32) / 25
            # Applying the filter
            im = cv2.filter2D(src=im, ddepth=-1, kernel=kernel2)
            desired_width = 100
            desired_height = 100
            im = cv2.resize(im, (desired_width, desired_height))
            imlist[i + 1].append(im)
            x_test.append(im)
            y_test.append(i)
            count += 1
    return [count, x_test, y_test]


# def getFiles_predict(path):
#     imlist = {}
#     count = 0
#     x_predict = []
#     y_predict=[]
#     for imagefolder in os.listdir(path):
#         # print("Reading file ", imagefile)
#         for imagefile in os.listdir(path +'/'+imagefolder):
#             im = cv2.imread(path +'/'+imagefile)
#             # kernel2 = np.ones((5, 5), np.float32) / 25
#             # # Applying the filter
#             # im = cv2.filter2D(src=im, ddepth=-1, kernel=kernel2)
#             desired_width = 100
#             desired_height = 100
#             im = cv2.resize(im, (desired_width, desired_height))
#             x_predict.append(im)
#             y_predict.append()
#             count += 1
#     return [count,x_predict]

def getFiles_s(path):
        imlist = []
        count = 0
        for each in os.listdir(path):
            print(" #### Reading image category ", each, " ##### ")
            # imlist[each] = []
            for imagefile in os.listdir(path + '/' + each):
                print("Reading file ", imagefile)
                im = cv2.imread(path + '/' + each + '/' + imagefile)
                kernel2 = np.ones((5, 5), np.float32) / 25
                # Applying the filter
                im = cv2.filter2D(src=im, ddepth=-1, kernel=kernel2)
                desired_width = 100
                desired_height = 100
                im = cv2.resize(im, (desired_width, desired_height))
                imlist.append(im)
                count += 1
        return imlist,count
def train_CNN(train_path,test_path):
    xpredict, xpredict_count= getFiles_s("Test Samples Classification")
    count_train,x_train,y_train = getFiles_train(train_path)
    count_test , x_test, y_test = getFiles_test(test_path)

    print("Data Type 1:", type(xpredict))
    print("Data Type 2:", type(x_train))
    print("Data Type 3:", type(x_test))

    # print("Num of images",len(self.x_train))
    #
    # for i in range(self.trainImageCount):
    #     plt.imshow(cv2.cvtColor(self.x_train[i], cv2.COLOR_GRAY2RGB))
    #     plt.title(self.y_train[i])
    #     cv2.waitKey()
    #     plt.show()
    # x_train = np.array(x_train)
    # y_train = np.array(y_train)
    # x_test = np.array(x_test)
    # y_test = np.array(y_test)

    xpredict_new = np.array(xpredict)
    x_train_new = np.array(x_train)
    y_train_new = np.array(y_train)
    x_test_new = np.array(x_test)
    y_test_new = np.array(y_test)

    print("Data Type 1:", type(xpredict))
    print("Data Type 2:", type(x_train))
    print("Data Type 3:", type(x_test))
    # extra_channel = np.zeros((x_train_new.shape[0], x_train_new.shape[1], x_train_new.shape[2], 1))
    # x_train_new_with_extra_channel = np.concatenate((x_train_new, extra_channel), axis=-1)
    # extra_channel = np.zeros((y_train_new.shape[0], y_train_new.shape[1], y_train_new.shape[2], 1))
    # y_train_new_with_extra_channel = np.concatenate((y_train_new, extra_channel), axis=-1)

    # x_pred_array = np.array(x_pred)
    # for i in range (count_train) :
    #     plt.imshow(cv2.cvtColor(x_train[i], cv2.COLOR_GRAY2RGB))
    #     plt.title(y_train[i])
    #     cv2.waitKey()
    #     plt.show()
    # for j in range(count_test):
    #     plt.imshow(cv2.cvtColor(x_test[j], cv2.COLOR_GRAY2RGB))
    #     plt.title(y_test[j])
    #     cv2.waitKey()
    #     plt.show()

    print(f'X_train shape  is {x_train_new.shape}')
    # # print(f'X_pred shape  is {self.x_pred_array.shape}')
    print(f'y_train shape  is {y_train_new.shape}')
    print(f'X_test shape  is {x_test_new.shape}')
    print(f'y_test shape  is {y_test_new.shape}')
    KerasModel = keras.models.Sequential([
        keras.layers.Conv2D(200, kernel_size=(3, 3), activation='relu', input_shape=(100, 100, 3)),
        keras.layers.Conv2D(150, kernel_size=(3, 3), activation='relu'),
        keras.layers.MaxPool2D(4, 4),
        keras.layers.Conv2D(120, kernel_size=(3, 3), activation='relu'),
        keras.layers.Conv2D(80, kernel_size=(3, 3), activation='relu'),
        keras.layers.Conv2D(50, kernel_size=(3, 3), activation='relu'),
        keras.layers.MaxPool2D(4, 4),
        keras.layers.Flatten(),
        keras.layers.Dense(120, activation='relu'),
        keras.layers.Dense(100, activation='relu'),
        keras.layers.Dense(50, activation='relu'),
        keras.layers.Dropout(rate=0.5),
        keras.layers.Dense(20, activation='softmax'),
    ])

    KerasModel.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print('Model Details are : ')
    print(KerasModel.summary())

    epochs = 50
    ThisModel = KerasModel.fit(x_train_new, y_train_new, epochs=epochs, batch_size=64, verbose=1)
    ModelLoss_train, ModelAccuracy_train = KerasModel.evaluate(x_train_new, y_train_new)
    #
    print('Train Loss is {}'.format(ModelLoss_train))
    print('Train Accuracy is {}'.format(ModelAccuracy_train))

    # ThisModel = KerasModel.fit(x_test_new, y_test_new, epochs=epochs, batch_size=64, verbose=1)
    ModelLoss, ModelAccuracy = KerasModel.evaluate(x_test_new,y_test_new)
    # print('Test Loss is {}'.format(ModelLoss))
    print('Test Accuracy is {}'.format(ModelAccuracy))

    # # Save the model
    # save_path = ""  # Replace with the desired save path
    # KerasModel.save(save_path)
    # +("Model saved successfully.")

    # Plotting the histogram of train and test labels
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist(y_train_new, bins=20, color='blue', alpha=0.7)
    plt.title('Train Labels')
    plt.xlabel('Labels')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    plt.hist(y_test_new, bins=20, color='red', alpha=0.7)
    plt.title('Test Labels')
    plt.xlabel('Labels')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()


    # for i in range (count) :
    #     plt.imshow(cv2.cvtColor(x_train[i], cv2.COLOR_GRAY2RGB))
    #     plt.title(y_train[i])
    #     cv2.waitKey()
    #     plt.show()
    #y_result = KerasModel.predict(xpredict_new)
    #y_pred = KerasModel.predict(xpredict_new)


    score = KerasModel.evaluate(x_test_new, y_test_new)
    # accuracy
    print("KerasModel test Accuracy", score[1] * 100)
    # print('Prediction Shape is {}'.format(y_result.shape))

    plt.figure(figsize=(20, 20))
    # for n, i in enumerate(list(np.random.randint(0, len(xpredict_count), 20))):
    #     plt.subplot(6, 6, n + 1)
    #     plt.imshow(xpredict[i])
    #     plt.axis('off')
    #     plt.title(y_result[i])

    # for i in range (xpredict_count):
    #     # plt.subplot(6, 6, n + 1)
    #     plt.imshow(xpredict[i])
    #     plt.axis('off')
    #     max_val = np.max(y_pred, axis=None, keepdims=False)
    #     plt.title(max_val)
    #     plt.show()

train_path = "BoWCodeStudents/Product Classification"
test_path = "BoWCodeStudents/Product Classification"
train_CNN(train_path,test_path )
