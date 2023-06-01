import json
import numpy as np

from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt

DATASET_PATH = "data_gender_3.json"

def load_data(data_path):
    with open(data_path, "r") as fp:
        data = json.load(fp)
    
    # convert lists into numpy array
    inputs = np.array(data["mfcc"])
    targets = np.array(data["labels"])

    return inputs, targets

def prepare_datasets(test_size, validation_size):
    # load data
    x, y = load_data(DATASET_PATH)

    # create train/test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=test_size, random_state=42, shuffle=True)

    # create train/validation split
    x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train,stratify=y_train, test_size=validation_size, random_state=42, shuffle=True)

    # 3d array (adds the "gray chanel" to the mfccs)
    x_train = x_train[..., np.newaxis] # 4d array -> (num_samples, 130, 13, 1)
    x_validation = x_validation[..., np.newaxis]
    x_test = x_test[..., np.newaxis]
    
    return x_train, x_validation, x_test, y_train, y_validation, y_test

def build_model(input_shape):
    # create model
    model = keras.Sequential()

    # 1st conv layer             # kernals  
    model.add(keras.layers.Conv2D(32,
                                 # size of the kernel   
                                 (3, 3), 
                                 # wich activation function
                                 activation= "relu", 
                                 # input shape (13,130,1)
                                 input_shape=input_shape
                                 ))
    
    model.add(keras.layers.MaxPool2D((3, 3), strides = (2,2), padding = "same"))
    model.add(keras.layers.BatchNormalization()) # o modelo vai convergir mais rápido por isso tem que colocar essa linha    

    # 2nd conv layer

     # 1st conv layer             # kernals  
    model.add(keras.layers.Conv2D(32,
                                 # size of the kernel   
                                 (3, 3), 
                                 # wich activation function
                                 activation= "relu", 
                                 # input shape (13,130,1)
                                 #input_shape=input_shape
                                 ))
    
    model.add(keras.layers.MaxPool2D((3, 3), strides = (2,2), padding = "same"))
    model.add(keras.layers.BatchNormalization()) # o modelo vai convergir mais rápido por isso tem que colocar essa linha    

    # 3rd conv layer

     # 1st conv layer             # kernals  
    model.add(keras.layers.Conv2D(32,
                                 # size of the kernel   
                                 (2, 2), 
                                 # wich activation function
                                 activation= "relu", 
                                 # input shape (13,130,1)
                                 #input_shape=input_shape
                                 ))
    
    model.add(keras.layers.MaxPool2D((2, 2), strides = (2,2), padding = "same"))
    model.add(keras.layers.BatchNormalization()) # o modelo vai convergir mais rápido por isso tem que colocar essa linha    

    # flatten the output and feed it into dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation= "relu"))
    model.add(keras.layers.Dropout(0.3))

    # output layer
    model.add(keras.layers.Dense(2, activation="softmax"))
    
    return model

def plot_history(history):
    fig, axs = plt.subplots(2)

    # create accuracy subplot
    axs[0].plot(history.history["accuracy"], label = "train accuaracy")
    axs[0].plot(history.history["val_accuracy"], label = "val accuaracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error subplot
    axs[1].plot(history.history["loss"], label = "train error")
    axs[1].plot(history.history["val_loss"], label = "val error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()

def predict(model, x, y):

    classes = ['female', 'male']

    x = x[np.newaxis, ...]
    # prediction [[0.1, 0.2, ...]]
    prediction = model.predict(x) # x -> (1, 130, 13, 1)

    #extract idenx with max value
    predicted_index = np.argmax(prediction, axis=1) # [4]

    print(f"Expected genre: {classes[predicted_index[0]]}")
    print(f"Predicted genre: {classes[y]}")



def main():
    # create train, validation and test sets
    x_train, x_validation, x_test, y_train, y_validation, y_test = prepare_datasets(0.01, 0.01)
    
    # build the CNN net
    input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])
    model = build_model(input_shape)
    
    # # compile the network
    optimizer = keras.optimizers.Adam(learning_rate = 0.0001)

    model.compile(optimizer = optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics= ["accuracy"])

    print(x_train.shape)
    print(x_validation.shape)
    print(x_test.shape)

    ##train the CNN

    model.summary()

    # train network
    history = model.fit(x_train,
              y_train, 
              validation_data = (x_validation , y_validation), 
              epochs=200,
              batch_size=32)
    
    #plot_history(history)

    model.save_weights("./checkpoints/gender_classifier_checkpoint")
    
    # evaluate the CNN on the test set
    # test_error, test_accuracy = model.evaluate(x_test, y_test, verbose=1)
    # print(f"Acuracy on test set is: {test_accuracy}")
    # print(f"Error on test set is: {test_error}")

    # model = build_model(input_shape)
    # model.load_weights('./checkpoints/gender_classifier_checkpoint').expect_partial()
    # make predictions on a sample
    #x = x_test[102]
    #y = y_test[102]
    #predict(model, x, y)

    
    
if (__name__ == "__main__"):
    main()