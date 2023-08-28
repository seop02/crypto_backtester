import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, models
import logging
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.WARNING)
logging.getLogger().setLevel(logging.INFO)
LOG = logging.getLogger(__name__)


if __name__=="__main__":
    # mid = int(data.shape[0]/2)
    # mid1 = int(data1.shape[0]/2)

    # ask = data[:mid]
    # ask1 = data1[:mid1]

    # bid = data[mid:]
    # bid1 = data1[mid:]

    # X = []

    # for idx in range(mid1):
    #     asks = np.transpose(ask1[idx])
    #     bids = np.transpose(bid1[idx])
    #     raw = np.concatenate((asks, bids), axis=0)
    #     X.append(raw)

    # X = np.array(X)
    # LOG.info(X.shape)
    # np.save('orderbook_data_2.npy', X)

    data = np.load('orderbook_data_2.npy')
    
    X = data
    length = X.shape[0]
    price = []
    diff = []
    for idx in range(length):
        mid = (X[idx][0][0] + X[idx][2][0])/2
        price.append(mid)

    price = np.array(price)
    price = price[0::10]
    X = X[0::10]
    length = X.shape[0]

    for idx in range(length-1):
        difference = price[idx+1] - price[idx]
        if difference > 0:
            label = 2
        elif difference < 0:
            label = 1
        else:
            label = 0
        diff.append(label)
    
    y = np.array(diff)

    X = X[:-1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=None)

    input_shape = (4, 100)
    output_classes = 3
    num_samples = 2078

    model = models.Sequential()

    model = models.Sequential()
    model.add(layers.Conv2D(16, (3, 3), activation='relu', input_shape=(*input_shape, 1)))
    model.add(layers.MaxPooling2D((2, 2)))  # Adding pooling layer
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(output_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model with your data
    model.fit(X_train, y_train, epochs=10, batch_size=32)
    loss, accuracy = model.evaluate(X_test, y_test)
    LOG.info(f"Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}")
