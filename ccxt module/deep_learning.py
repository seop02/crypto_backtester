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
    
    X_raw = data
    X_raw = np.array(X_raw)
    length = X_raw.shape[0]
    X = np.delete(X_raw, [0,2], axis = 1)
    LOG.info(X[0])
    
    price = []
    diff = []
    for idx in range(length):
        mid = (X_raw[idx][0][0] + X_raw[idx][2][0])/2
        price.append(mid)

    price = np.array(price)
    price = price[0::60]
    X = X[0::60]
    length = X.shape[0]
    index = []

    for idx in range(length-1):
        difference = (price[idx+1] - price[idx])/price[idx]
        if difference > 0.0003:
            label = 1
            diff.append(label)

        elif difference < -0.0003:
            label = 0
            diff.append(label)
        else:
            index.append(idx)
    
    y = np.array(diff)
    LOG.info(f"y: {y}")
    X = X[:-1]
    X = np.delete(X, index, axis = 0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False, random_state=None)

    input_shape = (2, 100)
    output_classes = 2
    num_samples = 2078

    model = models.Sequential()

    # Layer 1: Flatten the input data
    model.add(layers.Flatten(input_shape=input_shape))

    # Layer 2: Fully Connected Layer
    model.add(layers.Dense(64, activation='relu'))

    # Layer 3: Additional Fully Connected Layer
    model.add(layers.Dense(32, activation='relu'))

    # Layer 4: Additional Fully Connected Layer
    model.add(layers.Dense(16, activation='relu'))

    # Output Layer: Binary Classifier
    model.add(layers.Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model with your data
    model.fit(X_train, y_train, epochs=10, batch_size=32)
    y_pred = model.predict(X_test)
    label = np.argmax(y_pred, axis=1)
    LOG.info(f"y_test: {y_test}")
    LOG.info(f"y_pred: {label}")
    loss, accuracy = model.evaluate(X_test, y_test)
    LOG.info(f"Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}")
