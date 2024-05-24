import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from backtest import data_path, months

if __name__ == '__main__':
    date = '2024-03'
    coin = 'KRW-GLM'
    #df = pd.read_parquet(f'{data_path}/ticker/{date}/upbit_volume.parquet')
    df = pd.read_csv(f'{data_path}/ticker/{date}/upbit_volume.csv', index_col=0)
    df = df[df['coin'] == 'KRW-HIFI']

    # Assuming df is your DataFrame
    times = df['time'].values
    devs = df['dev'].values
    prices = df['trade_price'].values
    ma = df['trade_price'].rolling(100).mean().values
    window_size = 1000

    window = np.ones(window_size) / window_size
    moving_avg = np.convolve(np.abs(devs), window, mode='same')

    max_length = max(len(devs), len(moving_avg))
    moving_avg = np.pad(moving_avg, (0, max_length - len(moving_avg)), mode='constant', constant_values=0)

    threshold = 1e2  # Set your threshold value here
    signal = np.abs(devs) > threshold*moving_avg
    indices = np.where(signal)[0]

    # Number of points to label
    n = len(indices)

    # List to store the labels
    labels = [None] * n
    current_index = 0

    # Create a single figure
    fig, ax = plt.subplots(figsize=(12, 6))

    def onclick(event):
        global current_index

        # Check if labeling is complete
        if current_index >= n:
            plt.close()
            return

        # Left click for class 0
        if event.button == 1:
            labels[current_index] = 0
            print(f'Point {indices[current_index]} labeled as class 0')
        # Right click for class 1
        elif event.button == 3:
            labels[current_index] = 1
            print(f'Point {indices[current_index]} labeled as class 1')

        # Move to the next point
        current_index += 1
        if current_index < n:
            plot_point(current_index)
        else:
            print("Labeling completed!")
            plt.close()

    def plot_point(index):
        ax.clear()  # Clear the previous plot

        # Plot the new data
        length = 10000
        time_diff = times[indices[index]:indices[index] + length] - times[indices[index]]
        bought_price = prices[indices[index]]
        decaying_price = (1.001 / 0.9995 + 0.1 * np.exp(-time_diff / 3000)) * bought_price

        ax.plot(times[max(0, indices[index] - 5000):indices[index] + length],
                prices[max(0, indices[index] - 5000):indices[index] + length], color='black')
        ax.plot(times[max(0, indices[index] - 5000):indices[index] + length],
                ma[max(0, indices[index] - 5000):indices[index] + length], color='orange')
        ax.plot(times[indices[index]:indices[index] + length],
                decaying_price, color='red', linestyle='dashed')
        ax.scatter(times[indices[index]], prices[indices[index]], color='blue', s=40)
        ax.set_ylabel('Price')
        ax.set_xlabel('Time')
        ax.set_title(f'Label the point {index + 1} of {n}')
        plt.draw()  # Draw the plot

    # Plot the initial data
    plot_point(current_index)

    # Connect the onclick event to the handler function
    fig.canvas.mpl_connect('button_press_event', onclick)

    # Show the plot
    plt.show()

    # After closing the plot, print all labels
    print("Final labels:", labels)
