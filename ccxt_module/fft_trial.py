import numpy as np
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.WARNING)
logging.getLogger().setLevel(logging.INFO)
LOG = logging.getLogger(__name__)

if __name__=="__main__":

    data = np.load('2023-09-05_orderbook1.npy')
    data_upbit = np.load('2023-09-05_upbit_orderbook1.npy')

    X_raw = data
    X_raw = np.array(X_raw)
    length = int(X_raw.shape[0]/2)

    Y_raw = data_upbit
    Y_raw = np.array(Y_raw)
    length_upbit = int(Y_raw.shape[0]/2)

    idx_values = []
    idx_values_upbit = []


    price = []
    for idx in range(length):
        mid = (X_raw[idx][0][0] + X_raw[idx+length][2][0])/2
        price.append(mid)
        idx_values.append(idx)

    price = np.array(price)

    price_upbit = []
    for idx_upbit in range(length):
        mid_upbit = (Y_raw[idx_upbit][0][0] + Y_raw[idx_upbit+length_upbit][2][0])/2
        price_upbit.append(mid_upbit)
        idx_values_upbit.append(idx_upbit)

    price_upbit = np.array(price_upbit)

    fft_binance = np.fft.fft(price)
    fft_upbit = np.fft.fft(price_upbit)
    N = len(price)
    N_upbit = len(price_upbit)
    frequencies_binance = np.fft.fftfreq(N)
    frequencies_upbit = np.fft.fftfreq(N_upbit)
    magnitude_binance = np.abs(fft_binance)
    magnitude_upbit = np.abs(fft_upbit)
    phase_binance = np.angle(fft_binance)
    phase_upbit = np.angle(fft_upbit)


fft_binance_normalized = fft_binance / np.linalg.norm(fft_binance)
fft_upbit_normalized = fft_upbit / np.linalg.norm(fft_upbit)

fft_binance_normalized = fft_binance_normalized[1:]
fft_upbit_normalized = fft_upbit_normalized[1:]

inner_product = np.dot(fft_binance_normalized, fft_upbit_normalized)

LOG.info(inner_product)  


plt.figure(figsize=(8, 4))
plt.title('FFT Frequencies')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')

plt.stem(frequencies_upbit, np.abs(fft_upbit_normalized), use_line_collection=True)
plt.show()

