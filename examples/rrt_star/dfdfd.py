

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('EUR_USD.csv')
data.columns = ['Date', 'open', 'high', 'low', 'close','volume']

data.Date = pd.to_datetime(data.Date, format='%d.%m.%Y %H:%M:%S.%f')

data = data.set_index(data.Date)

data = data[['open', 'high', 'low', 'close']]

data = data.drop_duplicates(keep=False)

price = data.close.values
plt.rcParams.update({'font.size': 45})

def fft_detect(price, p=0.4):

    trans = np.fft.rfft(price)
    # trans[round(p*len(trans)):] = 0
    inv = np.fft.irfft(trans)
    dy = np.gradient(inv)
    peaks_idx = np.where(np.diff(np.sign(dy)) == -2)[0] + 1
    valleys_idx = np.where(np.diff(np.sign(dy)) == 2)[0] + 1

    patt_idx = list(peaks_idx) + list(valleys_idx)
    patt_idx.sort()

    label = [x for x in np.diff(np.sign(dy)) if x != 0]

    # Look for Better Peaks

    l = 2

    new_inds = []

    for i in range(0,len(patt_idx[:-1])):

        search = np.arange(patt_idx[i]-(l+1),patt_idx[i]+(l+1))

        if label[i] == -2:
            idx = price[search].argmax()
        elif label[i] == 2:
            idx = price[search].argmin()

        new_max = search[idx]
        new_inds.append(new_max)

    # plt.plot(price)
    plt.plot(inv)
    # plt.scatter(patt_idx,price[patt_idx])
    # plt.scatter(new_inds,price[new_inds],c='g')
    plt.show()

    return peaks_idx, price[peaks_idx]


fft_detect(price, p=0.4)