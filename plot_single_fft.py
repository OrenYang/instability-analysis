import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import matplotlib
matplotlib.use("TkAgg")

root = tk.Tk()
root.withdraw()

f = filedialog.askopenfilename()

data = np.load(f)

wl = data['fft_wavelengths_detrended']
psd = data['fft_psd_detrended']

plt.plot(wl,psd)
plt.show()
