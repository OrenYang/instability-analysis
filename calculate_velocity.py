import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import matplotlib
matplotlib.use("TkAgg")
from scipy.optimize import curve_fit

negative_mask = False
min_radius_mask = True

relative_time = False


def exp_radius_fit(time_range, radius_range, num_points=500):
    """
    Fit imploding Z-pinch radius with a flipped exponential:
        r(t) = -A * exp(alpha * (t - t0)) + C

    Parameters
    ----------
    time_range : ndarray
        Time values
    radius_range : ndarray
        Radius values
    num_points : int
        Number of points in fit curve

    Returns
    -------
    t_fit : ndarray
        Evenly spaced times across the input range
    r_fit : ndarray
        Fitted radius at t_fit
    """

    def negexp_model(t, A, alpha, C, t0):
        dt = t - t0
        return -A * np.exp(alpha * dt) + C

    # Clean input
    mask = np.isfinite(time_range) & np.isfinite(radius_range)
    time_range = np.array(time_range)[mask]
    radius_range = np.array(radius_range)[mask]

    if len(time_range) < 2:
        raise ValueError("Need at least 2 points to fit exponential")

    # Initial guesses
    A_guess = max(1e-6, np.max(radius_range) - np.min(radius_range))
    alpha_guess = 1.0 / (np.max(time_range) - np.min(time_range) + 1e-12)
    C_guess = np.max(radius_range)
    t0_guess = np.min(time_range)

    p0 = [A_guess, alpha_guess, C_guess, t0_guess]
    bounds = ([0, 0, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf])

    # Fit
    popt, _ = curve_fit(negexp_model, time_range, radius_range,
                        p0=p0, bounds=bounds, maxfev=20000)

    # Evaluate
    t_fit = np.linspace(np.min(time_range), np.max(time_range), num_points)
    r_fit = negexp_model(t_fit, *popt)

    return t_fit, r_fit


def radius_fit(time, radius, num_points=500):
    """
    Fit imploding Z-pinch radius with:
        r(t) = r0 - B * (t - t0)**p,   p > 1

    Returns
    -------
    t_fit : ndarray
        Evenly spaced times spanning input time range
    r_fit : ndarray
        Radius fit evaluated at t_fit
    """
    def r_model(t, r0, B, p, t0):
        dt = np.clip(t - t0, 0.0, None)
        return r0 - B * np.power(dt, p)

    # Initial guesses
    r0_guess = radius[0]
    B_guess = max(1e-6, (radius[0] - radius[-1]) / ((time[-1] - time[0])**2 + 1e-12))
    p_guess = 2.0
    t0_guess = time[0]

    p0 = [r0_guess, B_guess, p_guess, t0_guess]
    bounds = ([-np.inf, 0, 2, -np.inf], [np.inf, np.inf, 10.0, np.inf])

    popt, _ = curve_fit(r_model, time, radius, p0=p0, bounds=bounds, maxfev=20000)

    # Evaluate fit
    t_fit = np.linspace(np.min(time), np.max(time), num_points)
    r_fit = r_model(t_fit, *popt)

    return t_fit, r_fit

if __name__ == '__main__':

    root = tk.Tk()
    root.withdraw()

    f = filedialog.askopenfilename(title="Select CSV")
    if not f:
        raise ValueError("No CSV selected")

    df = pd.read_csv(f)
    #plt.scatter(df['Instability Relative_Timing'],df['Instability Radius'])
    #plt.show()

    if relative_time:
        time_name = 'Instability Relative_Timing'
    else:
        time_name = 'Instability Timing'

    time = np.array(df[time_name])
    radius = np.array(df['Instability Radius'])

    # Sort by time
    sort_idx = np.argsort(time)
    time = time[sort_idx]
    radius = radius[sort_idx]

    mask = np.isfinite(time) & np.isfinite(radius)
    time = time[mask]
    radius = radius[mask]

    time_range = time
    radius_range = radius

    # Apply negative mask if requested
    if negative_mask:
        mask_neg = time_range < 0
        time_range = time_range[mask_neg]
        radius_range = radius_range[mask_neg]

    # Apply minimum-radius cutoff if requested
    if min_radius_mask and len(radius_range) > 0:
        idx_min = np.argmin(radius_range)
        time_range = time_range[:idx_min+1]
        radius_range = radius_range[:idx_min+1]

    coef = np.polyfit(time_range,radius_range,3)
    fit = np.poly1d(coef)

    t = np.linspace(np.min(time_range),np.max(time_range),500)
    r = fit(t)

    t,r = radius_fit(time_range,radius_range)

    v = np.gradient(r,t)
    v = v*1e3

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax2.plot(t,v, 'orange')

    ax1.plot(t,r)
    ax1.scatter(time, radius)
    plt.show()
