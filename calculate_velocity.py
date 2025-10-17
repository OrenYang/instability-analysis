import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import matplotlib
matplotlib.use("TkAgg")
from scipy.optimize import curve_fit
from scipy.special import erf


# -------------------- USER OPTIONS --------------------
negative_mask = False
min_radius_mask = True
relative_time = True
fit_method = "erf"    # options: "poly", "power", "exp", "erf"
# ------------------------------------------------------
# -------------------- ADDITIONAL VELOCITY SCATTER --------------------
def velocity_scatter(time, radius, dt_average=5):
    """
    Average radii within dt_average window (default 5 ns).
    Compute velocity from averaged points and return std for error bars.
    """
    time = np.array(time)
    radius = np.array(radius)

    sort_idx = np.argsort(time)
    time = time[sort_idx]
    radius = radius[sort_idx]

    averaged_time = []
    averaged_radius = []
    std_radius = []

    i = 0
    N = len(time)
    while i < N:
        t_group = [time[i]]
        r_group = [radius[i]]
        j = i + 1
        while j < N and time[j] - t_group[0] <= dt_average:
            t_group.append(time[j])
            r_group.append(radius[j])
            j += 1

        averaged_time.append(np.mean(t_group))
        averaged_radius.append(np.mean(r_group))
        std_radius.append(np.std(r_group) if len(r_group) > 1 else 0)

        i = j

    averaged_time = np.array(averaged_time)
    averaged_radius = np.array(averaged_radius)
    std_radius = np.array(std_radius)

    dt = np.diff(averaged_time)
    dr = np.diff(averaged_radius)
    vel = dr / dt * 1e3   # km/ns → m/s
    t_vel = averaged_time[:-1] + dt/2

    return averaged_time, averaged_radius, std_radius, t_vel, vel


def erf_radius_fit(time_range, radius_range, num_points=500):
    mask = np.isfinite(time_range) & np.isfinite(radius_range)
    t_data = np.array(time_range)[mask]
    r_data = np.array(radius_range)[mask]

    R0_guess = np.max(r_data)
    Rmin_guess = np.min(r_data)
    t0_guess = t_data[np.argmin(r_data)]
    Delta_guess = (np.max(t_data) - np.min(t_data)) / 4

    def r_model(t, R0, Rmin, t0, Delta):
        return Rmin + (R0 - Rmin) * (1 + erf((t0 - t)/Delta))/2

    p0 = [R0_guess, Rmin_guess, t0_guess, Delta_guess]
    bounds = ([0, 0, np.min(t_data), 1e-9],
              [np.inf, np.inf, np.max(t_data), np.inf])

    popt, _ = curve_fit(r_model, t_data, r_data, p0=p0, bounds=bounds, maxfev=20000)

    t_fit = np.linspace(np.min(t_data), np.max(t_data), num_points)
    r_fit = r_model(t_fit, *popt)

    return t_fit, r_fit


def exp_radius_fit(time_range, radius_range, num_points=500):
    def negexp_model(t, A, alpha, C, t0):
        dt = t - t0
        return -A * np.exp(alpha * dt) + C

    mask = np.isfinite(time_range) & np.isfinite(radius_range)
    time_range = np.array(time_range)[mask]
    radius_range = np.array(radius_range)[mask]

    A_guess = max(1e-6, np.max(radius_range) - np.min(radius_range))
    alpha_guess = 1.0 / (np.max(time_range) - np.min(time_range) + 1e-12)
    C_guess = np.max(radius_range)
    t0_guess = np.min(time_range)

    p0 = [A_guess, alpha_guess, C_guess, t0_guess]
    bounds = ([0, 0, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf])

    popt, _ = curve_fit(negexp_model, time_range, radius_range,
                        p0=p0, bounds=bounds, maxfev=20000)

    t_fit = np.linspace(np.min(time_range), np.max(time_range), num_points)
    r_fit = negexp_model(t_fit, *popt)

    return t_fit, r_fit


def radius_fit(time, radius, num_points=500):
    def r_model(t, r0, B, p, t0):
        dt = np.clip(t - t0, 0.0, None)
        return r0 - B * np.power(dt, p)

    r0_guess = radius[0]
    B_guess = max(1e-6, (radius[0] - radius[-1]) / ((time[-1] - time[0])**2 + 1e-12))
    p_guess = 2.0
    t0_guess = time[0]

    p0 = [r0_guess, B_guess, p_guess, t0_guess]
    bounds = ([-np.inf, 0, 2, -np.inf], [np.inf, np.inf, 10.0, np.inf])

    popt, _ = curve_fit(r_model, time, radius, p0=p0, bounds=bounds, maxfev=20000)

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

    time_name = 'Instability Relative_Timing' if relative_time else 'Instability Timing'
    time = np.array(df[time_name])
    radius = np.array(df['Instability Radius'])

    sort_idx = np.argsort(time)
    time = time[sort_idx]
    radius = radius[sort_idx]

    mask = np.isfinite(time) & np.isfinite(radius)
    time = time[mask]
    radius = radius[mask]

    time_range = time
    radius_range = radius

    if negative_mask:
        mask_neg = time_range < 0
        time_range = time_range[mask_neg]
        radius_range = radius_range[mask_neg]

    if min_radius_mask and len(radius_range) > 0:
        idx_min = np.argmin(radius_range)
        time_range = time_range[:idx_min+1]
        radius_range = radius_range[:idx_min+1]

    # -------------------- FIT SELECTOR --------------------
    if fit_method == "poly":
        coef = np.polyfit(time_range, radius_range, 3)
        fit_func = np.poly1d(coef)
        t = np.linspace(np.min(time_range), np.max(time_range), 500)
        r = fit_func(t)

    elif fit_method == "exp":
        t, r = exp_radius_fit(time_range, radius_range)

    elif fit_method == "power":
        t, r = radius_fit(time_range, radius_range)

    elif fit_method == "erf":
        t, r = erf_radius_fit(time_range, radius_range)

    else:
        raise ValueError("Invalid fit_method. Use 'poly', 'power', or 'exp'.")
    # ------------------------------------------------------

    v = np.gradient(r, t) * 1e3  # km/s to m/s

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    # Scatter velocity from consecutive points (averaging 5 ns)
    t_avg, r_avg, r_err, t_scatter, v_scatter = velocity_scatter(time_range, radius_range, dt_average=5)

    # Plot
    ax1.errorbar(t_avg, r_avg, yerr=r_err, fmt='o', color='green', label='Avg radius (5 ns bins)', capsize=4)
    ax2.scatter(t_scatter, v_scatter, color='orange', label='Data velocity', s=30)


    ax1.scatter(time, radius, label="Data", s=30)
    ax1.plot(t, r, label=f"{fit_method} fit", linewidth=2)
    ax2.plot(t, v, label="Velocity", color='red')

    ax1.set_xlabel("Time")
    ax1.set_ylabel("Radius")
    ax2.set_ylabel("Velocity (m/s)")

    plt.show()
