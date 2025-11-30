import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
import matplotlib
matplotlib.use("TkAgg")
from scipy.optimize import curve_fit
from scipy.special import erf, erfinv

# -------------------- USER OPTIONS --------------------
negative_mask = False
min_radius_mask = True
fit_method = "erf_model"    # options: "poly", "power", "exp", "erf","erf_model"
save = True
plot = True
make_title = False
use_radius_err = True
# ------------------------------------------------------

"""
Radius–vs–time analysis and velocity extraction tool.

This script loads a CSV containing time and radius measurements,
optionally with radius uncertainties, then applies a selected radius-fit
model (poly, power, exp, erf, or erf_model). It outputs plots and a .npz
file containing the fitted radius, velocity, and uncertainties. The user
selects the CSV via a file-dialog window.

--------------------
USER INPUT OPTIONS
--------------------
negative_mask : bool
    If True, only keep data with time < 0.

min_radius_mask : bool
    If True, truncate data at the minimum radius point.

fit_method : str
    Select the radius model to fit:
        "poly"       – cubic polynomial
        "power"      – r = r0 – B*(t - t0)^p
        "exp"        – negative exponential
        "erf"        – error-function transition model
        "erf_model"  – analytic erf-based model with full uncertainty propagation

save : bool
    If True, save output plot and fit .npz file into /plots and /fits subfolders.

plot : bool
    If True, display plot.

make_title : bool
    If True, auto-generate a plot title from the CSV filename pattern.

use_radius_err : bool
    If True, include the radius error column (if present) in the erf_model fit.

--------------------
EXPECTED CSV CONTENTS
--------------------
• One column containing "time" or "timing"
• One column containing "radius"
• Optional: a column containing "err", "error", "std", or "sem"
"""

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
    vel = dr / dt * 1e3   # mm/ns → km/s
    t_vel = averaged_time[:-1] + dt/2

    return averaged_time, averaged_radius, std_radius, t_vel, vel

def erf_model(time_range, radius_range, radius_err=None, num_points=500):
    mask = np.isfinite(time_range) & np.isfinite(radius_range)
    t_data = np.array(time_range)[mask]
    r_data = np.array(radius_range)[mask]

    if radius_err is not None:
        sigma = np.array(radius_err)[mask]
    else:
        sigma = None

    R0_guess = 18 #np.max(r_data)
    Delta_guess = np.abs(np.max(t_data))

    def r_model(t, R0, Delta):
        return R0*np.exp(-(erfinv(t/Delta))**2)

    p0 = [R0_guess, Delta_guess]


    popt, pcov = curve_fit(
        r_model, t_data, r_data,
        p0=p0,
        sigma=sigma,
        absolute_sigma=(sigma is not None),
        maxfev=50000
    )
    perr = np.sqrt(np.diag(pcov))

    t_fit = np.linspace(np.min(t_data), np.max(t_data), num_points)
    r_fit = r_model(t_fit, *popt)

    J = np.zeros((len(t_fit),len(popt)))
    eps = 1e-8
    for i in range(len(popt)):
        p_eps = popt.copy()
        p_eps[i] += eps
        J[:,i] = (r_model(t_fit, *p_eps) - r_fit) / eps

    r_var = np.sum(J @ pcov * J, axis=1)
    r_std = np.sqrt(r_var)

    def v_model(t, R0, Delta):
        return -R0*np.sqrt(np.pi)/Delta*erfinv(t/Delta)*1e3

    v_fit = v_model(t_fit, *popt)

    R0, Delta = popt
    inv = erfinv(t_fit / Delta)
    exp_term = np.exp(inv**2)

    dv_dR0 = -np.sqrt(np.pi) / Delta * inv * 1e3
    dv_dDelta = R0 * np.sqrt(np.pi) * 1e3 * (
        (1 / Delta**2) * inv + (t_fit * np.sqrt(np.pi) / (2 * Delta**3)) * exp_term
    )

    Jv = np.vstack([dv_dR0, dv_dDelta]).T
    v_var = np.sum(Jv @ pcov * Jv, axis=1)
    v_std = np.sqrt(v_var)

    print(R0, Delta)

    return t_fit, r_fit, r_std, perr, v_fit, v_std


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

    popt, pcov = curve_fit(r_model, t_data, r_data, p0=p0, bounds=bounds, maxfev=20000)

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

    timing_cols = [c for c in df.columns if "time" in c.lower() or "timing" in c.lower()]
    if not timing_cols:
        raise ValueError("No column containing 'time' or 'timing' found in CSV.")
    time_name = timing_cols[0]
    print(f"✅ Using time column: {time_name}")
    time = np.array(df[time_name])

    radius_cols = [col for col in df.columns if 'radius' in col.lower()]
    if not radius_cols:
        raise ValueError("No column containing 'radius' found in CSV.")
    if len(radius_cols) > 1:
        print(f"Multiple radius-like columns found, using '{radius_cols[0]}' by default.")
    radius_name = radius_cols[0]
    radius = np.array(df[radius_name])
    print(f"✅ Using radius column: {radius_name}")

    # ---- radius error detection ---
    err_keywords = ["err", "error", "std", "sem"]
    radius_err = None
    err_cols = [c for c in df.columns if any(k in c.lower() for k in err_keywords)]
    if err_cols:
        err_col = err_cols[0]
        radius_err = np.array(df[err_col])
        print(f"✅ Using radius uncertainty column: {err_col}")
    else:
        print("No radius uncertainty column")

    sort_idx = np.argsort(time)
    time = time[sort_idx]
    radius = radius[sort_idx]
    if radius_err is not None:
        radius_err = radius_err[sort_idx]

    mask = np.isfinite(time) & np.isfinite(radius)
    time = time[mask]
    radius = radius[mask]
    if radius_err is not None:
        radius_err = radius_err[mask]

    time_range = time
    radius_range = radius
    radius_err_range = radius_err

    if negative_mask:
        mask_neg = time_range < 0
        time_range = time_range[mask_neg]
        radius_range = radius_range[mask_neg]
        if radius_err_range is not None:
            radius_err_range = radius_err_range[mask_neg]

    if min_radius_mask and len(radius_range) > 0:
        idx_min = np.argmin(radius_range)
        time_range = time_range[:idx_min+1]
        radius_range = radius_range[:idx_min+1]
        if radius_err_range is not None:
            radius_err_range = radius_err_range[:idx_min+1]

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
    elif fit_method == "erf_model":
        if not use_radius_err:
            radius_err_range = None
            print("Not using radius error bars in fit")
        t, r, r_std, perr, v, v_std = erf_model(time_range, radius_range, radius_err_range)

    else:
        raise ValueError("Invalid fit_method. Use 'poly', 'power', or 'exp'.")
    # ------------------------------------------------------

    if fit_method != "erf_model":
        v = np.gradient(r, t) * 1e3  # mm/ns → km/s


    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    # Scatter velocity from consecutive points (averaging 5 ns)
    t_avg, r_avg, r_err, t_scatter, v_scatter = velocity_scatter(time_range, radius_range, dt_average=5)

    # Plot
    #ax1.errorbar(t_avg, r_avg, yerr=r_err, fmt='o', color='green', label='Avg radius (5 ns bins)', capsize=4)
    #ax2.scatter(t_scatter, v_scatter, color='orange', label='Data velocity', s=30)
    fit_label = fit_method
    if fit_method == "erf_model":
        fit_label = "ERF model"

    ax1.plot(t, r, label=f"{fit_label} radius fit", linewidth=2)
    if fit_method == "erf_model":
        ax1.fill_between(t, r - r_std, r + r_std, color='b', alpha=0.3, label='Fit 1σ uncertainty')

    ax2.plot(t, v, label="Velocity", color='red')
    if fit_method == "erf_model":
        ax2.fill_between(t, v - v_std, v + v_std, color='red', alpha=0.2, label='Velocity 1σ uncertainty')

    if radius_err is not None:
        ax1.errorbar(
            time,
            radius,
            yerr=radius_err,
            fmt='o',
            markersize=np.sqrt(30),     # match scatter s=30
            markerfacecolor='blue',     # match scatter color
            markeredgecolor='blue',
            ecolor='blue',              # error bar color
            capsize=3,
            linestyle='none',
            label="Experimental radius data"
        )
    else:
        ax1.scatter(time, radius, label="Experimental radius data", s=30)

    ax1.set_xlabel("Time [ns]")
    ax1.set_ylabel("Radius [mm]")
    ax2.set_ylabel("Velocity [km/s]")
    # Cap velocity y-axis
    v_min = np.min(v - v_std)
    v_max = np.max(v + v_std)
    v_max_cap = np.max(v) + 50  # maximum 50 km/s above the max velocity
    v_min_cap = np.min(v) - 50  # optional, max 50 km/s below the min velocity

    ax2.set_ylim(max(v_min, v_min_cap), min(v_max, v_max_cap))

    # -------------------- LEGEND WITHOUT UNCERTAINTY BANDS --------------------
    # Collect handles from BOTH axes
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()

    handles = h1 + h2
    labels = l1 + l2

    filtered_handles = []
    filtered_labels = []

    for h, l in zip(handles, labels):
        if "uncertainty" in l.lower():   # exclude 1σ bands
            continue
        filtered_handles.append(h)
        filtered_labels.append(l)

    # Place combined legend on radius axis
    ax1.legend(filtered_handles, filtered_labels, loc="best")
    # --------------------------------------------------------------------------

    base_name = os.path.splitext(os.path.basename(f))[0]

    parts = base_name.split('-')

    title = None
    try:
        if len(parts) == 2:
            psi, timing = parts
            title = f"{psi} psi, -{timing} us"
        elif len(parts) == 4:
            psi1, timing1, psi2, timing2 = parts
            title = f"Liner: {psi1} psi, {timing1} ns | Target: {psi2} psi, {timing2} ns"
    except Exception:
        title = base_name  # fallback

    if title and make_title:
        plt.title(title)

    if save:
        base_dir = os.path.dirname(f)

        # Create subfolders if missing
        plot_dir = os.path.join(base_dir, "plots")
        fit_dir = os.path.join(base_dir, "fits")
        os.makedirs(plot_dir, exist_ok=True)
        os.makedirs(fit_dir, exist_ok=True)

        # Save plot
        plot_path = os.path.join(plot_dir, f"{base_name}_{fit_method}_fit.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        print(f"✅ Plot saved to: {plot_path}")

        # Save fit data to npz
        fit_path = os.path.join(fit_dir, f"{base_name}_{fit_method}_fit.npz")

        # Store whatever arrays exist safely
        fit_data = {"t": t, "r": r, "v": v}
        if fit_method == "erf_model":
            fit_data.update({"r_std": r_std, "v_std": v_std, "param_err": perr})
        np.savez(fit_path, **fit_data)
        print(f"✅ Fit data saved to: {fit_path}")

    if plot:
        plt.show()
