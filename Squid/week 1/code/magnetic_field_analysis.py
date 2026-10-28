#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 16:00:57 2026

@author: lianachagin
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

p = "../week1 day 2 samples/magnetic field mapping/"

# Ic parameters
V_threshold = 1
consecutive_points = 50
baseline_points = 200

x_arr = np.array([])
Ic_arr = np.array([])

for x in np.arange(0, 22.5, 0.5):

    clean_x = round(x, 1)

    if clean_x.is_integer():
        filename = f"x{int(clean_x)}.csv"
    else:
        filename = f"x{clean_x}.csv"

    filepath = p + filename

    if not os.path.exists(filepath):
        continue

    df = pd.read_csv(filepath, header=None, usecols=[3, 4, 9, 10])

    # Manual conversion: divide by 10^4, then multiply by 10^6 for micro-units.
    window = 100
    I = df[4].rolling(window).mean() * 100
    V = df[10].rolling(window).mean() * 100

    # Remove NaN values
    valid_data = pd.DataFrame({"I": I, "V": V}).dropna().reset_index(drop=True)

    if len(valid_data) < baseline_points + consecutive_points:
        continue

    I_vals = valid_data["I"].values
    V_vals = valid_data["V"].values

    # Finding Ic
    V_baseline = np.mean(V_vals[:baseline_points])

    condition = V_vals > (V_baseline + V_threshold)

    Ic_found = None
    for idx in range(len(condition) - consecutive_points + 1):
        if np.all(condition[idx : idx + consecutive_points]):
            Ic_found = I_vals[idx]
            break

    if Ic_found is not None:
        x_arr = np.append(x_arr, clean_x)
        Ic_arr = np.append(Ic_arr, Ic_found)

    # Plot individual I-V curve
    plt.figure()
    plt.plot(I, V, "-", label=f"x={clean_x}")
    if Ic_found is not None:
        plt.axvline(
            x=Ic_found,
            color="r",
            linestyle="--",
            label=f"Ic = {Ic_found:.2f}",
        )
    plt.xlabel(r"I [$\mu$A]")
    plt.ylabel(r"V [$\mu$V]")
    plt.title(f"I-V Curve for x = {clean_x}")
    plt.legend()
    plt.grid(True)
    plt.show()


# Sort extracted arrays explicitly by position x

if len(x_arr) > 0:
    sort_indices = np.argsort(x_arr)
    x_arr = x_arr[sort_indices]
    Ic_arr = Ic_arr[sort_indices]

# Plot normalized Ic vs position x

if len(Ic_arr) > 0:

    # Normalizing
    Ic_max = np.max(Ic_arr)
    Ic_min = np.min(Ic_arr)
    delta_Ic = Ic_max - Ic_min
    Ic_norm = (Ic_arr - Ic_min) / delta_Ic

    # Plotting
    plt.figure()
    plt.plot(
        x_arr,
        Ic_norm,
        "o-",
        color="b",
        label=r"$I_c / I_{c,\mathrm{max}}$",
    )
    plt.xlabel(r"$X_{\mathrm{magnet}}$ [mm]")
    plt.ylabel(r"$I_{c,\mathrm{normalized}}$")
    plt.title(r"$I_c$ VS Magnet Distance")
    plt.grid(True)
    plt.show()


# ---------------------------------------------------------
# 3D Plot: V - I - X (Filtered for 60 <= I <= 80 uA)
# ---------------------------------------------------------

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection="3d")

for x in np.arange(0, 22.5, 0.5):

    clean_x = round(x, 1)

    if clean_x.is_integer():
        filename = f"x{int(clean_x)}.csv"
    else:
        filename = f"x{clean_x}.csv"

    filepath = p + filename

    if not os.path.exists(filepath):
        continue

    df = pd.read_csv(filepath, header=None, usecols=[3, 4, 9, 10])

    window = 100
    I = df[4].rolling(window).mean() * 100
    V = df[10].rolling(window).mean() * 100

    valid_data = pd.DataFrame({"I": I, "V": V}).dropna().reset_index(drop=True)

    # Filter current range: 60 <= I <= 80 uA
    filtered_data = valid_data[(valid_data["I"] >= 60) & (valid_data["I"] <= 80)]

    if len(filtered_data) > 0:
        I_filt = filtered_data["I"].values
        V_filt = filtered_data["V"].values
        # Create constant X array matching current position x
        X_filt = np.full_like(I_filt, clean_x)

        # Plot 3D line for current x position in black
        ax.plot(X_filt, I_filt, V_filt, color="black", linewidth=1)

# Set I-axis ticks explicitly to multiples of 5 (60, 65, 70, 75, 80)
ax.set_yticks(np.arange(60, 85, 5))

# Axis labels and title using safe LaTeX formatting
ax.set_xlabel(r"$X_{\mathrm{magnet}}$ [mm]")
ax.set_ylabel(r"I [$\mu$A]")
ax.set_zlabel(r"V [$\mu$V]")
ax.set_title("I-V-X Profile")
plt.show()


# ---------------------------------------------------------
# Data Collection for Constant I and Constant V Slices
# ---------------------------------------------------------

all_x = []
all_I = []
all_V = []

for x in np.arange(0, 22.5, 0.5):
    clean_x = round(x, 1)
    
    # Exclude measurement x = 14.5 from slices
    if clean_x == 14.5:
        continue

    filename = f"x{int(clean_x)}.csv" if clean_x.is_integer() else f"x{clean_x}.csv"
    filepath = p + filename

    if not os.path.exists(filepath):
        continue

    df = pd.read_csv(filepath, header=None, usecols=[3, 4, 9, 10])
    window = 100
    I = df[4].rolling(window).mean() * 100
    V = df[10].rolling(window).mean() * 100
    valid_data = pd.DataFrame({"I": I, "V": V}).dropna().reset_index(drop=True)

    all_x.append(np.full(len(valid_data), clean_x))
    all_I.append(valid_data["I"].values)
    all_V.append(valid_data["V"].values)

if len(all_x) > 0:
    full_x = np.concatenate(all_x)
    full_I = np.concatenate(all_I)
    full_V = np.concatenate(all_V)

    # Determine common limits across all data for identical scaling
    x_lims = (np.min(full_x) - 0.5, np.max(full_x) + 0.5)

    # ---------------------------------------------------------
    # Subplots 1: Constant Currents I = [60, 70, 80] uA (Unified Scale)
    # ---------------------------------------------------------
    const_I_targets = [60, 70, 80]
    
    # Pre-calculate global V min/max for constant I slices
    I_masks = [np.abs(full_I - t) < 0.5 for t in const_I_targets]
    all_sliced_V = full_V[np.logical_or.reduce(I_masks)]
    v_lims_I = (np.min(all_sliced_V) - 1, np.max(all_sliced_V) + 1)

    fig_I, axes_I = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    fig_I.suptitle(r"V vs X at Constant Currents ($I$) [Unified Scale]", fontsize=14)

    for idx, target_I in enumerate(const_I_targets):
        ax_sub = axes_I[idx]
        mask = I_masks[idx]
        
        if np.any(mask):
            sub_x = full_x[mask]
            sub_V = full_V[mask]
            
            sort_id = np.argsort(sub_x)
            ax_sub.plot(sub_x[sort_id], sub_V[sort_id], "o-", color="black")

        ax_sub.set_title(rf"$I \approx {target_I} \ \mu\mathrm{{A}}$")
        ax_sub.set_xlabel(r"$X_{\mathrm{magnet}}$ [mm]")
        ax_sub.set_ylabel(r"V [$\mu$V]")
        ax_sub.set_xlim(x_lims)
        ax_sub.set_ylim(v_lims_I)
        ax_sub.grid(True)

    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------
    # Subplots 2: Constant Voltages V = [6, 10, 14] uV (Unified Scale)
    # ---------------------------------------------------------
    const_V_targets = [6, 10, 14]
    
    # Pre-calculate global I min/max for constant V slices
    V_masks = [np.abs(full_V - t) < 0.3 for t in const_V_targets]
    all_sliced_I = full_I[np.logical_or.reduce(V_masks)]
    i_lims_V = (np.min(all_sliced_I) - 1, np.max(all_sliced_I) + 1)

    fig_V, axes_V = plt.subplots(1, 3, figsize=(15, 5), sharex=True, sharey=True)
    fig_V.suptitle(r"I vs X at Constant Voltages ($V$) [Unified Scale]", fontsize=14)

    for idx, target_V in enumerate(const_V_targets):
        ax_sub = axes_V[idx]
        mask = V_masks[idx]
        
        if np.any(mask):
            sub_x = full_x[mask]
            sub_I = full_I[mask]
            
            sort_id = np.argsort(sub_x)
            ax_sub.plot(sub_x[sort_id], sub_I[sort_id], "o-", color="blue")

        ax_sub.set_title(rf"$V \approx {target_V} \ \mu\mathrm{{V}}$")
        ax_sub.set_xlabel(r"$X_{\mathrm{magnet}}$ [mm]")
        ax_sub.set_ylabel(r"I [$\mu$A]")
        ax_sub.set_xlim(x_lims)
        ax_sub.set_ylim(i_lims_V)
        ax_sub.grid(True)

    plt.tight_layout()
    plt.show()

    # ---------------------------------------------------------
    # Combined Plot 1: All Constant Currents (I = 60, 70, 80 uA) in One Plot
    # ---------------------------------------------------------
    colors_I = ["crimson", "darkorange", "teal"]
    
    plt.figure(figsize=(9, 6))
    for idx, target_I in enumerate(const_I_targets):
        mask = I_masks[idx]
        if np.any(mask):
            sub_x = full_x[mask]
            sub_V = full_V[mask]
            sort_id = np.argsort(sub_x)
            plt.plot(
                sub_x[sort_id],
                sub_V[sort_id],
                "o-",
                color=colors_I[idx],
                label=rf"Constant $I = {target_I} \ \mu\mathrm{{A}}$"
            )

    plt.xlabel(r"$X_{\mathrm{magnet}}$ [mm]")
    plt.ylabel(r"V [$\mu$V]")
    plt.title(r"V vs X for Various Constant Currents ($I$)")
    plt.xlim(x_lims)
    plt.ylim(v_lims_I)
    plt.legend()
    plt.grid(True)
    plt.show()

    # ---------------------------------------------------------
    # Combined Plot 2: All Constant Voltages (V = 6, 10, 14 uV) in One Plot
    # ---------------------------------------------------------
    colors_V = ["royalblue", "purple", "forestgreen"]
    
    plt.figure(figsize=(9, 6))
    for idx, target_V in enumerate(const_V_targets):
        mask = V_masks[idx]
        if np.any(mask):
            sub_x = full_x[mask]
            sub_I = full_I[mask]
            sort_id = np.argsort(sub_x)
            plt.plot(
                sub_x[sort_id],
                sub_I[sort_id],
                "o-",
                color=colors_V[idx],
                label=rf"Constant $V = {target_V} \ \mu\mathrm{{V}}$"
            )

    plt.xlabel(r"$X_{\mathrm{magnet}}$ [mm]")
    plt.ylabel(r"I [$\mu$A]")
    plt.title(r"I vs X for Various Constant Voltages ($V$)")
    plt.xlim(x_lims)
    plt.ylim(i_lims_V)
    plt.legend()
    plt.grid(True)
    plt.show()