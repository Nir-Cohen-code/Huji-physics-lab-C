# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 10:32:03 2026

@author: cohen
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

point_size = 8
line_width = 2
text_size = 16
fit_extension = 400  # µA: extend the right-branch fit line at both ends.

plt.rcParams.update({"font.size": text_size, "axes.titlesize": text_size,
                     "axes.labelsize": text_size, "xtick.labelsize": text_size,
                     "ytick.labelsize": text_size, "legend.fontsize": text_size,
                     "lines.markersize": point_size, "lines.linewidth": line_width})

p = "..\\week1 day 1 samples\\"
df1 = pd.read_csv(p+"iv room temp.csv", header=None, usecols=[3, 4, 9, 10])
df2 = pd.read_csv(p+"iv cooled xy flipped.csv", header=None, usecols=[3, 4, 9, 10])
df3 = pd.read_csv(p+"iv cooled more xy flipped.csv", header=None, usecols=[3, 4, 9, 10])

window = 10
# Manual conversion: divide by 10^4, then multiply by 10^6 for micro-units.
I_hot = df1[4].rolling(window).mean() * 100
V_hot = df1[10].rolling(window).mean() * 100
I_cold = df3[10].rolling(window).mean() * 100
V_cold = df3[4].rolling(window).mean() * 100

# Fit non-overlapping averages so shared samples do not inflate the point count.
fit_points = np.arange(len(df1)) % window == window - 1
cold_fit_points = np.arange(len(df3)) % window == window - 1

# Center the hot current, and exclude clipped voltage windows from the fit.
I_hot -= 50 * (df1[4].min() + df1[4].max())
margin = 0.05 * (df1[10].max() - df1[10].min())
unclipped = df1[10].between(df1[10].min()+margin, df1[10].max()-margin)
valid = fit_points & unclipped.rolling(window).min().eq(1) & I_hot.notna()
(R_hot, V0), cov = np.polyfit(I_hot[valid], V_hot[valid], 1, cov=True)
dR_hot, dV0 = np.sqrt(np.diag(cov))
V_hot -= V0

print(f"Room temperature: R = {R_hot:.5g} ± {dR_hot:.2g} ohm")
print(f"Removed hot voltage offset = {V0:.5g} ± {dV0:.2g} µV")

plt.figure()
plt.plot(I_hot, V_hot, '.', label="Room-temperature data")
I_line = np.linspace(I_hot.min(), I_hot.max(), 200)
plt.plot(I_line, R_hot*I_line, 'r-', label=f"V = RI; R = {R_hot:.4g} Ω")
plt.xlabel("Current, I (µA)")
plt.ylabel("Voltage, V (µV)")
plt.title("V-I graph: room temperature")
plt.grid(True, alpha=0.4)
plt.legend()
plt.tight_layout()

# Center the cooled voltage using the plateau.
V_cold -= V_cold[I_cold.abs() < 300].median()
V_threshold = 20  # µV: lower values mark an earlier onset; keep above plateau noise.

# Bin by current and require three consecutive bins above the voltage threshold.
cold = pd.DataFrame({"I": I_cold, "V": V_cold}).dropna()
bins = cold.groupby((cold["I"]/20).round()).median().reset_index(drop=True)
crossings = []
for sign in [-1, 1]:
    branch = bins[sign*bins["I"] > 0].sort_values("I", ascending=(sign > 0))
    above = (sign*branch["V"] >= V_threshold).rolling(3).sum().eq(3)
    found = np.flatnonzero(above.to_numpy())
    if not len(found) or found[0] < 3:
        raise ValueError("No resolved Ic crossing; check V_threshold.")
    j = found[0] - 2
    crossings.append(np.interp(V_threshold, sign*branch["V"].iloc[j-1:j+1],
                              branch["I"].iloc[j-1:j+1]))

I_cold -= np.mean(crossings)  # Assume symmetric positive/negative critical currents.
Ic = (crossings[1] - crossings[0]) / 2
print(f"Ic ≈ {Ic:.4g} µA (voltage criterion: {V_threshold} µV)")

# Combined fit: V = RI, using only 2000 <= |I| <= 3000 µA.
valid = cold_fit_points & I_cold.abs().between(2000, 3000) & V_cold.notna()
x = I_cold[valid].to_numpy()
y = V_cold[valid].to_numpy()
R_cold = np.dot(x, y) / np.dot(x, x)
dR_cold = np.sqrt(np.sum((y-R_cold*x)**2) / (len(x)-1) / np.dot(x, x))

# Right-branch fit: V = RI + b, using 2000 <= I <= 3000 µA.
(R_right, b_right), cov = np.polyfit(x[x > 0], y[x > 0], 1, cov=True)
dR_right, db_right = np.sqrt(np.diag(cov))

print(f"Cooled combined fit: R = {R_cold:.5g} ± {dR_cold:.2g} ohm")
print(f"Cooled right fit: R = {R_right:.5g} ± {dR_right:.2g} ohm")
print(f"Cooled right fit: b = {b_right:.5g} ± {db_right:.2g} µV")
print("V is in µV and I in µA in these fitted equations:")
print(f"Hot:   V = ({R_hot:.5g} ± {dR_hot:.2g}) I")
print(f"Both:  V = ({R_cold:.5g} ± {dR_cold:.2g}) I")
print(f"Right: V = ({R_right:.5g} ± {dR_right:.2g}) I + ({b_right:.5g} ± {db_right:.2g})")
print("± values are approximate 1σ regression errors; calibration/model errors excluded.")

# Two separate cooled figures.
for right_only in [False, True]:
    plt.figure()
    plt.plot(I_cold, V_cold, '.', label="Cooled data")
    if right_only:
        I_line = np.linspace(x[x > 0].min()-fit_extension,
                             I_cold.max()+fit_extension, 200)
        plt.plot(I_line, R_right*I_line+b_right, 'r-',
                 label=f"V = RI + b; R = {R_right:.4g} Ω")
    else:
        I_line = np.linspace(I_cold.min(), I_cold.max(), 200)
        plt.plot(I_line, R_cold*I_line, 'r-',
                 label=f"V = RI; R = {R_cold:.4g} Ω")
    for sign in [-1, 1]:
        plt.plot(sign*Ic, sign*V_threshold, 'ro', markersize=point_size+1)
        plt.annotate(f"I = {sign*Ic:+.3g} µA", (sign*Ic, sign*V_threshold),
                     xytext=(sign*12, -sign*18), textcoords="offset points",
                     ha="left" if sign > 0 else "right",
                     va="top" if sign > 0 else "bottom", color="red")
    plt.xlabel("Current, I (µA)")
    plt.ylabel("Voltage, V (µV)")
    plt.title("V-I graph: cooled, right-branch fit" if right_only
              else "V-I graph: cooled, combined fit")
    plt.grid(True, alpha=0.4)
    plt.legend()
    plt.tight_layout()

plt.show()
