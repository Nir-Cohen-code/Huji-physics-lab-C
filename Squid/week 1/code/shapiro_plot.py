import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

p = "../week1 day 2 samples/shapiro/"

df = pd.read_csv(p + "shapiro4.csv", header=None, usecols=[3, 4, 9, 10])

window = 10
# Manual conversion: divide by 10^4, then multiply by 10^6 for micro-units.
I = df[4].rolling(window).mean() * 100
V = df[10].rolling(window).mean() * 100

# --- Clean NaN values resulting from rolling mean and convert to arrays ---
mask_not_null = I.notnull() & V.notnull()
I_clean = I[mask_not_null].values
V_clean = V[mask_not_null].values

# Sort data by current I
sort_idx = np.argsort(I_clean)
I_sorted = I_clean[sort_idx]
V_sorted = V_clean[sort_idx]

# --- Step detection parameters (adjustable based on noise level) ---
TOLERANCE = 6      # Maximum allowed voltage deviation from step mean
MIN_POINTS = 100   # Minimum number of consecutive points to form a step

# --- Step Detection ---
steps = []
n = len(V_sorted)
i = 0

while i < n:
    current_step_V = [V_sorted[i]]
    current_step_I = [I_sorted[i]]
    j = i + 1

    while j < n:
        proposed_V = current_step_V + [V_sorted[j]]
        mean_V = np.mean(proposed_V)

        # Check if new and existing points remain within allowed range from mean
        if np.all(np.abs(proposed_V - mean_V) <= TOLERANCE):
            current_step_V.append(V_sorted[j])
            current_step_I.append(I_sorted[j])
            j += 1
        else:
            break

    # Save step if it contains enough data points
    if len(current_step_V) >= MIN_POINTS:
        v_avg = np.mean(current_step_V)
        i_min = np.min(current_step_I)
        i_max = np.max(current_step_I)

        steps.append({
            'V_mean': v_avg,
            'I_min': i_min,
            'I_max': i_max,
            'Delta_I': i_max - i_min,
            'num_points': len(current_step_V)
        })
        i = j  # Jump to the end of the detected step
    else:
        i += 1

# Convert to DataFrame and display results
df_steps = pd.DataFrame(steps)

print("Detected Shapiro Steps:")
print(df_steps)

# --- Merge close steps (up to a 10 V threshold) ---
V_DIFF_THRESHOLD = 10.0  # Maximum voltage distance for step merging

merged_steps = []

if len(steps) > 0:
    # Sort detected steps by their mean voltage
    steps_sorted = sorted(steps, key=lambda x: x['V_mean'])

    current_group = [steps_sorted[0]]

    for next_step in steps_sorted[1:]:
        # Check if the voltage of the next step is close to the current group mean
        group_v_mean = np.mean([s['V_mean'] for s in current_group])

        if abs(next_step['V_mean'] - group_v_mean) <= V_DIFF_THRESHOLD:
            current_group.append(next_step)
        else:
            # Summarize current group and start a new group
            total_points = sum(s['num_points'] for s in current_group)
            weighted_v_mean = sum(s['V_mean'] * s['num_points'] for s in current_group) / total_points

            merged_steps.append({
                'V_mean': weighted_v_mean,
                'I_min': min(s['I_min'] for s in current_group),
                'I_max': max(s['I_max'] for s in current_group),
                'Delta_I': max(s['I_max'] for s in current_group) - min(s['I_min'] for s in current_group),
                'num_points': total_points
            })
            current_group = [next_step]

    # Merge final group
    total_points = sum(s['num_points'] for s in current_group)
    weighted_v_mean = sum(s['V_mean'] * s['num_points'] for s in current_group) / total_points
    merged_steps.append({
        'V_mean': weighted_v_mean,
        'I_min': min(s['I_min'] for s in current_group),
        'I_max': max(s['I_max'] for s in current_group),
        'Delta_I': max(s['I_max'] for s in current_group) - min(s['I_min'] for s in current_group),
        'num_points': total_points
    })

# Convert to DataFrame
df_merged_steps = pd.DataFrame(merged_steps)

# --- Print merged steps results ---
print("\nShapiro Steps after Merging:")
print(df_merged_steps.to_string(index=False))

# --- Plotting ---
plt.figure(figsize=(9, 5))

# Plot raw experimental data points
plt.plot(I, V, '.', alpha=0.4)

# Draw red horizontal line for each merged step
for idx, row in df_merged_steps.iterrows():
    plt.hlines(
        y=row['V_mean'],
        xmin=row['I_min'],
        xmax=row['I_max'],
        color='red',
        linewidth=2.5,
        label='Shapiro Step' if idx == 0 else ""
    )

plt.xlabel("I [$\mu$A]")
plt.ylabel("V [$\mu$V]")
plt.title("I-V Measurement Under 44GHz Field")
plt.legend()
plt.grid(True)
plt.show()


# --- Calculate consecutive step differences and mean difference ---

# 1. Sort by mean voltage
df_merged_steps = df_merged_steps.sort_values(by='V_mean').reset_index(drop=True)

# 2. Calculate voltage difference between consecutive steps (Delta V in microV)
df_merged_steps['Delta_V'] = df_merged_steps['V_mean'].diff()

# 3. Print differences between consecutive step pairs
print("\nVoltage differences between consecutive steps:")
for i in range(1, len(df_merged_steps)):
    v_diff = df_merged_steps.loc[i, 'Delta_V']
    v_prev = df_merged_steps.loc[i - 1, 'V_mean']
    v_curr = df_merged_steps.loc[i, 'V_mean']
    print(f"Step {i} to Step {i+1}: Delta V = {v_diff:.4f} microV (from {v_prev:.4f} microV to {v_curr:.4f} microV)")

# 4. Extract differences (excluding initial NaN value)
delta_v_series = df_merged_steps['Delta_V'].dropna()

# --- Calculate Josephson constant K_J = 2e/h and its uncertainty ---

f_ext_Hz = 44e9       # External frequency in Hz (44 GHz)
f_ext_GHz = 44.0      # External frequency in GHz

# Mean voltage difference (in microV and V) and Standard Error of the Mean (SE)
mean_delta_v_microV = delta_v_series.mean()
mean_delta_v_V = mean_delta_v_microV * 1e-6

n_diffs = len(delta_v_series)
std_delta_v = delta_v_series.std() if n_diffs > 1 else 0.0

se_delta_v_microV = std_delta_v / np.sqrt(n_diffs) if n_diffs > 0 else 0.0
se_delta_v_V = se_delta_v_microV * 1e-6

# 1. Calculation in GHz/microV
K_J_exp_GHz_microV = f_ext_GHz / mean_delta_v_microV
K_J_err_GHz_microV = K_J_exp_GHz_microV * (se_delta_v_microV / mean_delta_v_microV) if mean_delta_v_microV != 0 else 0.0
K_J_theory_GHz_microV = 0.48359785  # GHz/microV

# 2. Calculation in Hz/V
K_J_exp_Hz_V = f_ext_Hz / mean_delta_v_V
K_J_err_Hz_V = K_J_exp_Hz_V * (se_delta_v_V / mean_delta_v_V) if mean_delta_v_V != 0 else 0.0
K_J_theory_Hz_V = 483597.8484e9  # Hz/V

# --- Output Results ---

print(f"\nExternal Frequency: {f_ext_GHz:.1f} GHz ({f_ext_Hz:.2e} Hz)")
print(f"Mean Voltage Difference (Delta V): {mean_delta_v_microV:.4f} microV +/- {se_delta_v_microV:.4f} microV ({mean_delta_v_V:.6e} V)")

print("\n--- Results in GHz/microV ---")
print(f"Calculated Josephson Constant (K_J): {K_J_exp_GHz_microV:.6f} +/- {K_J_err_GHz_microV:.6f} GHz/microV")
print(f"Theoretical Accepted Value (K_J):  {K_J_theory_GHz_microV:.6f} GHz/microV")

print("\n--- Results in Hz/V ---")
print(f"Calculated Josephson Constant (K_J): {K_J_exp_Hz_V:.6e} +/- {K_J_err_Hz_V:.6e} Hz/V")
print(f"Theoretical Accepted Value (K_J):  {K_J_theory_Hz_V:.6e} Hz/V")