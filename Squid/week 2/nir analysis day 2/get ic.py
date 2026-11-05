# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 16:49:45 2026

@author: cohen
"""

# -*- coding: utf-8 -*-
"""Run this file with F5 in Spyder. No command-line arguments needed.

Install once in the Spyder IPython console:
    %pip install numpy pandas scipy matplotlib pillow imageio-ffmpeg

Put this script beside the A0.00.csv, A0.05.csv, ... files, or change DATA_FOLDER.
CH1 is current and CH2 is voltage in the two supplied example files.
All internal analysis units are microamperes and microvolts, so uV/uA = ohms.

Model: V = V0 + R_SQUID*sqrt(max(I**2-Ic**2, 0)) for positive current.
R_SQUID = R_junction/2 for identical parallel junctions. This is an ideal,
noise-free, overdamped RSJ approximation. Noise rounding, finite inductance,
and hysteresis can bias Ic; the covariance errors do not include that bias.
Sources: Mr. SQUID User Guide sections 3.5, 3.7, 4.3;
https://arxiv.org/abs/1608.02325 (small-inductance limit, eq. 3).
"""
from pathlib import Path
import re
import shutil
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter, PillowWriter
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import curve_fit, OptimizeWarning

# ---------------- Edit these settings ----------------
DATA_FOLDER = Path(r"C:\Users\Nir\Documents\Courses\year 3 semester a\Lab c1\Squid\week 2\week 2 day 1\week 2\one branch")
OUTPUT_FOLDER = Path(r"C:\Users\Nir\Documents\Courses\year 3 semester a\Lab c1\Squid\week 2\week 2 day 1\week 2\one branch analysis")
CURRENT_COLUMN = 4          # Zero-based: CH1 voltage values
VOLTAGE_COLUMN = 10         # Zero-based: CH2 voltage values
CURRENT_UA_PER_V = 100.0    # Manual: 1 V current monitor = 100 uA
VOLTAGE_GAIN = 10000.0      # Actual SQUID volts = monitor volts / gain
CURRENT_ZERO_MONITOR_V = 0.0  # Enter a measured zero-current monitor offset
VOLTAGE_ZERO_MONITOR_V = None # None: median of lowest-current 15% per file
VOLTAGE_POLARITY = 1        # Use -1 only if voltage cable polarity is reversed

V_THRESHOLD_UV = 2.0       # Lower this to obtain a lower threshold Ic
THRESHOLD_CONSECUTIVE_BINS = 3
BIN_WIDTH_UA = 1.0
ROLLING_WINDOW = 3         # 1 disables smoothing; threshold/display only
DIRECTION_WINDOW = 31      # Odd Savitzky-Golay window, raw samples
TURNAROUND_GUARD = 8       # Exclude raw samples around maxima/minima
MIN_SWEEP_FRACTION = 0.8   # Reject incomplete sweeps spanning less than this
FIT_MAX_FACTOR = 1.5       # Fit I <= this * threshold Ic (avoid far-tail curvature)
FIT_I_MIN_UA = None        # Override fit bounds if needed
FIT_I_MAX_UA = None        # e.g. 110.0; None uses FIT_MAX_FACTOR

POINT_SIZE = 4
LINE_WIDTH = 2
TEXT_SIZE = 12
FRAME_SECONDS = 1.0
VIDEO_FPS = 10
VIDEO_DPI = 110
# -----------------------------------------------------


def rsj(current, resistance, critical_current, offset):
    return offset + resistance * np.sqrt(np.maximum(current**2-critical_current**2, 0))


def read_file(path):
    # OpenChoice metadata lives in other columns; the waveform starts at row 0.
    df = pd.read_csv(path, header=None)
    raw_i = pd.to_numeric(df[CURRENT_COLUMN], errors="coerce").to_numpy()
    raw_v = pd.to_numeric(df[VOLTAGE_COLUMN], errors="coerce").to_numpy()
    good = np.isfinite(raw_i) & np.isfinite(raw_v)
    rows = np.flatnonzero(good)
    current = (raw_i[good]-CURRENT_ZERO_MONITOR_V)*CURRENT_UA_PER_V
    voltage = VOLTAGE_POLARITY*raw_v[good]*1e6/VOLTAGE_GAIN
    if len(current) < 100 or np.ptp(current) == 0:
        raise ValueError("Too few samples or no current sweep")
    # Do not center current from its extrema: these captures are unipolar.
    if VOLTAGE_ZERO_MONITOR_V is None:
        low_current = current <= np.quantile(current, 0.15)
        if np.ptp(np.percentile(voltage[low_current], [5, 95])) > 2*V_THRESHOLD_UV:
            raise ValueError("Lowest-current data lack a clear voltage plateau: set a measured "
                             "VOLTAGE_ZERO_MONITOR_V; do not infer zero from a resistive trace")
        voltage_zero = np.median(voltage[low_current])
    else:
        voltage_zero = VOLTAGE_POLARITY*VOLTAGE_ZERO_MONITOR_V*1e6/VOLTAGE_GAIN
    return rows, current, voltage-voltage_zero, voltage_zero, len(df)


def separate_sweeps(current):
    w = min(DIRECTION_WINDOW, len(current)//2*2-1)
    w = max(5, w if w % 2 else w-1)
    smooth_i = savgol_filter(current, w, 2)
    span = np.ptp(smooth_i)
    maxima = find_peaks(smooth_i, prominence=0.4*span, distance=w)[0]
    minima = find_peaks(-smooth_i, prominence=0.4*span, distance=w)[0]
    edges = np.unique(np.r_[0, maxima, minima, len(current)-1])
    direction = np.full(len(current), "excluded", dtype=object)
    sweep_id = np.full(len(current), -1, dtype=int)
    sweeps = {"up": [], "down": []}
    for start, stop in zip(edges[:-1], edges[1:]):
        delta = smooth_i[stop]-smooth_i[start]
        if abs(delta) < MIN_SWEEP_FRACTION*span:
            continue
        name = "up" if delta > 0 else "down"
        ix = np.arange(start+TURNAROUND_GUARD, stop-TURNAROUND_GUARD+1)
        if len(ix) < 10:
            continue
        sid = len(sweeps[name])
        sweeps[name].append(ix)
        direction[ix] = name
        sweep_id[ix] = sid
    if any(len(sweeps[k]) == 0 for k in sweeps):
        raise ValueError("No complete up/down sweeps; check channel mapping/direction settings")
    return sweeps, direction, sweep_id


def average_sweeps(current, voltage, sweeps):
    pieces = []
    for sid, ix in enumerate(sweeps):
        part = pd.DataFrame({"I_uA": current[ix], "V_uV": voltage[ix]})
        part["bin"] = np.floor(part.I_uA/BIN_WIDTH_UA).astype(int)
        # First average within each sweep, then give repeated sweeps equal weight.
        binned = part.groupby("bin").agg(I_uA=("I_uA", "mean"),
                                          V_uV=("V_uV", "mean"),
                                          n_raw=("V_uV", "size"))
        pieces.append(binned.assign(sweep=sid).reset_index())
    out = pd.concat(pieces).groupby("bin").agg(
        I_uA=("I_uA", "mean"), V_uV=("V_uV", "mean"),
        V_sem_uV=("V_uV", "sem"), n_sweeps=("sweep", "nunique"),
        n_raw=("n_raw", "sum")).reset_index(drop=True)
    out = out.sort_values("I_uA").reset_index(drop=True)
    out["V_smooth_uV"] = out.V_uV.rolling(ROLLING_WINDOW, center=True,
                                           min_periods=1).mean()
    return out


def threshold_ic(avg):
    i, v = avg.I_uA.to_numpy(), avg.V_smooth_uV.to_numpy()
    above = v >= V_THRESHOLD_UV
    n = THRESHOLD_CONSECUTIVE_BINS
    for k in range(len(i)-n+1):
        if not above[k:k+n].all():
            continue
        if k == 0:
            return np.nan, np.nan, "already above threshold at smallest current"
        # Sustained first crossing, interpolated between the adjacent bins.
        if v[k] <= v[k-1]:
            continue
        ic = np.interp(V_THRESHOLD_UV, v[k-1:k+1], i[k-1:k+1])
        slope = (v[k]-v[k-1])/(i[k]-i[k-1])
        sem = np.nanmax(np.r_[avg.V_sem_uV.iloc[k-1:k+1], 0.0])
        uncertainty = np.hypot((i[k]-i[k-1])/2, sem/abs(slope))
        return ic, uncertainty, "ok"
    return np.nan, np.nan, "no sustained threshold crossing"


def fit_ic(avg, threshold):
    # Fit the UNSMOOTHED bin averages to avoid extra rolling-window correlations.
    x, y = avg.I_uA.to_numpy(), avg.V_uV.to_numpy()
    estimate = threshold if np.isfinite(threshold) else np.median(x)
    lower = x.min() if FIT_I_MIN_UA is None else FIT_I_MIN_UA
    upper = min(x.max(), FIT_MAX_FACTOR*estimate) if FIT_I_MAX_UA is None else FIT_I_MAX_UA
    mask = (x >= lower) & (x <= upper) & (x >= 0)
    xf, yf = x[mask], y[mask]
    if len(xf) < 12 or np.ptp(yf) < 2*V_THRESHOLD_UV:
        raise ValueError("Fit range does not contain enough of the transition")
    r_guess = max((yf[-1]-yf[0])/max(np.ptp(xf), 1e-9), 1e-6)
    # Multiple initial Ic values reduce sensitivity to the sharp model knee.
    best = None
    for start_ic in np.unique(np.clip(estimate*np.array([0.7, 0.85, 1, 1.1]),
                                      0.01*xf.max(), 0.95*xf.max())):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", OptimizeWarning)
                par, cov = curve_fit(rsj, xf, yf, p0=[r_guess, start_ic, 0],
                                    bounds=([0, 0, -np.inf], [np.inf, xf.max(), np.inf]),
                                    maxfev=20000)
            residual = yf-rsj(xf, *par)
            sse = np.sum(residual**2)
            if best is None or sse < best[0]:
                best = sse, par, cov, residual
        except (RuntimeError, ValueError, OptimizeWarning):
            pass
    if best is None:
        raise ValueError("RSJ fit failed for all initial guesses")
    sse, par, cov, residual = best
    errors = np.sqrt(np.maximum(np.diag(cov), 0))
    r2 = 1-sse/np.sum((yf-yf.mean())**2)
    rmse = np.sqrt(np.mean(residual**2))
    status = "ok"
    if not np.isfinite(errors).all() or par[1] < 0.01*xf.max() or par[1] > 0.98*xf.max():
        status = "Ic poorly constrained or near fit boundary"
    elif r2 < 0.98:
        status = "check RSJ residuals/model mismatch"
    return dict(par=par, err=errors, low=xf.min(), high=xf.max(),
                rmse=rmse, r2=r2, status=status)


def make_frame(axes, sample, method, limits):
    for ax, name in zip(axes, ("up", "down")):
        ax.clear()
        data = sample["directions"][name]
        avg = data["average"]
        ax.plot(avg.I_uA, avg.V_uV, ".", ms=POINT_SIZE, color="tab:blue",
                label="Mean of repeated sweeps")
        if ROLLING_WINDOW > 1:
            ax.plot(avg.I_uA, avg.V_smooth_uV, color="tab:blue", lw=1,
                    alpha=0.6, label=f"Rolling mean ({ROLLING_WINDOW} bins)")
        if method == "threshold":
            ax.axhline(V_THRESHOLD_UV, color="green", ls="--", lw=LINE_WIDTH,
                       label=f"Voltage threshold = {V_THRESHOLD_UV:g} uV")
            ic, err = data["threshold"]
            if np.isfinite(ic):
                ax.plot(ic, V_THRESHOLD_UV, "ro", ms=POINT_SIZE+3)
                label = f"Threshold Ic = {ic:.2f} +/- {err:.2f} uA"
            else:
                label = data["threshold_status"]
        else:
            fit = data["fit"]
            if fit is not None:
                xx = np.linspace(fit["low"], fit["high"], 700)
                ax.plot(xx, rsj(xx, *fit["par"]), "r-", lw=LINE_WIDTH,
                        label="RSJ fit (fitted range only)")
                resistance, ic, v0 = fit["par"]
                er, ei, ev = fit["err"]
                ax.plot(ic, v0, "ro", ms=POINT_SIZE+3)
                label = (f"RSJ Ic = {ic:.2f} +/- {ei:.2f} uA\n"
                         f"R_SQUID = {resistance:.3f} +/- {er:.3f} ohm\n"
                         f"RMSE = {fit['rmse']:.2f} uV; {fit['status']}")
            else:
                label = data["fit_status"]
        ax.text(0.03, 0.97, label, transform=ax.transAxes, va="top", fontsize=TEXT_SIZE-1,
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))
        ax.set(xlim=limits[0], ylim=limits[1], xlabel="SQUID current I (uA)",
               ylabel="Baseline-subtracted voltage V (uV)",
               title=f"V-I graph: current going {name}")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=TEXT_SIZE-2)
    axes[0].figure.suptitle(f"Coil current = {sample['coil']:.3f} A | {sample['file']}")


def save_video(samples, method, limits):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.14, top=0.85, wspace=0.24)
    repeats = max(1, round(FRAME_SECONDS*VIDEO_FPS))
    def update(frame):
        make_frame(axes, samples[min(frame//repeats, len(samples)-1)], method, limits)
    animation = FuncAnimation(fig, update, frames=len(samples)*repeats,
                              interval=1000/VIDEO_FPS, blit=False)
    if not shutil.which("ffmpeg"):
        try:
            import imageio_ffmpeg
            plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            pass
    if FFMpegWriter.isAvailable():
        path = OUTPUT_FOLDER / f"iv_{method}.mp4"
        animation.save(path, writer=FFMpegWriter(fps=VIDEO_FPS, codec="libx264",
                       extra_args=["-pix_fmt", "yuv420p"]), dpi=VIDEO_DPI)
    else:
        path = OUTPUT_FOLDER / f"iv_{method}.gif"
        animation.save(path, writer=PillowWriter(fps=VIDEO_FPS), dpi=VIDEO_DPI)
        print("FFmpeg unavailable: saved GIF. Install imageio-ffmpeg for MP4.")
    plt.close(fig)
    print("Saved", path)


def main():
    if CURRENT_UA_PER_V <= 0 or VOLTAGE_GAIN <= 0 or BIN_WIDTH_UA <= 0:
        raise ValueError("Calibration factors and bin width must be positive")
    if ROLLING_WINDOW < 1 or V_THRESHOLD_UV <= 0 or THRESHOLD_CONSECUTIVE_BINS < 1:
        raise ValueError("Rolling window, threshold and consecutive bins must be positive")
    files = []
    for path in DATA_FOLDER.glob("*.csv"):
        match = re.fullmatch(r"A([+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+))", path.stem, re.I)
        if match:
            files.append((float(match[1]), path))
    files.sort(key=lambda z: z[0])
    if not files:
        raise FileNotFoundError(f"No A<number>.csv files found in {DATA_FOLDER}")
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": TEXT_SIZE})
    samples, summary = [], []
    for coil, path in files:
        rows, current, voltage, v_zero, nrows = read_file(path)
        sweeps, direction, ids = separate_sweeps(current)
        # Raw CSV row indices are zero-based, matching df.iloc[index].
        indices = pd.DataFrame({"row_index": np.arange(nrows), "direction": "excluded",
                                "sweep_id": -1})
        indices.loc[rows, "direction"] = direction
        indices.loc[rows, "sweep_id"] = ids
        indices.to_csv(OUTPUT_FOLDER/f"{path.stem}_indices.csv", index=False)
        sample = dict(coil=coil, file=path.name, directions={})
        print(f"\n{path.name}: {len(sweeps['up'])} up / {len(sweeps['down'])} down sweeps; "
              f"subtracted voltage baseline = {v_zero:.4g} uV")
        for name in ("up", "down"):
            avg = average_sweeps(current, voltage, sweeps[name])
            avg.to_csv(OUTPUT_FOLDER/f"{path.stem}_{name}_averaged.csv", index=False)
            ic_t, err_t, status_t = threshold_ic(avg)
            try:
                fit = fit_ic(avg, ic_t)
                fit_status = fit["status"]
            except ValueError as exc:
                fit, fit_status = None, str(exc)
            sample["directions"][name] = dict(average=avg, threshold=(ic_t, err_t),
                threshold_status=status_t, fit=fit, fit_status=fit_status)
            result = dict(file=path.name, coil_A=coil, direction=name,
                sweeps=len(sweeps[name]), voltage_baseline_uV=v_zero,
                Ic_threshold_uA=ic_t, Ic_threshold_error_uA=err_t, threshold_status=status_t,
                Ic_fit_uA=np.nan, Ic_fit_error_uA=np.nan, R_SQUID_ohm=np.nan,
                R_SQUID_error_ohm=np.nan, R_junction_ohm=np.nan, R_junction_error_ohm=np.nan,
                V0_fit_uV=np.nan, V0_fit_error_uV=np.nan, fit_I_min_uA=np.nan,
                fit_I_max_uA=np.nan, fit_RMSE_uV=np.nan, fit_R2=np.nan, fit_status=fit_status)
            print(f"  {name}: threshold Ic = {ic_t:.4g} +/- {err_t:.2g} uA ({status_t})")
            if fit is not None:
                r, ic, v0 = fit["par"]
                er, ei, ev = fit["err"]
                result.update(Ic_fit_uA=ic, Ic_fit_error_uA=ei, R_SQUID_ohm=r,
                    R_SQUID_error_ohm=er, R_junction_ohm=2*r, R_junction_error_ohm=2*er,
                    V0_fit_uV=v0, V0_fit_error_uV=ev, fit_I_min_uA=fit["low"],
                    fit_I_max_uA=fit["high"], fit_RMSE_uV=fit["rmse"], fit_R2=fit["r2"])
                print(f"       RSJ Ic = {ic:.4g} +/- {ei:.2g} uA; "
                      f"R_SQUID = {r:.4g} +/- {er:.2g} ohm; "
                      f"R_junction = {2*r:.4g} +/- {2*er:.2g} ohm; "
                      f"V0 = {v0:.4g} +/- {ev:.2g} uV\n"
                      f"       Fit I = [{fit['low']:.2f}, {fit['high']:.2f}] uA; "
                      f"RMSE = {fit['rmse']:.3g} uV; R2 = {fit['r2']:.5f}; {fit_status}")
            else:
                print("       RSJ:", fit_status)
            summary.append(result)
        samples.append(sample)
    table = pd.DataFrame(summary)
    table.to_csv(OUTPUT_FOLDER/"critical_currents.csv", index=False)
    for name in ("up", "down"):
        part = table[table.direction == name]
        fig, ax = plt.subplots(figsize=(8, 5))
        for method, color in (("threshold", "tab:blue"), ("fit", "red")):
            ax.errorbar(part.coil_A, part[f"Ic_{method}_uA"],
                        yerr=part[f"Ic_{method}_error_uA"], fmt="o-", color=color,
                        capsize=4, ms=POINT_SIZE+1, lw=LINE_WIDTH, label=method)
        ax.set(xlabel="Coil current (A)", ylabel="Critical current Ic (uA)",
               title=f"Critical current vs coil current: SQUID current going {name}")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUTPUT_FOLDER/f"Ic_coil_{name}.png", dpi=180)
    all_avg = [d["average"] for s in samples for d in s["directions"].values()]
    xmin = min(a.I_uA.min() for a in all_avg)
    xmax = max(a.I_uA.max() for a in all_avg)
    ymin = min(a.V_uV.min() for a in all_avg)
    ymax = max(a.V_uV.max() for a in all_avg)
    limits = ((xmin-0.03*(xmax-xmin), xmax+0.03*(xmax-xmin)),
              (min(ymin, -V_THRESHOLD_UV)-0.03*ymax, ymax*1.08))
    for method in ("threshold", "fit"):
        save_video(samples, method, limits)
    print("\nDone:", OUTPUT_FOLDER)
    print("Errors are approximate statistical/resolution errors, not calibration or RSJ-model errors.")
    print("Up/down refer to SQUID bias current within each capture, not coil sweep direction.")
    plt.show()
    return table, samples


if __name__ == "__main__":
    results, processed_samples = main()
