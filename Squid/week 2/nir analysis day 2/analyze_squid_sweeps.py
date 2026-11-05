# -*- coding: utf-8 -*-
"""Run this file with F5 in Spyder. No command-line arguments needed.

Install once in the Spyder IPython console:
    %pip install numpy pandas scipy matplotlib pillow imageio-ffmpeg

Put this script beside the A0.00.csv, A0.05.csv, ... files, or change DATA_FOLDER.
CH1 is current and CH2 is voltage in the two supplied example files.
All internal analysis units are microamperes and microvolts, so uV/uA = ohms.

Two extraction methods: sustained voltage threshold and local empirical knee.
Knee model: V = V0 + R_local*max(I-I_knee, 0).
Knee window sensitivity is reported separately from statistical uncertainty.
"""
from pathlib import Path
import re
import shutil
from time import perf_counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter, PillowWriter
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from scipy.signal import find_peaks, savgol_filter

# ---------------- Edit these settings ----------------
DATA_FOLDER = Path(r"C:\Users\Nir\Documents\Courses\year 3 semester a\Lab c1\Squid\week 2\week 2 day 1\week 2\one branch")
OUTPUT_FOLDER = Path(r"C:\Users\Nir\Documents\Courses\year 3 semester a\Lab c1\Squid\week 2\week 2 day 1\week 2\one branch analysis3")
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
# Local plateau/line intersection: V = V0 + R_local*max(I-I_knee, 0).
KNEE_LEFT_FACTOR = 0.5    # Lower local fit limit / threshold Ic
KNEE_RIGHT_FACTOR = 1.25  # Upper local fit limit / threshold Ic
KNEE_I_MIN_UA = None     # Optional absolute overrides
KNEE_I_MAX_UA = None
KNEE_RANGE_FACTORS = (1.15, 1.25, 1.35)  # Check upper-limit sensitivity

POINT_SIZE = 4
LINE_WIDTH = 2
TEXT_SIZE = 12
FRAME_SECONDS = 1.0
VIDEO_FPS = 10
VIDEO_DPI = 110
MAKE_VIDEOS = True
VIDEO_METHODS = ("threshold", "knee")  # Or ("knee",) for one video
# -----------------------------------------------------


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


def knee_model(current, resistance, knee_current, offset):
    return offset + resistance*np.maximum(current-knee_current, 0)


def fit_knee_range(avg, lower, upper):
    mask = (avg.I_uA >= lower) & (avg.I_uA <= upper)
    x, y = avg.I_uA[mask].to_numpy(), avg.V_uV[mask].to_numpy()
    if len(x) < 12 or np.ptp(y) < V_THRESHOLD_UV:
        raise ValueError("Local knee range has too few bins or too little voltage rise")
    # Search every possible plateau/rising split. This avoids local minima at
    # the hinge that a single nonlinear optimizer initial guess can produce.
    candidates = list(x[1:-1])
    for cut in range(2, len(x)-1):
        slope, intercept = np.polyfit(x[cut:], y[cut:], 1)
        if slope > 0:
            knee = (np.mean(y[:cut])-intercept)/slope
            if x[cut-1] <= knee <= x[cut]:
                candidates.append(knee)
    best = None
    for knee in candidates:
        design = np.column_stack([np.maximum(x-knee, 0), np.ones(len(x))])
        resistance, offset = np.linalg.lstsq(design, y, rcond=None)[0]
        if resistance <= 0:
            continue
        par = np.array([resistance, knee, offset])
        sse = np.sum((y-knee_model(x, *par))**2)
        if best is None or sse < best[0]:
            best = sse, par
    if best is None:
        raise ValueError("Local knee fit failed")
    sse, par = best
    if min(np.sum(x < par[1]), np.sum(x > par[1])) < 4:
        raise ValueError("Knee is not bracketed by enough plateau/rising bins")
    jacobian = np.column_stack([np.maximum(x-par[1], 0),
                               -par[0]*(x > par[1]), np.ones(len(x))])
    if np.linalg.matrix_rank(jacobian) < 3:
        raise ValueError("Local knee parameters are not identifiable")
    cov = np.linalg.pinv(jacobian.T@jacobian)*sse/(len(x)-3)
    errors = np.sqrt(np.maximum(np.diag(cov), 0))
    if not np.isfinite(errors).all():
        raise ValueError("Local knee covariance is not finite")
    return dict(par=par, err=errors, low=x.min(), high=x.max(),
                rmse=np.sqrt(sse/len(x)), r2=1-sse/np.sum((y-y.mean())**2), status="ok")


def fit_knee(avg, threshold):
    if not np.isfinite(threshold) and (KNEE_I_MIN_UA is None or KNEE_I_MAX_UA is None):
        raise ValueError("No threshold seed: provide absolute KNEE_I_MIN_UA / KNEE_I_MAX_UA")
    lower = KNEE_LEFT_FACTOR*threshold if KNEE_I_MIN_UA is None else KNEE_I_MIN_UA
    upper = KNEE_RIGHT_FACTOR*threshold if KNEE_I_MAX_UA is None else KNEE_I_MAX_UA
    fit = fit_knee_range(avg, lower, upper)
    # This spread is a range-sensitivity diagnostic, NOT a statistical error.
    variants = []
    if KNEE_I_MAX_UA is None and np.isfinite(threshold):
        for factor in KNEE_RANGE_FACTORS:
            try:
                trial = fit_knee_range(avg, lower, factor*threshold)
                variants.append((factor, trial["par"][1]))
            except ValueError:
                pass
    fit["variants"] = variants
    values = [fit["par"][1]] + [v for _, v in variants]
    fit["range_min"], fit["range_max"] = min(values), max(values)
    if fit["range_max"]-fit["range_min"] > 2*fit["err"][1]:
        fit["status"] = "fit-range sensitivity exceeds statistical error"
    return fit


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
                ax.axvline(ic, color="red", ls="--", lw=LINE_WIDTH,
                           label=f"Ic = {ic:.2f} uA")
                label = f"Threshold Ic = {ic:.2f} +/- {err:.2f} uA"
            else:
                label = data["threshold_status"]
        elif method == "knee":
            fit = data["knee"]
            if fit is not None:
                xx = np.linspace(fit["low"], fit["high"], 700)
                ax.plot(xx, knee_model(xx, *fit["par"]), "r-", lw=LINE_WIDTH,
                        label="Local knee fit (fitted range only)")
                resistance, ic, v0 = fit["par"]
                er, ei, ev = fit["err"]
                ax.plot(ic, v0, "ro", ms=POINT_SIZE+3)
                label = (f"Local knee Ic = {ic:.2f} +/- {ei:.2f} uA\n"
                         f"R_local = {resistance:.3f} +/- {er:.3f} ohm\n"
                         f"RMSE = {fit['rmse']:.2f} uV\n"
                         f"Range check: {fit['range_min']:.2f} to {fit['range_max']:.2f} uA")
            else:
                label = data["knee_status"]
        else:
            raise ValueError(f"Unknown video method: {method}")
        ax.text(0.03, 0.97, label, transform=ax.transAxes, va="top", fontsize=TEXT_SIZE-1,
                bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"))
        ax.set(xlim=limits[0], ylim=limits[1], xlabel="SQUID current I (uA)",
               ylabel="Baseline-subtracted voltage V (uV)",
               title=f"V-I graph: current going {name}")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=TEXT_SIZE-2)
    axes[0].figure.suptitle(f"Coil current = {sample['coil']:.3f} A | {sample['file']}")


def save_video(samples, method, limits):
    if not samples:
        raise ValueError("No samples to render")
    if FRAME_SECONDS <= 0 or VIDEO_FPS <= 0 or VIDEO_DPI <= 0:
        raise ValueError("Video duration, frame rate and DPI must be positive")
    # A noninteractive canvas avoids repeated Spyder GUI/Plots-pane updates.
    fig = Figure(figsize=(13, 5.6))
    FigureCanvasAgg(fig)
    axes = fig.subplots(1, 2)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.14, top=0.85, wspace=0.24)
    # Keep the original timing rounded to whole output frames. Matplotlib draws
    # each sample ONCE; FFmpeg duplicates the held frames cheaply during encoding.
    hold_frames = max(1, round(FRAME_SECONDS*VIDEO_FPS))
    hold_seconds = hold_frames/VIDEO_FPS
    if not shutil.which("ffmpeg"):
        try:
            import imageio_ffmpeg
            plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            pass
    if FFMpegWriter.isAvailable():
        path = OUTPUT_FOLDER / f"iv_{method}.mp4"
        writer = FFMpegWriter(fps=1/hold_seconds, codec="libx264",
            extra_args=["-preset", "fast", "-pix_fmt", "yuv420p", "-r", str(VIDEO_FPS)])
    else:
        path = OUTPUT_FOLDER / f"iv_{method}.gif"
        writer = PillowWriter(fps=1/hold_seconds)
        print("FFmpeg unavailable: saving GIF. Install imageio-ffmpeg for MP4.")
    print(f"Rendering {method}: {len(samples)} samples, {hold_seconds:g} s each", flush=True)
    started = perf_counter()
    try:
        with writer.saving(fig, str(path), dpi=VIDEO_DPI):
            for index, sample in enumerate(samples, 1):
                make_frame(axes, sample, method, limits)
                writer.grab_frame()
                if index == 1 or index == len(samples) or index % max(1, len(samples)//10) == 0:
                    print(f"  {method}: {index}/{len(samples)} samples "
                          f"({perf_counter()-started:.1f} s elapsed)", flush=True)
    finally:
        fig.clear()
    print(f"Saved {path} ({perf_counter()-started:.1f} s)", flush=True)


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
                knee = fit_knee(avg, ic_t)
                knee_status = knee["status"]
            except ValueError as exc:
                knee, knee_status = None, str(exc)
            sample["directions"][name] = dict(average=avg, threshold=(ic_t, err_t),
                threshold_status=status_t,
                knee=knee, knee_status=knee_status)
            result = dict(file=path.name, coil_A=coil, direction=name,
                sweeps=len(sweeps[name]), voltage_baseline_uV=v_zero,
                Ic_threshold_uA=ic_t, Ic_threshold_error_uA=err_t, threshold_status=status_t)
            print(f"  {name}: threshold Ic = {ic_t:.4g} +/- {err_t:.2g} uA ({status_t})")
            result.update(Ic_knee_uA=np.nan, Ic_knee_error_uA=np.nan,
                R_local_ohm=np.nan, R_local_error_ohm=np.nan,
                V0_knee_uV=np.nan, V0_knee_error_uV=np.nan,
                knee_I_min_uA=np.nan, knee_I_max_uA=np.nan,
                knee_RMSE_uV=np.nan, knee_R2=np.nan,
                knee_range_min_uA=np.nan, knee_range_max_uA=np.nan, knee_status=knee_status)
            if knee is not None:
                r, ik, v0 = knee["par"]
                er, ei, ev = knee["err"]
                result.update(Ic_knee_uA=ik, Ic_knee_error_uA=ei,
                    R_local_ohm=r, R_local_error_ohm=er, V0_knee_uV=v0,
                    V0_knee_error_uV=ev, knee_I_min_uA=knee["low"],
                    knee_I_max_uA=knee["high"], knee_RMSE_uV=knee["rmse"],
                    knee_R2=knee["r2"], knee_range_min_uA=knee["range_min"],
                    knee_range_max_uA=knee["range_max"])
                print(f"       Local knee Ic = {ik:.4g} +/- {ei:.2g} uA; "
                      f"R_local = {r:.4g} +/- {er:.2g} ohm; "
                      f"V0 = {v0:.4g} +/- {ev:.2g} uV\n"
                      f"       Fit I = [{knee['low']:.2f}, {knee['high']:.2f}] uA; "
                      f"RMSE = {knee['rmse']:.3g} uV; R2 = {knee['r2']:.5f}\n"
                      f"       Range-check knee Ic = {knee['range_min']:.3f} to "
                      f"{knee['range_max']:.3f} uA; {knee_status}")
                for factor, value in knee["variants"]:
                    result[f"knee_upper_factor_{factor:g}_Ic_uA"] = value
            else:
                print("       Local knee:", knee_status)
            summary.append(result)
        samples.append(sample)
    table = pd.DataFrame(summary)
    table.to_csv(OUTPUT_FOLDER/"critical_currents.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    # Related hues group each method; markers/line styles distinguish directions.
    for method, direction, color, style, label in (
            ("threshold", "up", "#0072B2", "o-", "Threshold: current up"),
            ("threshold", "down", "#56B4E9", "v--", "Threshold: current down"),
            ("knee", "up", "#D55E00", "s-", "Local knee: current up"),
            ("knee", "down", "#E69F00", "^--", "Local knee: current down")):
        part = table[table.direction == direction]
        ax.errorbar(part.coil_A, part[f"Ic_{method}_uA"],
                    yerr=part[f"Ic_{method}_error_uA"], fmt=style, color=color,
                    capsize=4, ms=POINT_SIZE+1, lw=LINE_WIDTH, label=label)
    ax.set(xlabel="Coil current (A)", ylabel="Critical current Ic (uA)",
           title="Critical current vs coil current: up/down comparison")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_FOLDER/"Ic_coil_comparison.png", dpi=180)
    all_avg = [d["average"] for s in samples for d in s["directions"].values()]
    xmin = min(a.I_uA.min() for a in all_avg)
    xmax = max(a.I_uA.max() for a in all_avg)
    ymin = min(a.V_uV.min() for a in all_avg)
    ymax = max(a.V_uV.max() for a in all_avg)
    limits = ((xmin-0.03*(xmax-xmin), xmax+0.03*(xmax-xmin)),
              (min(ymin, -V_THRESHOLD_UV)-0.03*ymax, ymax*1.08))
    if MAKE_VIDEOS:
        for method in VIDEO_METHODS:
            if method not in ("threshold", "knee"):
                raise ValueError(f"Unknown video method: {method}")
            save_video(samples, method, limits)
    print("\nDone:", OUTPUT_FOLDER)
    print("Errors are approximate statistical/resolution errors, not calibration or model errors.")
    print("Up/down refer to SQUID bias current within each capture, not coil sweep direction.")
    plt.show()
    return table, samples


if __name__ == "__main__":
    results, processed_samples = main()
