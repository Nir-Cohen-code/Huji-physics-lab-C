# -*- coding: utf-8 -*-
"""Centered I-V plots and fits for the supplied oscilloscope CSVs.

Requires numpy, pandas, matplotlib. Open in Spyder, set DATA_DIR below to the
folder containing the CSVs, and press Run (F5). No command-line arguments.

Apply only the manual's conversions: current-channel V / 10,000 gives A,
and voltage-channel V / 10,000 gives sample V. No extra oscilloscope correction.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# %% Folder settings: edit these in Spyder, then press F5.
SCRIPT_DIR = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
DATA_DIR = Path('.')              # Default: CSVs beside this Python file.
# For your original folder layout, you can use:
# DATA_DIR = Path(r'..\week1 day 1 samples')
# Or use an absolute path, for example Path(r'C:\Users\YourName\samples').
OUTPUT_DIR = Path('iv_results')   # Graphs and fit tables saved beside the script.
# Relative paths above are resolved from the Python file, not Spyder's working directory.

# %% Analysis settings
window = 10
# Set independently if the current monitor or voltage gain changed between runs.
CURRENT_A_PER_V_HOT = 1e-4
CURRENT_A_PER_V_COLD = 1e-4
VOLTAGE_V_PER_V_HOT = 1e-4
VOLTAGE_V_PER_V_COLD = 1e-4

IC_VOLTAGE_CRITERION = 0.30  # Exported voltage-channel V, before gain correction.
# About 5 plateau-noise levels with window=10 in these files. Keep the voltage
# criterion fixed when changing smoothing; otherwise the definition of Ic changes.
IC_CONSECUTIVE_BINS = 3
CLIP_MARGIN = 0.05       # Remove an extra 5% of voltage span inside each rail.
COLD_FIT_MIN = 20.0      # Centered current-channel volts, chosen for THESE data.
COLD_FIT_MAX = 30.0
FIT_COV_LAGS = 100      # Time-sample lags for correlation-aware (HAC) covariance.


def line_fit(x, y, mask, intercept=True):
    """OLS parameters and 1-sigma errors, allowing time-correlated residuals.

    Newey-West/Bartlett covariance, with excluded time samples assigned zero
    scores so separated fit intervals are not treated as adjacent. Uncertainties
    are conditional on centering, chosen fit range, calibration and linear model;
    they do not include errors in the measured current or systematic errors.
    """
    x, y, mask = np.asarray(x), np.asarray(y), np.asarray(mask, dtype=bool)
    design = np.column_stack((x, np.ones_like(x))) if intercept else x[:, None]
    selected = design[mask]
    n, parameters = selected.shape
    if n <= parameters or np.linalg.matrix_rank(selected) < parameters:
        raise ValueError('Insufficient independent points for the line fit.')
    beta = np.linalg.lstsq(selected, y[mask], rcond=None)[0]
    residuals = y[mask] - selected @ beta
    scores = np.zeros_like(design, dtype=float)
    scores[mask] = selected * residuals[:, None]
    meat = scores.T @ scores
    lags = min(FIT_COV_LAGS, len(x) - 1)
    for lag in range(1, lags + 1):
        cross = scores[lag:].T @ scores[:-lag]
        meat += (1 - lag / (lags + 1)) * (cross + cross.T)
    bread = np.linalg.inv(selected.T @ selected)
    covariance = bread @ meat @ bread * n / (n - parameters)
    errors = np.sqrt(np.maximum(np.diag(covariance), 0))
    return beta, errors, covariance


def read_curve(path, cold=False):
    df = pd.read_csv(path, header=None, usecols=[4, 10])
    x, y = (df[10], df[4]) if cold else (df[4], df[10])
    # Keep the original rolling-average operation and drop its initial NaNs.
    curve = pd.DataFrame({'I': x, 'V': y}).rolling(window).mean().dropna()
    return x, y, curve


def cold_analysis(curve):
    x, y = curve['I'].to_numpy(), curve['V'].to_numpy()
    midpoint = (x.min() + x.max()) / 2
    central = np.abs(x - midpoint) < 0.06 * np.ptp(x)
    v0 = np.median(y[central])
    sigma = 1.4826 * np.median(np.abs(y[central] - v0))
    if sigma <= 1e-12:
        sigma = np.std(y[central], ddof=1)
    if sigma <= 1e-12:
        raise ValueError('Cannot estimate plateau noise; supply a voltage criterion.')

    # Medians in current bins prevent a single noisy point setting Ic.
    edges = np.linspace(x.min(), x.max(), 321)
    grouped = curve.groupby(pd.cut(x, edges, include_lowest=True), observed=True)
    binned = grouped.median().dropna()
    bx, by = binned['I'].to_numpy(), binned['V'].to_numpy() - v0

    def crossings(voltage_threshold):
        roots = []
        for sign in (-1, 1):
            side = sign * (bx - midpoint) > 0
            xx, yy = sign * (bx[side] - midpoint), sign * by[side]
            order = np.argsort(xx)
            xx, yy = xx[order], yy[order]
            above = yy > voltage_threshold
            sustained = np.convolve(above.astype(int),
                                    np.ones(IC_CONSECUTIVE_BINS, dtype=int), 'valid')
            candidates = np.flatnonzero(sustained == IC_CONSECUTIVE_BINS)
            if not len(candidates) or candidates[0] == 0:
                raise ValueError('No resolved plateau-to-resistive transition found.')
            j = candidates[0]
            root = np.interp(voltage_threshold,
                             yy[j-1:j+1], xx[j-1:j+1])
            roots.append(midpoint + sign * root)
        return np.array(roots)

    roots = crossings(IC_VOLTAGE_CRITERION)
    # This centering assumes equal positive/negative critical-current magnitudes.
    # The data alone cannot distinguish an I offset from physical asymmetry.
    i0 = roots.mean()
    ic = (roots[1] - roots[0]) / 2
    sensitivity = [(crossings(f * IC_VOLTAGE_CRITERION)[1]
                    - crossings(f * IC_VOLTAGE_CRITERION)[0]) / 2 for f in (0.6, 1.4)]
    centered = curve - pd.Series({'I': i0, 'V': v0})
    # Fit one V = R*I line to both outer branches; exclude the middle region.
    mask = ((centered['I'].abs() >= COLD_FIT_MIN)
            & (centered['I'].abs() <= COLD_FIT_MAX)
            & (centered['I'].abs() > ic))
    for sign in (-1, 1):
        if (mask & (sign * centered['I'] > 0)).sum() < 10:
            raise ValueError('Too few cold fit points; adjust COLD_FIT_MIN/MAX.')
    fit_i = centered.loc[mask, 'I'].to_numpy()
    fit_v = centered.loc[mask, 'V'].to_numpy()
    slope = np.dot(fit_i, fit_v) / np.dot(fit_i, fit_i)
    return centered, i0, v0, sigma, ic, sensitivity, (slope, mask)


def main(data_dir=None, output_dir=None):
    p = Path(DATA_DIR if data_dir is None else data_dir).expanduser()
    out = Path(OUTPUT_DIR if output_dir is None else output_dir).expanduser()
    if not p.is_absolute():
        p = SCRIPT_DIR / p
    if not out.is_absolute():
        out = SCRIPT_DIR / out
    if not p.is_dir():
        raise FileNotFoundError(f'CSV folder not found: {p}\nEdit DATA_DIR at the top of this file.')
    out.mkdir(parents=True, exist_ok=True)
    print(f'Reading CSVs from: {p.resolve()}')
    print(f'Saving graphs and fit results to: {out.resolve()}')
    hot_path = p / 'iv room temp.csv'
    if not hot_path.exists():
        hot_path = p / 'iv room temp(1).csv'
    x_raw, y_raw, hot = read_curve(hot_path)

    # Detect rails from the original voltage, BEFORE smoothing. Reject any
    # rolling window containing a point near either rail, including clip edges.
    lower, upper = y_raw.min(), y_raw.max()
    margin = CLIP_MARGIN * (upper - lower)
    raw_good = (y_raw > lower + margin) & (y_raw < upper - margin)
    good = raw_good.rolling(window).min().reindex(hot.index).eq(1)
    hot_beta, hot_errors, hot_cov = line_fit(hot['I'], hot['V'], good)
    r_hot, intercept = hot_beta
    # A straight line determines V0 - R*I0, not both offsets independently.
    # Assume symmetric current-drive extrema to choose I0, then evaluate V0.
    i0_hot = (x_raw.min() + x_raw.max()) / 2
    v0_hot = intercept + r_hot * i0_hot
    hot = hot - pd.Series({'I': i0_hot, 'V': v0_hot})

    _, _, cold_raw = read_curve(p / 'iv cooled more xy flipped.csv', cold=True)
    cold, i0, v0, sigma, ic, sensitivity, cold_fit = cold_analysis(cold_raw)
    r_cold, cold_mask = cold_fit
    combined_beta, combined_errors, _ = line_fit(
        cold['I'], cold['V'], cold_mask, intercept=False)
    r_cold = combined_beta[0]
    right_mask = cold_mask & (cold['I'] > 0)
    right_beta, right_errors, _ = line_fit(cold['I'], cold['V'], right_mask)
    r_right, b_right = right_beta
    summary = [dict(dataset='room temperature', I_offset_channel_V=i0_hot,
                    V_offset_channel_V=v0_hot, slope_V_per_V=r_hot,
                    slope_error_V_per_V=hot_errors[0],
                    intercept_channel_V=intercept,
                    intercept_error_channel_V=hot_errors[1],
                    fit_coordinates='uncentered channel values',
                    Ic_channel_V=np.nan)]
    print(f'Hot offsets: I-channel={i0_hot:.6g} V, V-channel={v0_hot:.6g} V')
    print(f'Hot slope (unclipped): {r_hot:.6g} V/V; {good.sum()} points used')
    print(f'Cold offsets: I-channel={i0:.6g} V, V-channel={v0:.6g} V')
    print(f'Cold plateau noise: {sigma:.6g} V; criterion: '
          f'{IC_VOLTAGE_CRITERION:.6g} V')
    print(f'Cold Ic: {ic:.6g} current-channel V; 0.6-to-1.4 criterion '
          f'sensitivity: {sensitivity[0]:.6g} to {sensitivity[1]:.6g} V')

    fig, axes = plt.subplots(1, 3, figsize=(20, 5), layout='constrained')
    ax_hot, ax_cold, ax_right = axes
    ih = CURRENT_A_PER_V_HOT or 1.0
    ic_scale = CURRENT_A_PER_V_COLD or 1.0
    vh, vc = VOLTAGE_V_PER_V_HOT, VOLTAGE_V_PER_V_COLD
    # Display in microamperes and microvolts; fit parameters remain in SI units.
    ih_plot = ih * 1e6
    ic_plot = ic_scale * 1e6
    vh_plot = vh * 1e6
    vc_plot = vc * 1e6
    ax_hot.plot(hot['I'] * ih_plot, hot['V'] * vh_plot, '.', color='tab:blue', ms=2,
                label='Room-temperature data')
    # Fit only the unclipped data, but draw the line over the full current range.
    xx = np.linspace(hot['I'].min(), hot['I'].max(), 200)
    slope_text = (f'R = {r_hot*vh/ih:.4g} Ω' if CURRENT_A_PER_V_HOT
                  else f'slope = {r_hot:.4g} V/V')
    ax_hot.plot(xx * ih_plot, r_hot * xx * vh_plot, 'r-', label=f'V = RI; {slope_text}')
    ax_hot.set_title('Room temperature')

    for ax in (ax_cold, ax_right):
        ax.plot(cold['I'] * ic_plot, cold['V'] * vc_plot, '.', ms=2,
                label='Cooled more: centered data')
    xx = np.linspace(cold['I'].min(), cold['I'].max(), 300)
    value = r_cold * vc / ic_scale
    ax_cold.plot(xx * ic_plot, r_cold * xx * vc_plot,
                 color='tab:orange', linewidth=2,
                 label=f'V = RI; R = {value:.4g} Ω')
    print(f'Cold combined outer-branch slope: {r_cold:.6g} V/V; '
          f'{cold_mask.sum()} points used; middle excluded')
    summary.append(dict(dataset='cooled more combined outer branches',
                        I_offset_channel_V=i0, V_offset_channel_V=v0,
                        slope_V_per_V=r_cold,
                        slope_error_V_per_V=combined_errors[0],
                        intercept_channel_V=0.0, intercept_error_channel_V=np.nan,
                        fit_coordinates='centered channel values; intercept fixed to zero',
                        Ic_channel_V=ic))
    # Separate local slope of the right branch; allow a nonzero intercept.
    xx_right = np.linspace(cold.loc[right_mask, 'I'].min(), cold['I'].max(), 150)
    ax_right.plot(xx_right * ic_plot, (r_right * xx_right + b_right) * vc_plot,
                 color='tab:purple', linewidth=2,
                 label=f'Right branch: V = RI + b; R = {r_right*vc/ic_scale:.4g} Ω')
    summary.append(dict(dataset='cooled more right branch',
                        I_offset_channel_V=i0, V_offset_channel_V=v0,
                        slope_V_per_V=r_right, slope_error_V_per_V=right_errors[0],
                        intercept_channel_V=b_right,
                        intercept_error_channel_V=right_errors[1],
                        fit_coordinates='centered channel values', Ic_channel_V=ic))
    for ax in (ax_cold, ax_right):
        for sign in (-1, 1):
            point_i = sign * ic * ic_plot
            point_v = sign * IC_VOLTAGE_CRITERION * vc_plot
            ax.plot(point_i, point_v, 'ro', ms=5, zorder=5)
            ax.annotate(f'I = {point_i:+.3g} µA', (point_i, point_v),
                        xytext=(sign * 12, -sign * 18), textcoords='offset points',
                        ha='left' if sign > 0 else 'right',
                        va='top' if sign > 0 else 'bottom', color='crimson',
                        fontsize=9)
    ax_cold.set_title('Cooled more — fit to both outer branches')
    ax_right.set_title('Cooled more — right-branch fit only')

    for ax in axes:
        ax.set_xlabel('Current, I (µA)')
        ax.set_ylabel('Voltage, V (µV)')
        ax.axhline(0, color='0.4', linewidth=0.6)
        ax.axvline(0, color='0.4', linewidth=0.6)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='upper left')
    fig.savefig(out / 'iv_centered_fits.png', dpi=180)
    # Save the two cooled graphs separately as well as the three-panel overview.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig.set_layout_engine(None)  # Freeze positions while hiding neighboring axes.
    for ax, filename in ((ax_cold, 'iv_cooled_combined_fit.png'),
                         (ax_right, 'iv_cooled_right_fit.png')):
        bounds = ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
        for other in axes:
            other.set_visible(other is ax)
        fig.savefig(out / filename, dpi=180, bbox_inches=bounds.expanded(1.04, 1.06))
        for other in axes:
            other.set_visible(True)

    # Analyze the second cooled acquisition as a check, without mixing its
    # slightly different transition shape into the main fit.
    other_path = p / 'iv cooled xy flipped.csv'
    if other_path.exists():
        _, _, other_raw = read_curve(other_path, cold=True)
        _, other_i0, other_v0, _, other_ic, other_sensitivity, _ = cold_analysis(other_raw)
        print(f'Other cooled file: Ic={other_ic:.6g} current-channel V; '
              f'criterion sensitivity={other_sensitivity}')
        summary.append(dict(dataset='other cooled (check)', I_offset_channel_V=other_i0,
                            V_offset_channel_V=other_v0, slope_V_per_V=np.nan,
                            Ic_channel_V=other_ic))
    print(f'\nFIT PARAMETERS: value ± 1-sigma statistical uncertainty '
          f'(HAC covariance, {FIT_COV_LAGS} sample lags)')
    print(f'Room temperature, before centering: V = R I + b')
    print(f'  R = {r_hot*vh/ih:.6g} ± {hot_errors[0]*abs(vh/ih):.2g} ohm')
    print(f'  b = {intercept*vh_plot:.6g} ± {hot_errors[1]*abs(vh_plot):.2g} µV')
    offset_vector = np.array([i0_hot, 1.0])
    v0_hot_error = np.sqrt(max(offset_vector @ hot_cov @ offset_vector, 0))
    print(f'  Removed voltage offset = {v0_hot*vh_plot:.6g} ± '
          f'{v0_hot_error*abs(vh_plot):.2g} µV (current offset held fixed)')
    print('Cooled combined outer branches: V = R I')
    print(f'  R = {r_cold*vc/ic_scale:.6g} ± '
          f'{combined_errors[0]*abs(vc/ic_scale):.2g} ohm')
    print('  b = 0 µV (fixed by the model, not fitted)')
    print(f'Cooled right branch, {COLD_FIT_MIN*ic_plot:g} to '
          f'{COLD_FIT_MAX*ic_plot:g} µA: V = R I + b')
    print(f'  R = {r_right*vc/ic_scale:.6g} ± '
          f'{right_errors[0]*abs(vc/ic_scale):.2g} ohm')
    print(f'  b = {b_right*vc_plot:.6g} ± {right_errors[1]*abs(vc_plot):.2g} µV')
    print(f'Ic = {ic*ic_plot:.6g} µA; criterion-sensitivity range '
          f'{sensitivity[0]*ic_plot:.6g} to {sensitivity[1]*ic_plot:.6g} µA '
          '(not a statistical confidence interval).')
    print('Fit uncertainties exclude calibration, centering (except the hot voltage '
          'offset above), current-axis errors and deviations from the linear model.')
    # Print complete equations directly from Python, with I in µA and V in µV.
    equations = [
        'Fitted equations: V in µV, I in µA; ± values are 1-sigma statistical errors.',
        f'Room temperature (centered): V = ({r_hot*vh/ih:.6g} ± '
        f'{hot_errors[0]*abs(vh/ih):.2g}) I',
        f'Cooled, both outer branches: V = ({r_cold*vc/ic_scale:.6g} ± '
        f'{combined_errors[0]*abs(vc/ic_scale):.2g}) I',
        f'Cooled, right branch only: V = ({r_right*vc/ic_scale:.6g} ± '
        f'{right_errors[0]*abs(vc/ic_scale):.2g}) I + '
        f'({b_right*vc_plot:.6g} ± {right_errors[1]*abs(vc_plot):.2g})',
        'Uncertainties exclude calibration, centering and linear-model errors.'
    ]
    print('\n' + '\n'.join(equations))
    (out / 'iv_fit_parameters.txt').write_text('\n'.join(equations) + '\n', encoding='utf-8')
    for row in summary:
        is_hot = row['dataset'] == 'room temperature'
        current_scale = CURRENT_A_PER_V_HOT if is_hot else CURRENT_A_PER_V_COLD
        voltage_scale = VOLTAGE_V_PER_V_HOT if is_hot else VOLTAGE_V_PER_V_COLD
        row['R_ohm'] = row['slope_V_per_V'] * voltage_scale / current_scale
        row['R_error_ohm'] = row.get('slope_error_V_per_V', np.nan) * abs(voltage_scale / current_scale)
        row['intercept_uV'] = row.get('intercept_channel_V', np.nan) * voltage_scale * 1e6
        row['intercept_error_uV'] = row.get('intercept_error_channel_V', np.nan) * abs(voltage_scale) * 1e6
        row['uncertainty_method'] = f'1-sigma HAC; {FIT_COV_LAGS} sample lags; fixed centering/calibration'
        row['Ic_A'] = row['Ic_channel_V'] * current_scale
    pd.DataFrame(summary).to_csv(out / 'iv_fit_results.csv', index=False)
    plt.show()


if __name__ == '__main__':
    main()
