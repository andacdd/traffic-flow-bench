# Task 2 leakage audit (2026-10-19, src/queue_diag2.py step 1)

Question per item: can it touch data with timestamp > T (forecast origin) for (a) dev windows and (b) validation/private windows?
"Row T" = the 5-minute row at the origin itself (docs/SCORING_SPEC.md: "The released row at `T` itself may also be used").

Verified facts used below (reports/queue/maskmimic_checks.csv, all 8 panels, 80+80 official windows, 80 train windows):
- window_history rows are exactly T-60..T-5 (12 rows); the same rows in mainline_states_masked are bit-identical to window_history (0 differing cells, 0 pct differences).
- History is not Task-1-masked: 0 history cells with pct_observed > 0 are NaN (pct 1-74: 0, pct >= 75: 0). pct 1-74 cells keep their values in history and outside it.
- Row T in validation/private is Task-1-masked at the day's regime rate (mean NaN share of eligible cells 0.33 val / 0.31 private / 0.34 train); pct 1-74 cells at row T are never blanked.
- Horizon + 60 min buffer (T+5..T+90, 18 rows) is fully blank in validation/private (share 1.0), not blanked in train masked (0.42 = Task-1 mask only).
- No window history overlaps another window's horizon/buffer (0 of 160). About half of windows have an earlier same-day window, so its 18-row blank lies in "earlier hours" only.
- Task-1 mask gap-length distributions are identical across train/validation/private for every regime (geometric, e.g. R3 run-1 share 0.606/0.606/0.605); regime day shares similar (train 717/756/711, val 82/82/84, private 86/93/61 panel-days R1/R2/R3).

## Features (src/task2_model.py `features`)

| Feature group | Code | Dev (hybrid input) | Val/private (masked file) |
|---|---|---|---|
| History stats r_mean, r_min, r_slope, fc_mean, fc_max, fc_slope, o_mean, q_hist | L68 `hs, hf, ho = S[t0 - HIST:t0], F[t0 - HIST:t0], O[t0 - HIST:t0]` | rows T-60..T-5 from unmasked train (`td.hybrid` L19 `day[:, t0 - tm.HIST:t0] = full[:, t0 - tm.HIST:t0]`). <= T, same content as release | identical rows from masked file (= window_history). No leak |
| Row-T features r_last, fc_last, o_last, q_now, sT_missing, neighbour shifts, r_min_dn5/up5 | L69 `sT, fT, oT = S[t0].copy(), ...`; fallback `last_valid(hs)` | row T from train masked file (Task-1 mask of that day) | row T from val masked file (Task-1 mask). Same mechanism and rate. No leak |
| corr_q_frac, corr_fc_mean, corr_r_mean | L103-105 `np.nanmean(q_now)` etc. | from row T / T-5 fallback | same. No leak |
| day_min_r | L90 `day_min = np.nanmin(S[:t0 + 1], 0) / vf` | rows 0..T: Task-1-masked train rows before T-60, unmasked history, masked row T. No timestamp > T | rows 0..T of masked file; additionally the 18-row horizon+buffer blanks of earlier same-day windows. No leak; slightly less information in val (mismatch, not leak) |
| freq_t0, freq_max_h, freq_h | L106-107 `freq[t0]`, `freq[t0 + 1:t0 + H + 1].max(0)`; queue_cv L115-116 `tr = ~np.isin(months, f)` / `np.nanmean(P["Q"][tr], 0)` | time-of-day profile from the 6 (3-fold) or 8 (LOMO) training months, which include months AFTER the dev window. It reads other days' rows at clock times > T (that is the point of freq_max_h). Never the dev window's own month | profile from the 9 train months, all before March/April 2031. Causal |
| freq on training windows | queue_cv `set_freq` with FQ[k] built from all training months | in-sample: the profile includes the training window's own day (weight ~1/180), i.e. its own label leaks into freq_max_h at ~0.5% weight | same in production (`task2_model.main` / `queue_final_predict` L81 `np.nanmean(P["Q"], 0)` over all 9 months). Same in CV and production, so it does not create a dev-vs-LB difference |
| tod, dow, lanes, cap, pos, panel | L108-113 | static. No leak | static |
| drop_t (v3 training only) | L70-74, `td.gen` mode "full" passes 0.35 | training inputs use the FULL unmasked day (rows < T-60 unmasked too) with 35% of row T dropped; dev/val inputs have Task-1-masked earlier rows. Train/serve skew, not leakage | same skew in production |

## Data paths

- Training windows (td.gen "full"): features from rows <= t0 of the unmasked day; labels `y = P["Q"][d, t0 + h]` (td.gen L59/L68) are the target, used only as labels. Dev windows are never in training days: 3-fold, LOMO and forward splits exclude the held-out months entirely (verified via `_m` month index).
- Months: dev CV trains on months on both sides of the held-out month (interpolation); production trains only on earlier months and serves March/April (extrapolation). Measured below with forward-chaining folds.
- `queue_matrix` (L117) = observed speed <= 0.6 vf with NaN kept: labels only.
- Dev truth: fill `U = ffill().bfill()` (queue_devset L51) uses horizon rows and, via bfill, rows after T+30 for leading-NaN links; truth only, never a feature. `obs`: observed cells only. `elig` (new): observed eligible cells only, which is what score_task2.py scores (`eligible = target[target.is_score_eligible.astype(bool)]`).
- Dev selector (queue_devset L63-70) uses the underlying horizon (hz, pers <= 0.9) exactly like the organizer selector; it chooses windows, it does not feed features.
- Persistence (queue_devset L137 / queue_cv L140) uses `Qv[t - 1]` = eligible visible T-5 row: identical information in dev and release.
- Production (queue_final_predict L28-33, task2_model L293): features from `mainline_states_masked` of the split only; no window_history merge is needed because the rows are identical (verified).

## Verdict

No feature or data path reads a timestamp > T of the window's own day, in dev or in validation/private. The only non-causal element in dev is cross-month: freq profiles and training windows come partly from months after the dev month. Train/serve skews that exist identically in dev and production (full-day training inputs, in-sample freq on training windows) cannot explain a dev-vs-LB gap. The one dev-vs-release input difference is the 18-row horizon+buffer blank of earlier same-day windows, which only reaches `day_min_r`; its effect is measured in maskmimic_*.
