import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
from scipy import stats

# ── reproduction of generate_action_space from Step 4 ────────────────────────
AUTH_COEF = 1.4494  # from Step 1 log-linear regression

OFFICER_BASE = {
    "accident": 4.2, "water_logging": 3.1, "procession": 5.8, "vip_movement": 6.5,
    "vehicle_breakdown": 1.5, "pot_holes": 1.2, "construction": 2.8, "tree_fall": 2.1,
    "road_conditions": 1.8, "congestion": 3.0, "public_event": 4.0, "protest": 5.0,
    "others": 1.5, "Debris": 1.8, "debris": 1.8, "test_demo": 1.0,
    "Fog / Low Visibility": 2.5,
}
HIGH_RISK = {"accident", "procession", "vip_movement", "protest", "water_logging"}

def synthetic_resolution_sample(event_cause, requires_road_closure, authenticated, n=5000, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    auth_adj = AUTH_COEF if authenticated else 0.0
    samples = np.clip(rng.lognormal(4.526 + auth_adj, 1.760, n), 1, 1440)
    return samples


# ── load data ────────────────────────────────────────────────────────────────
df = pd.read_csv('eda_report/enriched_data.csv', low_memory=False)
df['auth_bin'] = df['authenticated'].astype(str).str.lower().isin(['true','1','yes','t'])
df['rc_bool']  = df['requires_road_closure'].astype(str).str.lower().isin(['true','1','yes'])

# Candidate pool: road_closure=True, High priority, resolution known, closed
pool = df[
    df['rc_bool'] &
    (df['priority'] == 'High') &
    df['resolution_minutes'].notna() &
    (df['status'] == 'closed')
].copy()

print(f'Candidate pool (rc=True, High priority, closed, res known): {len(pool)} incidents')
print()

# Sample up to 15 — stratify across event_cause to get variety
rng_sel = np.random.default_rng(99)
sample_rows = []
for cause in pool['event_cause'].unique():
    sub = pool[pool['event_cause'] == cause]
    n_take = min(3, len(sub))
    sample_rows.append(sub.sample(n_take, random_state=int(rng_sel.integers(1000))))
validation_set = pd.concat(sample_rows).head(15).reset_index(drop=True)

print(f'Validation set: {len(validation_set)} incidents')
print()

# ── per-incident comparison ───────────────────────────────────────────────────
print('=' * 100)
print(f'{"#":<3} {"cause":<22} {"rc":<5} {"auth":<5} {"actual_min":>10} '
      f'{"synth_p10":>10} {"synth_p50":>10} {"synth_p90":>10} {"in_range":>8} {"ratio":>7}')
print('-' * 100)

rng_gen = np.random.default_rng(42)
ratios = []
in_range_count = 0

for i, row in validation_set.iterrows():
    cause   = str(row['event_cause'])
    rc      = bool(row['rc_bool'])
    auth    = bool(row['auth_bin'])
    actual  = float(row['resolution_minutes'])

    samples = synthetic_resolution_sample(cause, rc, auth, n=5000, rng=rng_gen)
    p10, p50, p90 = np.percentile(samples, [10, 50, 90])

    in_range = p10 <= actual <= p90
    in_range_count += int(in_range)
    ratio = actual / p50 if p50 > 0 else float('nan')
    ratios.append(ratio)

    flag = '  OK' if in_range else ('  HIGH' if actual > p90 else '  LOW')
    print(f'{i+1:<3} {cause:<22} {str(rc):<5} {str(auth):<5} {actual:>10.1f} '
          f'{p10:>10.1f} {p50:>10.1f} {p90:>10.1f} {str(in_range):>8} {ratio:>6.2f}x{flag}')

print('=' * 100)

# ── aggregate bias metrics ────────────────────────────────────────────────────
print()
print('Aggregate bias metrics:')
ratios_arr = np.array([r for r in ratios if not np.isnan(r)])
print(f'  n incidents           : {len(validation_set)}')
print(f'  In synthetic P10-P90  : {in_range_count}/{len(validation_set)}  '
      f'({in_range_count/len(validation_set)*100:.0f}%)')
print(f'  actual/synth_median:')
print(f'    geometric mean ratio: {np.exp(np.log(ratios_arr).mean()):.3f}x  '
      f'(1.0 = perfect, >1 = model underestimates, <1 = overestimates)')
print(f'    median ratio        : {np.median(ratios_arr):.3f}x')
print(f'    min ratio           : {ratios_arr.min():.3f}x')
print(f'    max ratio           : {ratios_arr.max():.3f}x')

geo_mean = np.exp(np.log(ratios_arr).mean())
if geo_mean > 2.0:
    verdict = 'UNDERESTIMATE: real incidents resolve much slower — inflate mu or sigma'
elif geo_mean < 0.5:
    verdict = 'OVERESTIMATE: model predicts too slow — deflate mu'
elif 0.7 <= geo_mean <= 1.4:
    verdict = 'GOOD: synthetic distribution is broadly calibrated'
else:
    verdict = 'MILD BIAS: acceptable for simulation but note direction'
print(f'\n  Verdict: {verdict}')

# ── break down by auth vs not ─────────────────────────────────────────────────
print()
print('Ratio by authenticated flag:')
for auth_val in [False, True]:
    sub_ratios = [r for r, row in zip(ratios, validation_set.itertuples())
                  if bool(row.auth_bin) == auth_val and not np.isnan(r)]
    if sub_ratios:
        gm = np.exp(np.log(sub_ratios).mean()) if sub_ratios else float('nan')
        print(f'  authenticated={str(auth_val):<5}  n={len(sub_ratios)}  '
              f'geo_mean_ratio={gm:.3f}x')

# ── show the 3 worst misses ───────────────────────────────────────────────────
print()
print('3 worst misses (largest actual/synth_median ratio):')
worst_idx = np.argsort(ratios_arr)[::-1][:3]
for rank, idx in enumerate(worst_idx, 1):
    row = validation_set.iloc[idx]
    samples = synthetic_resolution_sample(
        str(row['event_cause']), bool(row['rc_bool']), bool(row['auth_bin']),
        n=5000, rng=np.random.default_rng(42)
    )
    p50 = np.median(samples)
    actual = float(row['resolution_minutes'])
    print(f'  {rank}. cause={row["event_cause"]}  auth={row["auth_bin"]}  '
          f'actual={actual:.1f}min  synth_median={p50:.1f}min  '
          f'ratio={actual/p50:.2f}x  description: {str(row.get("description",""))[:80]}')

print()
print('Calibration recommendation:')
print('  If geo_mean_ratio > 1.5 for authenticated=True incidents:')
print('    -> AUTH_COEF is underestimating: raise from 1.4494 toward 2.0')
print('  If in-range % < 60%:')
print('    -> widen sigma from 1.760 toward 2.0 (fatter tails)')
print('  If systematic cause-specific miss (e.g., water_logging always overestimates):')
print('    -> add per-cause mu offset dict to generation function')
