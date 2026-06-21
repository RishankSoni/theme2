import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from numpy.linalg import lstsq

df = pd.read_csv('eda_report/enriched_data.csv', low_memory=False)
print(f'Loaded {df.shape[0]} rows x {df.shape[1]} cols')

# ────────────────────────────────────────────────────────────
# STEP 1: authenticated vs resolution_minutes (controlled)
# ────────────────────────────────────────────────────────────
print()
print('=' * 70)
print('STEP 1 — authenticated vs resolution_minutes (controlling event_cause)')
print('=' * 70)

res_df = df.dropna(subset=['resolution_minutes', 'authenticated', 'event_cause']).copy()
auth_str = res_df['authenticated'].astype(str).str.lower()
res_df['auth_bin'] = auth_str.isin(['true', '1', 'yes', 't']).astype(int)

auth0 = res_df[res_df['auth_bin'] == 0]['resolution_minutes']
auth1 = res_df[res_df['auth_bin'] == 1]['resolution_minutes']
print(f'  authenticated=False  n={len(auth0)}  median={auth0.median():.1f} min  mean={auth0.mean():.1f}')
print(f'  authenticated=True   n={len(auth1)}  median={auth1.median():.1f} min  mean={auth1.mean():.1f}')
u_stat, u_p = stats.mannwhitneyu(auth0, auth1, alternative='two-sided')
print(f'  Mann-Whitney U={u_stat:.0f}  p={u_p:.4e}  (two-sided)')

print()
print('  Resolution by authenticated x event_cause (median minutes):')
pivot = res_df.pivot_table(
    values='resolution_minutes', index='event_cause',
    columns='auth_bin', aggfunc='median', observed=True
)
pivot.columns = ['Unauth', 'Auth']
pivot['n_u'] = res_df[res_df['auth_bin'] == 0].groupby('event_cause')['resolution_minutes'].count()
pivot['n_a'] = res_df[res_df['auth_bin'] == 1].groupby('event_cause')['resolution_minutes'].count()
pivot['diff'] = pivot['Auth'] - pivot['Unauth']
pivot = pivot.dropna(subset=['Unauth', 'Auth']).sort_values('diff', ascending=False)
for cause, row in pivot.iterrows():
    line = (f'    {str(cause):<22}  Unauth={row["Unauth"]:>6.1f}'
            f'  Auth={row["Auth"]:>6.1f}  diff={row["diff"]:>+7.1f}'
            f'  (n_u={int(row["n_u"])} n_a={int(row["n_a"])})')
    print(line)

# Log-linear regression controlling for event_cause
res_df['cause_enc'] = LabelEncoder().fit_transform(res_df['event_cause'].astype(str))
X = res_df[['auth_bin', 'cause_enc']].values
y = np.log1p(res_df['resolution_minutes'].values)
A = np.column_stack([np.ones(len(X)), X])
coef, _, _, _ = lstsq(A, y, rcond=None)
y_pred = A @ coef
ss_res = np.sum((y - y_pred) ** 2)
ss_tot = np.sum((y - y.mean()) ** 2)
auth_multiplier = np.exp(coef[1])
print()
print('  Log-linear regression (controlling event_cause):')
print(f'    intercept        = {coef[0]:.4f}')
print(f'    auth_bin coef    = {coef[1]:.4f}  (exp={auth_multiplier:.3f}x multiplier on resolution time)')
print(f'    event_cause coef = {coef[2]:.4f}')
print(f'    R2               = {1 - ss_res/ss_tot:.4f}')
if abs(coef[1]) > 0.05:
    print('  -> REAL EFFECT: authenticated incidents take longer even after controlling for cause')
else:
    print('  -> CONFOUND: auth effect disappears after controlling for cause')

# ────────────────────────────────────────────────────────────
# STEP 2: Rebuild severity_high from T2/T3 vocabulary
# ────────────────────────────────────────────────────────────
print()
print('=' * 70)
print('STEP 2 — Rebuild severity_high from cluster vocabulary')
print('=' * 70)

orig_hit = df['desc_severity_high'].mean() if 'desc_severity_high' in df.columns else 0.0
print(f'  Original severity_high hit rate: {orig_hit * 100:.1f}%')

HIGH_CAUSE = {'accident', 'water_logging', 'procession', 'vip_movement', 'protest', 'Fog / Low Visibility'}
high_cause_mask = df['event_cause'].isin(HIGH_CAUSE)
road_closure_mask = df['requires_road_closure'].astype(str).str.lower().isin(['true', '1', 'yes'])

desc_col = 'desc_normalized' if 'desc_normalized' in df.columns else 'description'
has_slow = df['desc_traffic_slow'].fillna(0).astype(bool)
has_problem_kw = df[desc_col].fillna('').str.lower().str.contains(
    r'\bproblem\b|\bheavy\b|\bblock\b|\bstandstill\b|\bcomplete\b', regex=True
)

new_sev_high = high_cause_mask | (road_closure_mask & (has_slow | has_problem_kw))
new_hit = new_sev_high.mean()
print(f'  New severity_high hit rate:      {new_hit * 100:.1f}%  ({new_sev_high.sum()} incidents)')
print()
print('  Breakdown of new severity_high sources:')
print(f'    High-risk cause only:            {(high_cause_mask & ~(road_closure_mask & (has_slow | has_problem_kw))).sum()}')
print(f'    Road closure + congestion only:  {(~high_cause_mask & road_closure_mask & (has_slow | has_problem_kw)).sum()}')
print(f'    Both conditions:                 {(high_cause_mask & road_closure_mask & (has_slow | has_problem_kw)).sum()}')
print()
print('  Top causes inside new severity_high:')
cause_counts = df[new_sev_high]['event_cause'].value_counts().head(10)
for cause, cnt in cause_counts.items():
    pct = cnt / new_sev_high.sum() * 100
    print(f'    {str(cause):<25}  {cnt:>5}  ({pct:.1f}%)')

# Validate: do new sev_high incidents resolve slower?
sev_df = df.dropna(subset=['resolution_minutes']).copy()
sev_df['new_sev'] = new_sev_high.reindex(sev_df.index).fillna(False)
sev0 = sev_df[~sev_df['new_sev']]['resolution_minutes']
sev1 = sev_df[sev_df['new_sev']]['resolution_minutes']
u2, p2 = stats.mannwhitneyu(sev0, sev1, alternative='less')
print()
print('  Validation — resolution time: severity_high=True vs False:')
print(f'    severity_high=False  median={sev0.median():.1f} min  n={len(sev0)}')
print(f'    severity_high=True   median={sev1.median():.1f} min  n={len(sev1)}')
label = 'CONFIRMED: high-sev resolves slower' if p2 < 0.05 else 'not statistically significant'
print(f'    Mann-Whitney (high>low)  p={p2:.4e}  -> {label}')

# ────────────────────────────────────────────────────────────
# STEP 3: resolution_minutes missingness pattern
# ────────────────────────────────────────────────────────────
print()
print('=' * 70)
print('STEP 3 — resolution_minutes missingness (MAR / MNAR analysis)')
print('=' * 70)

df['res_missing'] = df['resolution_minutes'].isna().astype(int)
fill_pct = (1 - df['res_missing'].mean()) * 100
print(f'  Overall fill rate: {fill_pct:.1f}%  ({df["resolution_minutes"].notna().sum()} / {len(df)})')
print()

print('  Fill rate by STATUS:')
for s, grp in df.groupby('status', observed=True):
    fill = grp['resolution_minutes'].notna().mean() * 100
    print(f'    {str(s):<12}  {fill:>6.1f}%  (n={len(grp)})')

print()
print('  Fill rate by event_cause (sorted ascending = hardest to get resolution):')
fill_by_cause = df.groupby('event_cause', observed=True)['resolution_minutes'].apply(
    lambda x: x.notna().mean() * 100
).sort_values()
for cause, pct in fill_by_cause.items():
    n_filled = df[df['event_cause'] == cause]['resolution_minutes'].notna().sum()
    n_total = (df['event_cause'] == cause).sum()
    print(f'    {str(cause):<25}  {pct:>6.1f}%  ({n_filled}/{n_total})')

print()
print('  Fill rate by time-of-day bucket:')
df_h = df.dropna(subset=['hour']).copy()
buckets = [(0, 6, 'Night (0-5h)'), (6, 12, 'Morning (6-11h)'),
           (12, 18, 'Afternoon (12-17h)'), (18, 24, 'Evening (18-23h)')]
for h0, h1, label in buckets:
    mask = (df_h['hour'] >= h0) & (df_h['hour'] < h1)
    fill = df_h[mask]['resolution_minutes'].notna().mean() * 100
    n = mask.sum()
    print(f'    {label:<22}  {fill:>6.1f}%  (n={n})')

print()
print('  Fill rate by corridor (top 10):')
top_corr = df['corridor'].value_counts().head(10).index
for c in top_corr:
    grp = df[df['corridor'] == c]
    fill = grp['resolution_minutes'].notna().mean() * 100
    print(f'    {str(c):<30}  {fill:>6.1f}%  (n={len(grp)})')

contingency = pd.crosstab(df['event_cause'], df['res_missing'])
chi2, p_chi, dof, _ = stats.chi2_contingency(contingency)
print()
print(f'  Chi2 test (missingness ~ event_cause):  chi2={chi2:.1f}  dof={dof}  p={p_chi:.4e}')
mnar = p_chi < 0.05
print(f'  -> {"MNAR: missingness is ASSOCIATED with event_cause" if mnar else "No significant association — closer to MAR"}')

status_fill = df.groupby('status', observed=True)['resolution_minutes'].apply(lambda x: x.notna().mean())
active_fill = status_fill.get('active', 0)
closed_fill = status_fill.get('closed', 0)
print()
if active_fill < 0.05 and closed_fill > 0.5:
    print('  DECISION RECOMMENDATION:')
    print('    Active incidents (12.3%) have near-zero fill — they are still open.')
    print('    Closed/resolved incidents have good fill. This is structural MNAR, not noise.')
    print('    Recommended approach:')
    print('      1. Train duration model ONLY on closed/resolved rows with resolution_minutes filled.')
    print('      2. For active incidents, predict resolution_minutes using the trained model.')
    print('      3. Do NOT impute mean/median — that would hallucinate resolution for open incidents.')
else:
    print('  DECISION: Review fill rate split above to determine imputation strategy.')

# ────────────────────────────────────────────────────────────
# STEP 4: Synthetic action-space generation profile
# ────────────────────────────────────────────────────────────
print()
print('=' * 70)
print('STEP 4 — Synthetic action-space generation profile')
print('=' * 70)

OFFICER_BASE = {
    'accident': 4.2, 'water_logging': 3.1, 'procession': 5.8, 'vip_movement': 6.5,
    'vehicle_breakdown': 1.5, 'pot_holes': 1.2, 'construction': 2.8, 'tree_fall': 2.1,
    'road_conditions': 1.8, 'congestion': 3.0, 'public_event': 4.0, 'protest': 5.0,
    'others': 1.5, 'Debris': 1.8, 'test_demo': 1.0, 'debris': 1.8, 'Fog / Low Visibility': 2.5
}
cause_probs = df['event_cause'].value_counts(normalize=True).to_dict()
road_closure_rate = df['requires_road_closure'].astype(str).str.lower().isin(['true', '1', 'yes']).mean()

print('  officers_deployed — Poisson(lambda) model:')
print(f'    Multipliers: road_closure=x1.8, priority_High=x1.3')
print('    Base lambda by cause:')
for cause, lam in sorted(OFFICER_BASE.items(), key=lambda x: -x[1]):
    p = cause_probs.get(cause, 0) * 100
    print(f'      {str(cause):<25}  lambda={lam}  (freq={p:.1f}%)')

print()
print('  diversion_route_used — Bernoulli model:')
print(f'    Road closure base rate in dataset: {road_closure_rate * 100:.1f}%')
print('    P(diversion | road_closure=True)                   = 0.85')
print('    P(diversion | no closure, high-risk cause)         = 0.45')
print('    P(diversion | no closure, breakdown/potholes/other) = 0.10')

print()
print('  barricades_placed — Geometric model:')
print('    road_closure=True:   Geometric(p=0.25) + 1  (mean~5, capped at 8)')
print('    road_closure=False:  Geometric(p=0.55) + 0  (mean~2, capped at 4)')
print('    accident/procession: +1 bonus barricade')

print()
print('  resolution_minutes — Log-normal conditioned on auth + cause:')
print(f'    base: lognormal(mu=4.526, sigma=1.760)')
print(f'    authenticated=True:  mu += {coef[1]:.4f}  (x{auth_multiplier:.3f} multiplier)')
print('    clamp to [1, 1440] minutes')

print()
print('  Generation function:')
code = '''
def generate_action_space(event_cause, requires_road_closure, priority, authenticated, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    OFFICER_BASE = {
        "accident": 4.2, "water_logging": 3.1, "procession": 5.8, "vip_movement": 6.5,
        "vehicle_breakdown": 1.5, "pot_holes": 1.2, "construction": 2.8, "tree_fall": 2.1,
        "road_conditions": 1.8, "congestion": 3.0, "public_event": 4.0, "protest": 5.0,
        "others": 1.5,
    }
    HIGH_RISK = {"accident", "procession", "vip_movement", "protest", "water_logging"}
    lam = OFFICER_BASE.get(event_cause, 1.5)
    if requires_road_closure:
        lam *= 1.8
    if priority == "High":
        lam *= 1.3
    officers = max(1, int(rng.poisson(lam)))

    if requires_road_closure:
        p_div = 0.85
    elif event_cause in HIGH_RISK:
        p_div = 0.45
    else:
        p_div = 0.10
    diversion = int(rng.random() < p_div)

    p_bar = 0.25 if requires_road_closure else 0.55
    bonus = 1 if event_cause in {"accident", "procession"} else 0
    barricades = min(int(rng.geometric(p_bar)) + bonus, 8 if requires_road_closure else 4)

    auth_adj = AUTH_COEF if authenticated else 0.0  # AUTH_COEF from regression
    res_min = float(np.clip(rng.lognormal(4.526 + auth_adj, 1.760), 1, 1440))

    return {
        "officers_deployed":    officers,
        "diversion_route_used": diversion,
        "barricades_placed":    barricades,
        "resolution_minutes":   res_min,
    }
'''
print(code)

print()
print('  Sanity-check synthetic distributions (n=10000):')
rng = np.random.default_rng(42)
samples = [
    ('vehicle_breakdown', False, 'High', False),
    ('accident',          True,  'High', True),
    ('procession',        True,  'Low',  False),
    ('pot_holes',         False, 'Low',  False),
]
HIGH_RISK_S = {'accident', 'procession', 'vip_movement', 'protest', 'water_logging'}
for cause, rc, pri, auth in samples:
    lam = OFFICER_BASE.get(cause, 1.5) * (1.8 if rc else 1.0) * (1.3 if pri == 'High' else 1.0)
    officers = np.maximum(1, rng.poisson(lam, 10000))
    p_div = 0.85 if rc else (0.45 if cause in HIGH_RISK_S else 0.10)
    divs = rng.random(10000) < p_div
    p_bar = 0.25 if rc else 0.55
    bars = np.minimum(rng.geometric(p_bar, 10000), 8 if rc else 4)
    mu_auth = 4.526 + (coef[1] if auth else 0.0)
    res_min = np.clip(rng.lognormal(mu_auth, 1.760, 10000), 1, 1440)
    print(f'    {str(cause):<22} rc={str(rc):<5} pri={pri}  auth={str(auth):<5}  '
          f'officers={officers.mean():.1f}  div={divs.mean():.0%}  '
          f'bars={bars.mean():.1f}  res_med={np.median(res_min):.0f}min')

print()
print('=' * 70)
print('All 4 steps complete.')
print('=' * 70)
