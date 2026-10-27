#!/usr/bin/env python3
"""ANCP Group P | D2 | Reproduces data-v1.0 from the raw NASA C-MAPSS files.
Usage: python clean_data.py RAW_DIR OUT_DIR      (RAW_DIR holds train_/test_/RUL_FD001|FD004.txt, left untouched)
Outputs: OUT_DIR/{FD001,FD004}_{train,test}_clean.csv, data_dictionary.csv, audit_results.json, SHA256SUMS.txt
"""
import sys, json, hashlib, pathlib
import numpy as np, pandas as pd
from scipy import stats

RAW, OUT = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
COLS = ['unit', 'cycle', 'op1', 'op2', 'op3'] + [f's{i}' for i in range(1, 22)]
SENS = COLS[5:]
CAP, ZLIM = 125, 4.0          # RUL cap (literature convention, tunable in D4); outlier |z| limit
audit, dictionary = {}, {}


def load(kind, fd):
    return pd.read_csv(RAW / f'{kind}_{fd}.txt', sep=r'\s+', header=None, names=COLS)


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


for fd in ['FD001', 'FD004']:
    tr, te = load('train', fd), load('test', fd)
    rul_true = pd.read_csv(RAW / f'RUL_{fd}.txt', header=None)[0].values
    a = {}
    # ---------- 1. quality audit ----------
    for nm, d in [('train', tr), ('test', te)]:
        gaps = d.groupby('unit')['cycle'].apply(lambda c: (c.diff().dropna() != 1).sum() + (c.iloc[0] != 1)).sum()
        a[nm] = dict(rows=len(d), engines=int(d.unit.nunique()), missing=int(d.isna().sum().sum()),
                     dup_rows=int(d.duplicated().sum()), dup_unit_cycle=int(d.duplicated(['unit', 'cycle']).sum()),
                     cycle_gaps=int(gaps), non_numeric=int((d.dtypes == object).sum()))
    const = [c for c in COLS[2:] if tr[c].std() < 1e-8]
    near = [c for c in COLS[2:] if c not in const and tr[c].nunique() <= 5]
    a['constant'], a['near_constant'] = const, near
    # ---------- 2. operating condition id (6 regimes in FD004, 1 in FD001) ----------
    key = lambda d: list(zip(d.op1.round(0).astype(int), d.op2.round(2)))
    ids = {k: i + 1 for i, k in enumerate(sorted(set(key(tr))))}
    tr['condition_id'], te['condition_id'] = [ids[k] for k in key(tr)], [ids.get(k, 0) for k in key(te)]
    a['n_conditions'] = len(ids)
    a['unseen_test_conditions'] = int((te.condition_id == 0).sum())
    # ---------- 3. RUL labels (train: last cycle - cycle; test: true RUL at last cycle + cycles remaining to it) ----------
    for d, truth in [(tr, None), (te, rul_true)]:
        last = d.groupby('unit').cycle.transform('max')
        base = 0 if truth is None else d.unit.map(dict(enumerate(truth, 1)))
        d['RUL'] = (last - d.cycle) + base
        d['RUL_cap'] = d.RUL.clip(upper=CAP)
    # ---------- 4. per-condition normalisation, statistics from TRAIN only (no leakage) ----------
    keep = [c for c in SENS if c not in const]
    zkeep = [c for c in keep if c not in near]   # z-rule invalid for discretised sensors
    mu = tr.groupby('condition_id')[keep].mean()
    sd = tr.groupby('condition_id')[keep].std().replace(0, 1)
    for d in (tr, te):
        m, s = mu.reindex(d.condition_id).values, sd.reindex(d.condition_id).values
        z = (d[keep].values - m) / s
        d[[f'{c}_z' for c in keep]] = z
        d['outlier_flag'] = (np.abs(d[[f'{c}_z' for c in zkeep]].values) > ZLIM).any(axis=1).astype(int)
    zc = [f'{c}_z' for c in keep]; zk = [f'{c}_z' for c in zkeep]
    a['outlier_rows_train'] = int(tr.outlier_flag.sum()); a['outlier_rows_test'] = int(te.outlier_flag.sum())
    a['outlier_readings_train'] = int((tr[zk].abs() > ZLIM).sum().sum())
    a['outlier_by_sensor'] = {c: int((tr[f'{c}_z'].abs() > ZLIM).sum()) for c in zkeep if (tr[f'{c}_z'].abs() > ZLIM).sum()}
    a['raw_range_ratio'] = {c: float(tr[c].max() / (abs(tr[c].min()) + 1e-9)) for c in ['s9', 's14']}
    # ---------- 5. lifetimes, representativeness, simulation inputs ----------
    life_tr = tr.groupby('unit').cycle.max().values
    life_te_obs = te.groupby('unit').cycle.max().values
    life_te_tot = life_te_obs + rul_true
    sh, _, sc = stats.weibull_min.fit(life_tr, floc=0)
    a['life_train'] = dict(min=int(life_tr.min()), q25=float(np.percentile(life_tr, 25)), median=float(np.median(life_tr)),
                           mean=float(life_tr.mean()), q75=float(np.percentile(life_tr, 75)), max=int(life_tr.max()),
                           sd=float(life_tr.std(ddof=1)))
    a['life_test_total'] = dict(min=int(life_te_tot.min()), median=float(np.median(life_te_tot)), mean=float(life_te_tot.mean()),
                                max=int(life_te_tot.max()))
    a['test_observed_cycles'] = dict(min=int(life_te_obs.min()), median=float(np.median(life_te_obs)), max=int(life_te_obs.max()))
    a['test_final_rul'] = dict(min=int(rul_true.min()), median=float(np.median(rul_true)), max=int(rul_true.max()))
    a['ks_train_vs_test_life_p'] = float(stats.ks_2samp(life_tr, life_te_tot).pvalue)
    a['weibull'] = dict(shape=float(sh), scale=float(sc))
    a['share_train_by_cond'] = (tr.condition_id.value_counts(normalize=True).sort_index().round(3)).to_dict()
    a['share_test_by_cond'] = (te.condition_id.value_counts(normalize=True).sort_index().round(3)).to_dict()
    a['rul_cap_share_train'] = float((tr.RUL > CAP).mean())
    a['trend_sensors_corr'] = {c: float(tr[f'{c}_z'].corr(tr.RUL_cap)) for c in keep}
    audit[fd] = a
    # ---------- 6. write cleaned files ----------
    out_cols = ['unit', 'cycle', 'condition_id', 'op1', 'op2'] + [c for c in ['op3'] if c not in const] + keep + zc + \
               ['outlier_flag', 'RUL', 'RUL_cap']
    for nm, d in [('train', tr), ('test', te)]:
        d[out_cols].round(4).to_csv(OUT / f'{fd}_{nm}_clean.csv', index=False)
    audit[fd]['out_cols'] = out_cols
    audit[fd]['dropped'] = const

# ---------- data dictionary ----------
dd = [('unit', 'int', 'Engine identifier (unique within a file)', 'raw'),
      ('cycle', 'int', 'Flight cycle number, starts at 1', 'raw'),
      ('condition_id', 'int', 'Operating regime from rounded op1/op2 (FD001: 1; FD004: 1-6)', 'engineered'),
      ('op1', 'float', 'Operating setting 1 (altitude)', 'raw'), ('op2', 'float', 'Operating setting 2 (Mach number)', 'raw'),
      ('op3', 'float', 'Operating setting 3 (throttle resolver angle); dropped where constant', 'raw'),
      ('s1..s21', 'float', 'Sensor readings (constant ones dropped per subset); numbering as in NASA readme', 'raw'),
      ('s*_z', 'float', 'Sensor z-score within condition_id; mean/sd from TRAIN file only', 'engineered'),
      ('outlier_flag', 'int', '1 if any |z| > 4 in the row (discretised near-constant sensors excluded); flag only, row retained', 'engineered'),
      ('RUL', 'int', 'Remaining useful life in cycles (train: last cycle - cycle; test: true final RUL + cycles to last)', 'target'),
      ('RUL_cap', 'int', 'RUL capped at 125 cycles', 'target')]
pd.DataFrame(dd, columns=['column', 'type', 'meaning', 'origin']).to_csv(OUT / 'data_dictionary.csv', index=False)
json.dump(audit, open(OUT / 'audit_results.json', 'w'), indent=1, default=float)
files = sorted(OUT.glob('*.csv')) + sorted(RAW.glob('*.txt'))
(OUT.parent / 'SHA256SUMS.txt').write_text(''.join(f'{sha(f)}  {f.parent.name}/{f.name}\n' for f in files))
print('done', [f.name for f in files])
