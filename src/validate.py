# %% [markdown]
# ## 7. Single-target vs multi-target untuk ketiga model
# Tiap keluarga model dibandingkan dalam dua versi.
# - Multi-target (MT): satu model untuk dua target. RF multi-output, CatBoost MultiRMSE, MultiTaskElasticNet.
# - Single-target (ST): dua model terpisah, satu per target.
# Semua versi di-tuning dengan protokol yang sama (Optuna, GroupKFold per provinsi), kecuali RF-MT yang memakai
# parameter terkunci di bagian 6 agar konsisten dengan angka yang dilaporkan.

# %%
from sklearn.linear_model import MultiTaskElasticNet
from sklearn.compose import TransformedTargetRegressor
from catboost import Pool
from numpy.linalg import norm

FAMILIES = ['RandomForest', 'CatBoost', 'ElasticNet']

def _space(fam, tr):
    if fam == 'ElasticNet':
        return dict(alpha=tr.suggest_float('alpha', 1e-3, 10, log=True), l1_ratio=tr.suggest_float('l1_ratio', 0.05, 1.0))
    return space(fam, tr)

def build(fam, prm, multi, seed=SEED):
    """Satu estimator. multi=True untuk dua target sekaligus, False untuk satu target."""
    if fam == 'RandomForest':
        return RandomForestRegressor(random_state=seed, n_jobs=-1, **prm)
    if fam == 'CatBoost':
        return CatBoostRegressor(loss_function='MultiRMSE' if multi else 'RMSE', random_seed=seed, verbose=0, thread_count=-1, **prm)
    # ElasticNet, target distandarkan agar kedua target berbobot sama pada penalti grup
    core = MultiTaskElasticNet if multi else ElasticNet
    reg = Pipeline([('sc', StandardScaler()), ('m', core(max_iter=50000, random_state=seed, **prm))])
    return TransformedTargetRegressor(regressor=reg, transformer=StandardScaler())

def shap_of(fam, est, X, multi):
    """Nilai SHAP berbentuk (n, f, 2) untuk multi, (n, f) untuk single."""
    if fam == 'RandomForest':
        return np.asarray(shap.TreeExplainer(est).shap_values(X))
    if fam == 'CatBoost':
        sv = est.get_feature_importance(Pool(X), type='ShapValues')
        return np.transpose(sv[:, :, :-1], (0, 2, 1)) if multi else sv[:, :-1]
    # ElasticNet, SHAP linear eksak dengan fitur independen = koefisien x (z - rata-rata z latih), dikali skala target
    reg = est.regressor_; z = reg.named_steps['sc'].transform(X)
    coef = np.atleast_2d(reg.named_steps['m'].coef_); ysd = np.atleast_1d(est.transformer_.scale_)
    sv = z[:, :, None] * (coef.T * ysd)[None, :, :]
    return sv if multi else sv[:, :, 0]

class Setup:
    """Membungkus versi MT atau ST agar keduanya punya predict dan shap berbentuk sama."""
    def __init__(self, fam, multi, prm):
        self.fam, self.multi, self.prm = fam, multi, prm
    def fit(self, X, Y, seed=SEED):
        if self.multi:
            self.models = [build(self.fam, self.prm, True, seed).fit(X, Y.values)]
        else:
            self.models = [build(self.fam, self.prm[k], False, seed).fit(X, Y.iloc[:, k].values) for k in range(2)]
        return self
    def predict(self, X):
        if self.multi: return self.models[0].predict(X)
        return np.column_stack([m.predict(X) for m in self.models])
    def shap(self, X):
        if self.multi: return shap_of(self.fam, self.models[0], X, True)
        return np.stack([shap_of(self.fam, m, X, False) for m in self.models], axis=2)

def tune_setup(fam, multi, X, Y, groups, k=None):
    def score(tr):
        prm = _space(fam, tr); sc = []
        for tri, vai in GroupKFold(n_splits=5).split(X, Y, groups):
            if multi:
                p = build(fam, prm, True).fit(X.iloc[tri], Y.iloc[tri].values).predict(X.iloc[vai])
                sc.append(np.mean([r2_score(Y.iloc[vai, j], p[:, j]) for j in range(2)]))
            else:
                p = build(fam, prm, False).fit(X.iloc[tri], Y.iloc[tri, k].values).predict(X.iloc[vai])
                sc.append(r2_score(Y.iloc[vai, k], p))
        return float(np.mean(sc))
    st = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
    st.optimize(score, n_trials=N_TRIALS)
    return st.best_params, st.best_value

def ct_metrics(SV, ids_, base_):
    """Kemiripan penjelasan antara dua target."""
    G = pd.DataFrame({k: np.abs(SV[:, :, k]).mean(0) / base_[k] for k in range(2)}, index=SE_FEATURES)
    out = dict(spearman=spearmanr(G[0], G[1]).statistic, cosine=float(G[0] @ G[1] / (norm(G[0]) * norm(G[1]))))
    PS = [pd.DataFrame(SV[:, :, k], columns=SE_FEATURES).groupby(ids_).mean() for k in range(2)]
    jac = []
    for pid in PS[0].index:
        a = set(PS[0].loc[pid].abs().nlargest(3).index); b = set(PS[1].loc[pid].abs().nlargest(3).index)
        jac.append(len(a & b) / len(a | b))
    out.update(jaccard_top3=float(np.mean(jac)), identik_top3=int(np.sum(np.array(jac) == 1)))
    return out

Xtr, Ytr, gtr = train[SE_FEATURES], train[TARGETS], train['id']
X_eval, ids_eval = data[SE_FEATURES], data['id'].values
base_eval = data[TARGETS].mean().values
SETUP_PARAMS, cmp_rows, SV_store = {}, [], {}
for fam in FAMILIES:
    for multi in [True, False]:
        if fam == 'RandomForest' and multi:
            prm, cvs = RF_WEB_PARAMS, np.nan
        elif multi:
            prm, cvs = tune_setup(fam, True, Xtr, Ytr, gtr)
        else:
            res_ = [tune_setup(fam, False, Xtr, Ytr, gtr, k) for k in range(2)]
            prm, cvs = [r[0] for r in res_], float(np.mean([r[1] for r in res_]))
        SETUP_PARAMS[(fam, multi)] = prm
        s = Setup(fam, multi, prm).fit(Xtr, Ytr)
        P = s.predict(test[SE_FEATURES]); SV = s.shap(X_eval); SV_store[(fam, multi)] = SV
        row = dict(model=fam, versi='Multi-target' if multi else 'Single-target', CV_R2=cvs)
        for k, t in enumerate(TARGETS):
            row[f'R2_{t}'] = r2_score(test[t], P[:, k]); row[f'MAE_{t}'] = mean_absolute_error(test[t], P[:, k])
        row.update(ct_metrics(SV, ids_eval, base_eval))
        cmp_rows.append(row); print(row)
cmp_df = pd.DataFrame(cmp_rows); cmp_df.to_csv(f'{OUT}/single_vs_multi.csv', index=False)
json.dump({f'{k[0]}|{"MT" if k[1] else "ST"}': v for k, v in SETUP_PARAMS.items()}, open(f'{OUT}/parameter_single_vs_multi.json', 'w'), indent=1, default=str)
print(cmp_df.round(3).to_string())

# %% [markdown]
# ## 8. Uji 2, bootstrap per provinsi
# Provinsi data latih diambil ulang dengan pengembalian (cluster bootstrap), model MT dan ST dilatih ulang dengan
# parameter dari bagian 7, lalu kemiripan penjelasan antartarget dihitung pada data evaluasi yang sama.
# Hasilnya interval kepercayaan 95% untuk selisih kemiripan MT dikurangi ST.

# %%
N_BOOT = int(os.environ.get('N_BOOT', 200))
rng = np.random.default_rng(SEED)
prov_ids = np.sort(train['id'].unique())
boot = []
for b in range(N_BOOT):
    pick = rng.choice(prov_ids, size=len(prov_ids), replace=True)
    tb = pd.concat([train[train['id'] == p] for p in pick], ignore_index=True)
    for fam in FAMILIES:
        r = {}
        for multi in [True, False]:
            s = Setup(fam, multi, SETUP_PARAMS[(fam, multi)]).fit(tb[SE_FEATURES], tb[TARGETS], seed=SEED + b)
            m = ct_metrics(s.shap(X_eval), ids_eval, base_eval)
            for key in ['spearman', 'cosine', 'jaccard_top3']:
                r[f'{key}_{"MT" if multi else "ST"}'] = m[key]
        boot.append(dict(b=b, model=fam, **r))
    if b % 20 == 0: print('bootstrap', b)
boot_df = pd.DataFrame(boot); boot_df.to_csv(f'{OUT}/bootstrap.csv', index=False)
boot_sum = []
for fam in FAMILIES:
    d = boot_df[boot_df['model'] == fam]
    for key in ['spearman', 'cosine', 'jaccard_top3']:
        diff = d[f'{key}_MT'] - d[f'{key}_ST']
        boot_sum.append(dict(model=fam, metrik=key, MT=d[f'{key}_MT'].mean(), ST=d[f'{key}_ST'].mean(),
                             selisih=diff.mean(), CI_bawah=diff.quantile(0.025), CI_atas=diff.quantile(0.975),
                             proporsi_MT_lebih_mirip=(diff > 0).mean()))
boot_sum = pd.DataFrame(boot_sum); boot_sum.to_csv(f'{OUT}/bootstrap_ringkasan.csv', index=False)
print(boot_sum.round(3).to_string())

# %% [markdown]
# ## 9. Uji 3, data sintetis dengan jawaban yang diketahui
# Dua target dibangkitkan dari 23 fitur (saling berkorelasi 0,3). Faktor penyebabnya ditentukan sendiri.
# - Sama: kedua target memakai x0, x1, x2.
# - Sebagian: target A memakai x0, x1, x2, target B memakai x0, x3, x4.
# - Berbeda: target A memakai x0, x1, x2, target B memakai x3, x4, x5.
# Ukuran yang dilihat
# - Kemiripan penjelasan antartarget dibanding kemiripan sebenarnya.
# - Kebocoran, yaitu porsi kepentingan suatu target yang jatuh ke fitur milik target lain (seharusnya 0).
# - Ketepatan tiga fitur teratas tiap target dibanding fitur penyebab sebenarnya.

# %%
NF = len(SE_FEATURES)
COMP = {0: lambda x: 2.0 * x, 1: lambda x: 1.5 * np.tanh(1.5 * x), 2: lambda x: x + 0.3 * x ** 3 / 3,
        3: lambda x: 2.0 * x, 4: lambda x: 1.5 * np.tanh(1.5 * x), 5: lambda x: x + 0.3 * x ** 3 / 3}
SCEN = {'Sama': ([0, 1, 2], [0, 1, 2]), 'Sebagian': ([0, 1, 2], [0, 3, 4]), 'Berbeda': ([0, 1, 2], [3, 4, 5])}
W_B = {0: 1.0, 1: 1.6, 2: 0.7, 3: 1.2, 4: 0.8, 5: 1.0}  # bobot target B, berbeda dari A agar tidak identik

def gen(n, scen, seed):
    r = np.random.default_rng(seed)
    C = np.full((NF, NF), 0.3); np.fill_diagonal(C, 1)
    X = r.multivariate_normal(np.zeros(NF), C, size=n)
    fa, fb = SCEN[scen]
    A = sum(COMP[j](X[:, j]) for j in fa); B = 5 * sum(W_B[j] * COMP[j](X[:, j]) for j in fb)
    A = A + r.normal(0, 0.6 * A.std(), n); B = B + r.normal(0, 0.6 * B.std(), n)
    return pd.DataFrame(X, columns=SE_FEATURES), pd.DataFrame({TARGETS[0]: A, TARGETS[1]: B})

def true_importance(scen):
    X, _ = gen(200000, scen, 0); fa, fb = SCEN[scen]
    ia, ib = np.zeros(NF), np.zeros(NF)
    for j in fa: c = COMP[j](X.iloc[:, j]); ia[j] = np.abs(c - c.mean()).mean()
    for j in fb: c = 5 * W_B[j] * COMP[j](X.iloc[:, j]); ib[j] = np.abs(c - c.mean()).mean()
    return ia, ib

SYN_N = [int(v) for v in os.environ.get('SYN_N', '100,300,1000,3000').split(',')]
SYN_SEEDS = int(os.environ.get('SYN_SEEDS', 5))
syn = []
for scen in SCEN:
    ia, ib = true_importance(scen); fa, fb = SCEN[scen]
    true_cos = float(ia @ ib / (norm(ia) * norm(ib)))
    only_a, only_b = sorted(set(fa) - set(fb)), sorted(set(fb) - set(fa))
    for n in SYN_N:
        for sd in range(SYN_SEEDS):
            Xs, Ys = gen(n, scen, 1000 + sd); Xe, _ = gen(500, scen, 9000 + sd)
            for fam in FAMILIES:
                for multi in [True, False]:
                    prm = SETUP_PARAMS[(fam, multi)]
                    s = Setup(fam, multi, prm).fit(Xs, Ys, seed=sd)
                    SV = s.shap(Xe); G = np.abs(SV).mean(0)  # (f, 2)
                    ga, gb = G[:, 0], G[:, 1]
                    row = dict(skenario=scen, n=n, seed=sd, model=fam, versi='MT' if multi else 'ST',
                               cos_estimasi=float(ga @ gb / (norm(ga) * norm(gb))), cos_sebenarnya=true_cos,
                               top3_A_benar=len(set(np.argsort(-ga)[:3]) & set(fa)) / 3,
                               top3_B_benar=len(set(np.argsort(-gb)[:3]) & set(fb)) / 3)
                    row['bias_kemiripan'] = row['cos_estimasi'] - true_cos
                    row['bocor_A'] = ga[only_b].sum() / ga.sum() if only_b else np.nan
                    row['bocor_B'] = gb[only_a].sum() / gb.sum() if only_a else np.nan
                    syn.append(row)
    print('skenario', scen, 'selesai')
syn_df = pd.DataFrame(syn); syn_df.to_csv(f'{OUT}/sintetis.csv', index=False)
syn_sum = syn_df.groupby(['skenario', 'n', 'model', 'versi'])[['cos_sebenarnya', 'cos_estimasi', 'bias_kemiripan', 'bocor_A', 'bocor_B', 'top3_A_benar', 'top3_B_benar']].mean().reset_index()
syn_sum.to_csv(f'{OUT}/sintetis_ringkasan.csv', index=False)
print(syn_sum.round(3).to_string())
