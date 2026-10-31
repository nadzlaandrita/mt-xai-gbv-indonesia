import matplotlib
matplotlib.use('Agg')
import os, json, warnings
import numpy as np, pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from catboost import CatBoostRegressor
import optuna, shap
from lime import lime_tabular
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)
SEED = 67
N_TRIALS = int(os.environ.get('N_TRIALS', 30))
OUT = os.environ.get('OUT', 'out'); os.makedirs(OUT, exist_ok=True)
# Di Colab, ganti path ini ke lokasi file di Google Drive, misalnya '/content/drive/My Drive/File Dataset/RAW_dataset.xlsx'
RAW_PATH = os.environ.get('RAW_PATH', 'RAW_dataset.xlsx')

PROV = ['ACEH','BALI','BANTEN','BENGKULU','DI YOGYAKARTA','DKI JAKARTA','GORONTALO','JAMBI','JAWA BARAT','JAWA TENGAH',
        'JAWA TIMUR','KALIMANTAN BARAT','KALIMANTAN SELATAN','KALIMANTAN TENGAH','KALIMANTAN TIMUR','KALIMANTAN UTARA',
        'KEP. BANGKA BELITUNG','KEP. RIAU','LAMPUNG','MALUKU','MALUKU UTARA','NUSA TENGGARA BARAT','NUSA TENGGARA TIMUR',
        'PAPUA','PAPUA BARAT','RIAU','SULAWESI BARAT','SULAWESI SELATAN','SULAWESI TENGAH','SULAWESI TENGGARA',
        'SULAWESI UTARA','SUMATERA BARAT','SUMATERA SELATAN','SUMATERA UTARA']
TARGETS = ['rate_seksual', 'rate_non_seksual']

def load_raw(path):
    # file xlsx yang dikirim berisi teks CSV dalam satu kolom, fungsi ini menangani keduanya
    if path.endswith('.csv'):
        return pd.read_csv(path)
    x = pd.read_excel(path, header=None)
    if x.shape[1] == 1:
        import io
        return pd.read_csv(io.StringIO('\n'.join(x.iloc[:, 0].astype(str))))
    return pd.read_excel(path)

df = load_raw(RAW_PATH)
df = df.drop(columns=[c for c in df.columns if c.startswith('Unnamed')])
df['id'] = df.index % 34
df['tahun'] = df.index // 34
df['year'] = 2021 + df['tahun']
df['provinsi'] = df['id'].map(dict(enumerate(PROV)))

# verifikasi urutan: populasi satu provinsi harus berubah sedikit antartahun (kecuali Papua/Papua Barat 2024)
pop = df.pivot(index='id', columns='tahun', values='populasi')
growth = (pop[2] / pop[0]).describe()
assert growth['min'] > 0.95 and growth['max'] < 1.1, 'Urutan provinsi/tahun tidak konsisten'

# verifikasi target = kasus / populasi x 100.000
assert np.allclose(df['seksual'] / df['populasi'] * 1e5, df['rate_seksual'])
assert np.allclose(df['non_seksual'] / df['populasi'] * 1e5, df['rate_non_seksual'])

# indeks yang terkalikan populasi dikembalikan ke skala asli
for c in ['IPG', 'edu_index', 'RLS_gap', 'PPDPSK_avg']:
    df[c] = df[c] / df['populasi'] * 1000

# PPDPSK_avg bernilai 0 (kosong) untuk DIY dan Kep. Riau 2024, diisi nilai tahun sebelumnya
df.loc[df['PPDPSK_avg'] == 0, 'PPDPSK_avg'] = np.nan
df = df.sort_values(['id', 'tahun'])
df['PPDPSK_avg'] = df.groupby('id')['PPDPSK_avg'].ffill()

# Papua & Papua Barat 2024, pembilang masih wilayah lama tetapi populasi sudah wilayah baru
df['invalid'] = df['id'].isin([23, 24]) & (df['tahun'] == 3)
df['log_populasi'] = np.log(df['populasi'])

# lag target (tahun sebelumnya, hanya untuk model prediktif)
for t in TARGETS:
    df[f'{t}_lag1'] = df.groupby('id')[t].shift(1)
df = df.sort_values(['tahun', 'id']).reset_index(drop=True)

SE_FEATURES = ['PDRB_per_kapita_HK','PDRB_per_kapita','PPKMLA','IPG','miskin_rate','ppm_rate','tk_sekolah_per_1000',
               'tk_murid_per_1000','pt_per_1000','dosen_per_1000','mahasiswa_per_1000','mahasiswa_per_PT','dosen_per_PT',
               'murid_per_sekolah','guru_per_sekolah','murid_per_guru','smk_density','pt_density','edu_index','RLS_gap',
               'PPDPSK_avg','school_density','log_populasi']
LAG_FEATURES = [f'{t}_lag1' for t in TARGETS]

data = df[~df['invalid']].copy()
train = data[data['tahun'] < 3]
test = data[data['tahun'] == 3]
print('Train', train.shape[0], 'Test', test.shape[0])
df.to_csv(f'{OUT}/dataset_bersih.csv', index=False)

def metrics(y, p):
    return dict(R2=r2_score(y, p), MAE=mean_absolute_error(y, p),
                RMSE=np.sqrt(mean_squared_error(y, p)), MAPE=mean_absolute_percentage_error(y, p))

def make(name, prm, seed=SEED):
    if name == 'ElasticNet':
        return MultiOutputRegressor(Pipeline([('sc', StandardScaler()),
               ('m', ElasticNet(alpha=prm['alpha'], l1_ratio=prm['l1_ratio'], max_iter=50000, random_state=seed))]))
    if name == 'RandomForest':
        return RandomForestRegressor(random_state=seed, n_jobs=-1, **prm)
    if name == 'CatBoost':
        return CatBoostRegressor(loss_function='MultiRMSE', random_seed=seed, verbose=0, **prm)

def space(name, tr):
    if name == 'ElasticNet':
        return dict(alpha=tr.suggest_float('alpha', 1e-3, 10, log=True), l1_ratio=tr.suggest_float('l1_ratio', 0.05, 1.0))
    if name == 'RandomForest':
        return dict(n_estimators=tr.suggest_int('n_estimators', 200, 600, step=100), max_depth=tr.suggest_int('max_depth', 3, 15),
                    min_samples_leaf=tr.suggest_int('min_samples_leaf', 1, 6),
                    max_features=tr.suggest_categorical('max_features', ['sqrt', 0.33, 0.5, 0.8, 1.0]))
    if name == 'CatBoost':
        return dict(iterations=tr.suggest_int('iterations', 200, 800, step=100), depth=tr.suggest_int('depth', 2, 6),
                    learning_rate=tr.suggest_float('learning_rate', 0.01, 0.2, log=True),
                    l2_leaf_reg=tr.suggest_float('l2_leaf_reg', 1, 20, log=True))

def cv_score(name, prm, X, Y, groups):
    sc = []
    for tri, vai in GroupKFold(n_splits=5).split(X, Y, groups):
        m = make(name, prm).fit(X.iloc[tri], Y.iloc[tri])
        p = m.predict(X.iloc[vai])
        sc.append(np.mean([r2_score(Y.iloc[vai, k], p[:, k]) for k in range(2)]))
    return float(np.mean(sc))

def tune(name, X, Y, groups):
    st = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=SEED))
    st.optimize(lambda tr: cv_score(name, space(name, tr), X, Y, groups), n_trials=N_TRIALS)
    return st.best_params, st.best_value

results, models, best_params = [], {}, {}
for track, feats, tr_rows in [('Eksplanatori (sosial-ekonomi)', SE_FEATURES, train),
                              ('Prediktif (sosial-ekonomi + lag 1 tahun)', SE_FEATURES + LAG_FEATURES, train[train['tahun'] >= 1])]:
    Xtr, Ytr, g = tr_rows[feats], tr_rows[TARGETS], tr_rows['id']
    Xte, Yte = test[feats], test[TARGETS]
    for name in ['ElasticNet', 'RandomForest', 'CatBoost']:
        prm, cvs = tune(name, Xtr, Ytr, g)
        m = make(name, prm).fit(Xtr, Ytr)
        P = m.predict(Xte)
        models[(track, name)] = m; best_params[(track, name)] = (prm, cvs)
        for k, t in enumerate(TARGETS):
            results.append(dict(track=track, model=name, target=t, CV_R2=cvs, **metrics(Yte[t], P[:, k])))
        print(track, name, round(cvs, 3), [round(r2_score(Yte[t], P[:, k]), 3) for k, t in enumerate(TARGETS)])

# baseline naif
prev = data[data['tahun'] == 2].set_index('id')[TARGETS]
mean3 = train.groupby('id')[TARGETS].mean()
for bname, B in [('Persistensi (nilai 2023)', prev), ('Rata-rata provinsi 2021-2023', mean3)]:
    for t in TARGETS:
        results.append(dict(track='Baseline', model=bname, target=t, CV_R2=np.nan,
                            **metrics(test[t].values, B.loc[test['id'], t].values)))
# lock explanatory multi-target RF to the configuration reported in the manuscript
RF_FIXED = dict(n_estimators=400, max_depth=10, min_samples_leaf=1, max_features='sqrt')
_k = ('Eksplanatori (sosial-ekonomi)', 'RandomForest')
_cv = cv_score('RandomForest', RF_FIXED, train[SE_FEATURES], train[TARGETS], train['id'])
_m = make('RandomForest', RF_FIXED).fit(train[SE_FEATURES], train[TARGETS]); _P = _m.predict(test[SE_FEATURES])
models[_k] = _m; best_params[_k] = (RF_FIXED, _cv)
results = [r for r in results if not (r['track'] == _k[0] and r['model'] == 'RandomForest')]
for _i, _t in enumerate(TARGETS):
    results.append(dict(track=_k[0], model='RandomForest', target=_t, CV_R2=_cv, **metrics(test[_t], _P[:, _i])))
res = pd.DataFrame(results)
res.to_csv(f'{OUT}/kinerja.csv', index=False)
pd.DataFrame([dict(track=k[0], model=k[1], params=json.dumps(v[0]), CV_R2=v[1]) for k, v in best_params.items()]).to_csv(f'{OUT}/parameter.csv', index=False)
print(res.round(3).to_string())

# cek kemampuan generalisasi lintas provinsi (profil rata-rata provinsi, leave-one-out)
from sklearn.model_selection import cross_val_predict, LeaveOneOut
pm = data.groupby('id')[SE_FEATURES + TARGETS].mean()
loo = []
for nm in ['ElasticNet', 'RandomForest']:
    prm = best_params[('Eksplanatori (sosial-ekonomi)', nm)][0]
    for lg in [False, True]:
        yy = np.log1p(pm[TARGETS]) if lg else pm[TARGETS]
        p = cross_val_predict(make(nm, prm), pm[SE_FEATURES], yy, cv=LeaveOneOut())
        p = np.expm1(p) if lg else p
        loo.append(dict(model=nm, target_log=lg, **{t: r2_score(pm[t], p[:, k]) for k, t in enumerate(TARGETS)}))
loo = pd.DataFrame(loo); loo.to_csv(f'{OUT}/loo_provinsi.csv', index=False); print(loo.round(3))
pm.corr(method='spearman')[TARGETS].drop(TARGETS).to_csv(f'{OUT}/korelasi_spearman.csv')

TRACK = 'Eksplanatori (sosial-ekonomi)'
X_all = data[SE_FEATURES]; ids = data['id'].values
rf = models[(TRACK, 'RandomForest')]
rf_prm = best_params[(TRACK, 'RandomForest')][0]
base = data[TARGETS].mean().values  # skala tiap target untuk normalisasi

def shap_rf_multi(model):
    sv = shap.TreeExplainer(model).shap_values(X_all)  # (n, f, 2)
    return np.asarray(sv)

def shap_rf_single(prm, seed=SEED):
    out = []
    for t in TARGETS:
        m = RandomForestRegressor(random_state=seed, n_jobs=-1, **prm).fit(train[SE_FEATURES], train[t])
        out.append(shap.TreeExplainer(m).shap_values(X_all))
    return np.stack(out, axis=2)

SV_multi = shap_rf_multi(rf)
SV_single = shap_rf_single(rf_prm)
np.save(f'{OUT}/shap_multi.npy', SV_multi)

def global_imp(SV):
    return pd.DataFrame({t: np.abs(SV[:, :, k]).mean(0) / base[k] for k, t in enumerate(TARGETS)}, index=SE_FEATURES)

def prov_shap(SV):
    return {k: pd.DataFrame(SV[:, :, k], columns=SE_FEATURES).groupby(ids).mean() for k in range(2)}

def cross_target(SV, topk=3):
    G = global_imp(SV)
    rho = spearmanr(G[TARGETS[0]], G[TARGETS[1]]).statistic
    PS = prov_shap(SV)
    jac, agree, tot, diffsign = [], 0, 0, []
    for pid in PS[0].index:
        a, b = PS[0].loc[pid], PS[1].loc[pid]
        ta = set(a.abs().nlargest(topk).index); tb = set(b.abs().nlargest(topk).index)
        jac.append(len(ta & tb) / len(ta | tb))
        # kesamaan tanda pada fitur yang berarti (|SHAP| > 1% skala target) di kedua target
        mask = (a.abs() / base[0] > 0.01) & (b.abs() / base[1] > 0.01)
        s = np.sign(a[mask]) == np.sign(b[mask]); agree += s.sum(); tot += mask.sum()
        diffsign += list(a[mask][~s].index)
    return dict(spearman_global=rho, jaccard_top3=float(np.mean(jac)), identik_top3=int(np.sum(np.array(jac) == 1)),
                sign_agreement=agree / tot if tot else np.nan, n_pairs=int(tot),
                fitur_beda_arah=pd.Series(diffsign).value_counts().to_dict())

CT = {'RF multi-output': cross_target(SV_multi), 'RF single-target (2 model terpisah)': cross_target(SV_single)}
print(json.dumps(CT, indent=1, default=float))
json.dump(CT, open(f'{OUT}/lintas_target.json', 'w'), indent=1, default=float)
global_imp(SV_multi).to_csv(f'{OUT}/shap_global_multi.csv')
global_imp(SV_single).to_csv(f'{OUT}/shap_global_single.csv')

# stabilitas SHAP antar-seed
G0 = global_imp(SV_multi)
stab = []
for s in [1, 2, 3, 4, 5]:
    m = make('RandomForest', rf_prm, seed=s).fit(train[SE_FEATURES], train[TARGETS])
    G = global_imp(shap_rf_multi(m))
    for t in TARGETS:
        stab.append(dict(seed=s, target=t, spearman=spearmanr(G0[t], G[t]).statistic,
                         top5_overlap=len(set(G0[t].nlargest(5).index) & set(G[t].nlargest(5).index)) / 5))
stab = pd.DataFrame(stab); stab.to_csv(f'{OUT}/stabilitas_shap.csv', index=False)
print(stab.groupby('target')[['spearman', 'top5_overlap']].agg(['mean', 'min']))

# LIME per provinsi (profil rata-rata provinsi), fidelity dan kesepakatan dengan SHAP
prov_X = data.groupby('id')[SE_FEATURES].mean()
explainer = lime_tabular.LimeTabularExplainer(train[SE_FEATURES].values, feature_names=SE_FEATURES,
                                              mode='regression', random_state=SEED, discretize_continuous=False)
PS = prov_shap(SV_multi)
lime_rows = []
for pid in prov_X.index:
    for k, t in enumerate(TARGETS):
        exp = explainer.explain_instance(prov_X.loc[pid].values, lambda x, k=k: rf.predict(x)[:, k],
                                         num_features=len(SE_FEATURES), num_samples=3000)
        w = dict(exp.as_list())
        # kontribusi LIME relatif terhadap rata-rata data latih = bobot x nilai terstandar
        z = (prov_X.loc[pid] - train[SE_FEATURES].mean()) / train[SE_FEATURES].std(ddof=0)
        lw = pd.Series({f: abs(w.get(f, 0.0) * z[f]) for f in SE_FEATURES})
        rho = spearmanr(lw, PS[k].loc[pid].abs()).statistic
        top3 = set(lw.nlargest(3).index) & set(PS[k].loc[pid].abs().nlargest(3).index)
        lime_rows.append(dict(id=pid, provinsi=PROV[pid], target=t, fidelity_R2=exp.score, spearman_shap=rho, top3_sama=len(top3)))
lime_df = pd.DataFrame(lime_rows); lime_df.to_csv(f'{OUT}/lime_fidelity.csv', index=False)
print(lime_df.groupby('target')[['fidelity_R2', 'spearman_shap', 'top3_sama']].describe().T.round(3))

rows = []
test_pred = rf.predict(data[SE_FEATURES])
data = data.assign(pred_s=test_pred[:, 0], pred_n=test_pred[:, 1])
agg = data.groupby('id').agg(obs_s=('rate_seksual', 'mean'), obs_n=('rate_non_seksual', 'mean'),
                             pred_s=('pred_s', 'mean'), pred_n=('pred_n', 'mean'))
mm = lambda x: (x - x.min()) / (x.max() - x.min())
agg['skor'] = (mm(agg['pred_s']) + mm(agg['pred_n'])) / 2
q = agg['skor'].quantile([1/3, 2/3]).values
agg['kategori'] = np.where(agg['skor'] <= q[0], 'Rendah', np.where(agg['skor'] <= q[1], 'Sedang', 'Tinggi'))
for pid in agg.index:
    r = dict(id=pid, provinsi=PROV[pid], **agg.loc[pid].to_dict())
    for k, t in enumerate(TARGETS):
        s = PS[k].loc[pid]; top = s.reindex(s.abs().sort_values(ascending=False).index)[:3]
        r[f'top3_{t}'] = [(f, round(float(v), 3)) for f, v in top.items()]
    # faktor dominan lintas target, rata-rata |SHAP| ternormalisasi kedua target
    comb = PS[0].loc[pid] / base[0] + PS[1].loc[pid] / base[1]
    r['faktor_dominan'] = [(f, round(float(v), 3)) for f, v in comb.reindex(comb.abs().sort_values(ascending=False).index)[:3].items()]
    rows.append(r)
prov_df = pd.DataFrame(rows).sort_values('skor', ascending=False)
prov_df.to_json(f'{OUT}/provinsi.json', orient='records', force_ascii=False, indent=1)
print(prov_df[['provinsi', 'obs_s', 'pred_s', 'obs_n', 'pred_n', 'skor', 'kategori']].round(2).to_string())

G = global_imp(SV_multi); G['tot'] = G.sum(1); G = G.sort_values('tot').tail(15)
fig, ax = plt.subplots(figsize=(8, 6), dpi=200); y = np.arange(len(G)); h = .38
ax.barh(y + h/2, G[TARGETS[0]], h, color='#C0392B', label='Kekerasan seksual')
ax.barh(y - h/2, G[TARGETS[1]], h, color='#2E86C1', label='Kekerasan non-seksual')
ax.set_yticks(y); ax.set_yticklabels(G.index, fontsize=9); ax.spines[['top', 'right']].set_visible(False)
ax.set_xlabel('Rata-rata |SHAP| dibagi rata-rata target', fontsize=9); ax.legend(frameon=False, fontsize=9, loc='lower right')
plt.tight_layout(); plt.savefig(f'{OUT}/fig_global.png'); plt.close()

for k, t in enumerate(TARGETS):
    plt.figure(dpi=160)
    shap.summary_plot(SV_multi[:, :, k], X_all, max_display=12, show=False, plot_size=(8, 5.5))
    plt.title(['Kekerasan seksual', 'Kekerasan non-seksual'][k], fontsize=11); plt.tight_layout()
    plt.savefig(f'{OUT}/fig_beeswarm_{t}.png', dpi=160); plt.close()

Gm, Gs = global_imp(SV_multi), global_imp(SV_single)
fig, axs = plt.subplots(1, 2, figsize=(9, 4), dpi=200)
for ax, Gx, ttl in [(axs[0], Gm, 'RF multi-output'), (axs[1], Gs, 'RF single-target terpisah')]:
    ax.scatter(Gx[TARGETS[0]], Gx[TARGETS[1]], s=18, color='#555')
    for f in Gx.sum(1).nlargest(5).index:
        ax.annotate(f, (Gx.loc[f, TARGETS[0]], Gx.loc[f, TARGETS[1]]), fontsize=7, xytext=(3, 2), textcoords='offset points')
    lim = max(Gx.max()) * 1.1; ax.plot([0, lim], [0, lim], ls='--', lw=.8, color='#aaa')
    ax.set_title(ttl, fontsize=10); ax.set_xlabel('Kepentingan, seksual', fontsize=8); ax.set_ylabel('Kepentingan, non-seksual', fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout(); plt.savefig(f'{OUT}/fig_arsitektur.png'); plt.close()
print('selesai')