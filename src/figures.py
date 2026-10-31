"""Reproduces Figures 3 to 6 of the paper from the saved results. Run from the repository root:
    python src/figures.py
"""
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt, shap
LBL={
 "PDRB_per_kapita_HK": "GRDP per capita (constant)",
 "PDRB_per_kapita": "GRDP per capita",
 "PPKMLA": "PPKMLA",
 "IPG": "Gender Development Index",
 "miskin_rate": "Poverty rate",
 "ppm_rate": "ppm_rate",
 "tk_sekolah_per_1000": "Kindergartens per 1,000",
 "tk_murid_per_1000": "Kindergarten pupils per 1,000",
 "pt_per_1000": "HEIs per 1,000",
 "dosen_per_1000": "Lecturers per 1,000",
 "mahasiswa_per_1000": "Students per 1,000",
 "mahasiswa_per_PT": "Students per HEI",
 "dosen_per_PT": "Lecturers per HEI",
 "murid_per_sekolah": "Pupils per school",
 "guru_per_sekolah": "Teachers per school",
 "murid_per_guru": "Pupil-teacher ratio",
 "smk_density": "Vocational school density",
 "pt_density": "HEI density",
 "edu_index": "Education index",
 "RLS_gap": "Schooling gap",
 "PPDPSK_avg": "Senior high school attainment",
 "school_density": "School density",
 "log_populasi": "Population (log)"
}
MAIN='results/main/'; VAL='results/validation/'; OUT='figures/'
C_S,C_N='#B03A2E','#1F618D'
# Fig 3 global importance
g=pd.read_csv(MAIN+'shap_global_multi.csv',index_col=0); g['tot']=g.sum(axis=1); g=g.sort_values('tot').tail(15)
fig,ax=plt.subplots(figsize=(7.5,5.6),dpi=300); y=np.arange(len(g)); h=.38
ax.barh(y+h/2,g['rate_seksual'],h,color=C_S,label='Sexual violence'); ax.barh(y-h/2,g['rate_non_seksual'],h,color=C_N,label='Non-sexual violence')
ax.set_yticks(y); ax.set_yticklabels([LBL[i] for i in g.index],fontsize=9); ax.spines[['top','right']].set_visible(False)
ax.set_xlabel('Mean |SHAP| / mean of target',fontsize=9); ax.legend(frameon=False,fontsize=9,loc='lower right'); ax.tick_params(axis='x',labelsize=8)
plt.tight_layout(); plt.savefig(OUT+'fig3_global_importance.png'); plt.close()
# Fig 4 beeswarm
d=pd.read_csv('data/processed/dataset_bersih.csv'); d=d[~d.invalid]
SE=list(LBL.keys()); X=d[SE].rename(columns=LBL)
SV=np.load(MAIN+'shap_multi.npy')
fig=plt.figure(figsize=(14,5.2),dpi=250)
for k,(ttl) in enumerate(['(A) Sexual violence','(B) Non-sexual violence']):
    ax=fig.add_subplot(1,2,k+1); plt.sca(ax)
    shap.summary_plot(SV[:,:,k],X,max_display=10,show=False,plot_size=None,color_bar=(k==1))
    ax.set_title(ttl,fontsize=12,loc='left'); ax.tick_params(labelsize=9); ax.set_xlabel('SHAP value (victims per 100,000)',fontsize=9)
plt.tight_layout(); plt.savefig(OUT+'fig4_shap_summary.png'); plt.close()
# Fig 5 bootstrap
b=pd.read_csv(VAL+'bootstrap_ringkasan.csv'); FAM={'RandomForest':'Random Forest','CatBoost':'CatBoost','ElasticNet':'ElasticNet'}; MET={'spearman':'Spearman','cosine':'Cosine','jaccard_top3':'Jaccard top-3'}
fig,ax=plt.subplots(figsize=(7,3.8),dpi=300); yy=0; ticks=[]
for f in FAM:
    for m in MET:
        r=b[(b.model==f)&(b.metrik==m)].iloc[0]
        ax.plot([r.CI_bawah,r.CI_atas],[yy,yy],color='#444',lw=1.5); ax.plot(r.selisih,yy,'o',color=C_S,ms=5)
        ticks.append((yy,f'{FAM[f]}, {MET[m]}')); yy-=1
    yy-=0.6
ax.axvline(0,color='#999',ls='--',lw=.8); ax.set_yticks([a for a,_ in ticks]); ax.set_yticklabels([b_ for _,b_ in ticks],fontsize=8)
ax.set_xlabel('Similarity difference (multi-target minus single-target), 95% interval',fontsize=8); ax.tick_params(axis='x',labelsize=8)
ax.spines[['top','right']].set_visible(False); plt.tight_layout(); plt.savefig(OUT+'fig5_bootstrap.png'); plt.close()
# Fig 6 synthetic
s=pd.read_csv(VAL+'sintetis_ringkasan.csv'); SC={'Sama':'Shared','Sebagian':'Partial','Berbeda':'Distinct'}
fig,axs=plt.subplots(3,3,figsize=(9,7.5),dpi=300,sharex=True,sharey=True)
for i,sc in enumerate(SC):
    for j,f in enumerate(FAM):
        ax=axs[i,j]; q=s[(s.skenario==sc)&(s.model==f)]
        for v,c,lb in [('MT',C_S,'Multi-target'),('ST',C_N,'Single-target')]:
            qq=q[q.versi==v].sort_values('n'); ax.plot(qq.n,qq.cos_estimasi,'o-',color=c,ms=4,lw=1.4,label=lb)
        ax.axhline(q.cos_sebenarnya.iloc[0],color='#333',ls='--',lw=1,label='True similarity')
        ax.set_xscale('log'); ax.set_ylim(-0.05,1.05); ax.spines[['top','right']].set_visible(False); ax.tick_params(labelsize=7)
        if i==0: ax.set_title(FAM[f],fontsize=10)
        if j==0: ax.set_ylabel(f'{SC[sc]} drivers\nmeasured similarity',fontsize=8)
        if i==2: ax.set_xlabel('Training sample size',fontsize=8)
axs[0,0].legend(fontsize=7,frameon=False,loc='lower left'); plt.tight_layout(); plt.savefig(OUT+'fig6_synthetic.png'); plt.close()
print('ok')
