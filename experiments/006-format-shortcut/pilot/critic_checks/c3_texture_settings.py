"""Critic check 3: texture (HF energy) of the exact 224 input in every setting/arm, and RIGID AUROC after removing
what log-HF-energy predicts. Residual model fitted on REAL rows only (variant R) or on all rows (variant A).
Also per-device real drift vs texture. Descriptive, no bootstrap."""
import sys, importlib.util, numpy as np, pandas as pd
from PIL import Image
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
sys.path.insert(0, 'experiments/lib'); sys.path.insert(0, 'experiments/004-phase4-features')
from preprocess import center_crop
spec = importlib.util.spec_from_file_location("p4run", "experiments/004-phase4-features/run.py")
p4 = importlib.util.module_from_spec(spec); spec.loader.exec_module(p4)
COND = {'i': 'clean', 'iii-p': 'jpeg(90)', 'iv-p': 'jpeg75_resize05'}
d = pd.read_csv('experiments/006-format-shortcut/pilot/scores_long.csv.gz')
d = d[(d.feat=='post')&(d.split=='test')&(d.pool.isin(['vision_native','fake']))]
def lap_of(img):
    g = np.asarray(center_crop(img.convert('RGB'),224), np.float32).mean(2)/255.
    return np.abs(4*g[1:-1,1:-1]-g[:-2,1:-1]-g[2:,1:-1]-g[1:-1,:-2]-g[1:-1,2:]).mean()
rows=[]
for arm, f in [('orig','experiments/004-phase4-features/pilot/splits_pilot.csv'), ('fov512','experiments/004-phase4-features/pilot/splits_pilot_fov512.csv')]:
    sp = pd.read_csv(f, keep_default_na=False).set_index('sha256')
    for st, c in COND.items():
        for sha in d[(d.arm==arm)&(d.setting==st)].sha256.unique():
            rows.append(dict(arm=arm, setting=st, sha256=sha, lap=lap_of(p4.load_cond(sp.loc[sha,'path'], sha, c))))
T = pd.DataFrame(rows); T.to_csv('experiments/006-format-shortcut/pilot/critic_checks/c3_lap.csv.gz', index=False)
m = d.merge(T, on=['arm','setting','sha256'])
print('median lap (HF energy) by arm/setting/class:')
print(m[m.sigma==0.01].groupby(['arm','setting','pool']).lap.median().unstack('setting').round(4))
gens = sorted(m[m.label==1].generator.unique())
def macro(q, col):
    R = q[q.label==0][col].values
    return np.mean([roc_auc_score(np.r_[np.zeros(len(R)),np.ones(60)], np.r_[R, q[q.generator==g][col].values]) for g in gens])
print('\nLOGO-macro AUROC (with Firefly): raw | AUROC(-lap) | residual R (fit on reals) | residual A (fit on all)')
res = {}
for arm in ['orig','fov512']:
    for sg in [0.01,0.02,0.05]:
        line=[]
        for st in COND:
            q = m[(m.arm==arm)&(m.sigma==sg)&(m.setting==st)].copy()
            X = np.c_[np.ones(len(q)), np.log(q.lap)]; y = np.log(q.score)
            bR = np.linalg.lstsq(X[q.label.values==0], y[q.label.values==0], rcond=None)[0]
            bA = np.linalg.lstsq(X, y, rcond=None)[0]
            q['rR'], q['rA'], q['nl'] = y - X@bR, y - X@bA, -q.lap
            v = (macro(q,'score'), macro(q,'nl'), macro(q,'rR'), macro(q,'rA')); res[(arm,sg,st)] = v
            line.append(f"{st}: {v[0]:.3f} | {v[1]:.3f} | {v[2]:.3f} | {v[3]:.3f}")
        print(arm, sg, ' || '.join(line))
print('\ngap (i)-(iii-p): raw vs residual-R; orig-fov512 of AUROC(i): raw vs residual-R')
for sg in [0.01,0.02,0.05]:
    print(sg, {arm: (round(res[(arm,sg,'i')][0]-res[(arm,sg,'iii-p')][0],3), round(res[(arm,sg,'i')][2]-res[(arm,sg,'iii-p')][2],3)) for arm in ['orig','fov512']},
          'orig-fov512:', round(res[('orig',sg,'i')][0]-res[('fov512',sg,'i')][0],3), round(res[('orig',sg,'i')][2]-res[('fov512',sg,'i')][2],3))
# per-device
q = m[(m.arm=='orig')&(m.sigma==0.01)&(m.setting=='i')&(m.label==0)]
pdv = q.groupby('device').agg(n=('score','size'), med_score=('score','median'), med_lap=('lap','median'))
F = m[(m.arm=='orig')&(m.sigma==0.01)&(m.setting=='i')&(m.label==1)].score.values
pdv['auroc_vs_all_fakes'] = [roc_auc_score(np.r_[np.zeros(len(g)),np.ones(len(F))], np.r_[g.score.values, F]) for _, g in q.groupby('device')]
print('\nper-device (orig, setting i, s=.01):'); print(pdv.sort_values('med_score').round(4).to_string())
print('spearman across devices (med_score, med_lap): %.2f' % spearmanr(pdv.med_score, pdv.med_lap)[0])
print('devices with AUROC>0.5:', int((pdv.auroc_vs_all_fakes>0.5).sum()), 'of', len(pdv))
