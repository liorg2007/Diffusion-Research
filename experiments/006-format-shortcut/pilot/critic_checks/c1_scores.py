"""Critic check 1: label alignment vs splits, recomputed AUROCs (sklearn), per-generator AUROC, score medians per class/arm."""
import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score
d = pd.read_csv('experiments/006-format-shortcut/pilot/scores_long.csv.gz')
sp = pd.read_csv('experiments/004-phase4-features/pilot/splits_pilot.csv', keep_default_na=False)
m = d.merge(sp[['sha256','label','generator','pool','split']].drop_duplicates('sha256'), on='sha256', suffixes=('','_sp'))
print('rows merged', len(m), 'of', len(d))
for c in ['label','pool','split']:
    print(c, 'mismatches', int((m[c].astype(str) != m[c+'_sp'].astype(str)).sum()))
gens = sorted(d[d.label==1].generator.unique())
def macro(sub, gl):
    R = sub[(sub.label==0)&(sub.pool=='vision_native')&(sub.split=='test')].score.values
    return np.mean([roc_auc_score(np.r_[np.zeros(len(R)),np.ones((sub.generator==g).sum())], np.r_[R, sub[sub.generator==g].score.values]) for g in gl])
print('\nrecomputed LOGO-macro AUROC (post):')
for arm in ['orig','fov512']:
  for st in ['i','iii-p','iv-p']:
    row=[]
    for sg in [0.01,0.02,0.05]:
      sub = d[(d.arm==arm)&(d.setting==st)&(d.feat=='post')&(d.sigma==sg)]
      row.append(f"{macro(sub,gens):.3f}/{macro(sub,[g for g in gens if g!='firefly']):.3f}")
    print(arm, st, row)
print('\nper-generator AUROC, setting i, post, sigma .01 (native test reals):')
for arm in ['orig','fov512']:
  sub = d[(d.arm==arm)&(d.setting=='i')&(d.feat=='post')&(d.sigma==0.01)]
  R = sub[(sub.label==0)&(sub.pool=='vision_native')&(sub.split=='test')].score.values
  print(arm, {g: round(roc_auc_score(np.r_[np.zeros(len(R)),np.ones(60)], np.r_[R, sub[sub.generator==g].score.values]),3) for g in gens})
print('\nmedian score by class/pool (setting i, post):')
s = d[(d.setting=='i')&(d.feat=='post')&(d.split=='test')]
print(s.groupby(['sigma','arm','pool']).score.median().unstack('arm').round(4))
print('\nmedian fake score per generator (setting i, post, s=.01):')
print(s[(s.sigma==0.01)&(s.label==1)].groupby(['generator','arm']).score.median().unstack().round(4))
print('\nmedian score settings (post, s=.01):')
s2=d[(d.feat=='post')&(d.sigma==0.01)&(d.split=='test')&(d.pool.isin(['vision_native','fake']))]
print(s2.groupby(['arm','setting','pool']).score.median().unstack('setting').round(4))
