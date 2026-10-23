"""Critic check 2: does the inverted drift track image content (contrast / high-frequency energy of the exact 224 input)?
Reads the clean 512 crops (both arms), 224 center crop, computes per-image stats, merges with setting-i post scores."""
import sys, numpy as np, pandas as pd
from PIL import Image
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
sys.path.insert(0, 'experiments/lib')
from preprocess import center_crop
d = pd.read_csv('experiments/006-format-shortcut/pilot/scores_long.csv.gz')
d = d[(d.setting=='i')&(d.feat=='post')&(d.split=='test')&(d.pool.isin(['vision_native','fake','vision_whatsapp','vision_facebook_hq']))]
out = []
for arm, f in [('orig','experiments/004-phase4-features/pilot/splits_pilot.csv'), ('fov512','experiments/004-phase4-features/pilot/splits_pilot_fov512.csv')]:
    sp = pd.read_csv(f, keep_default_na=False).set_index('sha256')
    for sha in d[d.arm==arm].sha256.unique():
        with Image.open(sp.loc[sha,'path']) as im:
            x = np.asarray(center_crop(im.convert('RGB'),224), np.float32)/255.
        g = x.mean(2)
        lap = np.abs(4*g[1:-1,1:-1]-g[:-2,1:-1]-g[2:,1:-1]-g[1:-1,:-2]-g[1:-1,2:]).mean()
        grad = np.hypot(np.diff(g,axis=0)[:,:-1], np.diff(g,axis=1)[:-1]).mean()
        clip = ((x<=0.0)|(x>=1.0)).mean()
        out.append(dict(arm=arm, sha256=sha, pix_std=g.std(), lap=lap, grad=grad, clipfrac=clip))
st = pd.DataFrame(out)
st.to_csv('experiments/006-format-shortcut/pilot/critic_checks/c2_stats.csv.gz', index=False)
m = d.merge(st, on=['arm','sha256'])
print('median stats by arm/pool:'); print(m[m.sigma==0.01].groupby(['arm','pool'])[['pix_std','lap','grad','clipfrac']].median().round(4))
for arm in ['orig','fov512']:
    for sg in [0.01,0.05]:
        s = m[(m.arm==arm)&(m.sigma==sg)]
        for pool in ['vision_native','fake']:
            q = s[s.pool==pool]
            print(arm, sg, pool, 'spearman(score, lap)=%.2f  (score, pix_std)=%.2f' % (spearmanr(q.score,q.lap)[0], spearmanr(q.score,q.pix_std)[0]))
        q = s[s.pool.isin(['vision_native','fake'])]
        y = (q.label==1).astype(int)
        print('   AUROC score=%.3f | AUROC(-lap)=%.3f AUROC(-pix_std)=%.3f AUROC(-grad)=%.3f' % (roc_auc_score(y,q.score), roc_auc_score(y,-q.lap), roc_auc_score(y,-q.pix_std), roc_auc_score(y,-q.grad)))
        # stratify by lap quintile (pooled over both classes): within-stratum AUROC
        q = q.assign(b=pd.qcut(q.lap, 5, labels=False))
        res=[]
        for b,qq in q.groupby('b'):
            if qq.label.nunique()==2: res.append((b, int((qq.label==0).sum()), int((qq.label==1).sum()), round(roc_auc_score(qq.label, qq.score),3)))
        print('   within lap-quintile AUROC (b, nreal, nfake, auc):', res)
        # residualise score on log(lap) & log(pix_std) fitted on reals+fakes jointly via OLS
        X = np.c_[np.ones(len(q)), np.log(q.lap), np.log(q.pix_std+1e-4)]
        beta = np.linalg.lstsq(X, np.log(q.score), rcond=None)[0]
        r = np.log(q.score) - X@beta
        print('   AUROC of residual log-score after regressing on log lap, log pix_std: %.3f' % roc_auc_score(q.label, r))
