"""Critic helpers (026-final-run). Imports r2r3/run.py by path (data, folds, bootstrap) without modifying it. No test-tuned choice is reported as a headline."""
import os,sys,json,importlib.util,time
for v in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"): os.environ.setdefault(v,"1")
from pathlib import Path
HERE=Path(__file__).resolve().parent; FR=HERE.parent; ROOT=FR.parents[1]
os.environ.setdefault("HF_HOME",str(ROOT/"data/hf_cache")); os.environ.setdefault("HF_HUB_OFFLINE","1")
os.chdir(ROOT)
import numpy as np
spec=importlib.util.spec_from_file_location("r2r3run",FR/"r2r3/run.py"); RR=importlib.util.module_from_spec(spec); sys.modules["r2r3run"]=RR; spec.loader.exec_module(RR)
CACHE=HERE/"cache"; CACHE.mkdir(exist_ok=True)
CLASSES=RR.CFG["classes"]; K=8; PRIM="jpeg75_resize05"; TMIX=["clean",PRIM,"social_sim"]
def data(): return RR.load_data()
def boot(D):
    r=RR.r20(); Wc,inv,ug=r.cluster_weights(D["groups"],1000,0,r.strata_of); return Wc[:,inv]
def bacc_ci(Wrow,y,pred,K_=K,idx=None):
    r=RR.r20()
    a=r.conf_stats(r.boot_conf(Wrow,y,pred,K_))["bacc"]; return RR.ci(a),a
def bacc(y,pred,classes=None):
    cl=np.unique(y) if classes is None else classes
    return float(np.mean([(pred[y==c]==c).mean() for c in cl]))
def getblk(manip,cond,b): return np.asarray(RR.getfeat(manip,cond)[:,b-1,:])
def fit_lr(X,y,C,max_iter=300):
    from sklearn.preprocessing import StandardScaler; from sklearn.linear_model import LogisticRegression
    sc=StandardScaler().fit(X); clf=LogisticRegression(C=C,max_iter=max_iter,class_weight="balanced").fit(sc.transform(X),y); return sc,clf
def cv_task(a):
    """generic: per-fold fit on stacked train conditions; returns decision scores on test for each eval key"""
    Xtr_list,y,fold,f,C,evals,classes=a
    tr=np.where((fold!=f)&np.isin(y,classes))[0]; te=np.where(fold==f)[0]
    Xtr=np.concatenate([X[tr] for X in Xtr_list]); ytr=np.tile(y[tr],len(Xtr_list))
    sc,clf=fit_lr(Xtr,ytr,C)
    return f,te,{k:clf.predict(sc.transform(X[te])) for k,X in evals.items()}
def cv_pred(Xtr_list,y,fold,evals,C=0.01,classes=None,workers=5):
    from concurrent.futures import ProcessPoolExecutor
    classes=np.unique(y) if classes is None else np.array(classes)
    out={k:np.full(len(y),-1,int) for k in evals}
    tasks=[(Xtr_list,y,fold,f,C,evals,classes) for f in range(5)]
    with ProcessPoolExecutor(workers) as ex:
        for f,te,p in ex.map(cv_task,tasks):
            for k in evals: out[k][te]=p[k]
    return out
def log(*a): print(time.strftime("%H:%M:%S"),*a,flush=True)
