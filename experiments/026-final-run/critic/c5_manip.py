"""(A) T-mix probe (block 6) tested on launderings outside the JPEG+resize family (never in training). (B) stronger manipulations, T-mix retrained per manipulation; frozen unmanipulated probe also tested."""
import sys,os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import *
D=data(); y,fold=D["y"],D["fold"]; W=boot(D); n=len(y); BL=[4,6,8,12]
def L(m,c,b=6): return np.load(CACHE/f"f_{m}__{RR.safe(c)}.npy",mmap_mode="r")[:,BL.index(b),:]
LA=["gamma07","gamma14","filter_cs13","warm","noise8_q90","rot5","unsharp","webp50","median3","nearest_x05","blur2_pre"]
def ref(c,b=6): return getblk("none",c,b)
for b in (6,):
    print(f"## A. Frozen T-mix probe block {b}, C=.01, new launderings. Native-scale column compares with clean; '+P' = then jpeg75_resize05")
    ev={"clean":ref("clean",b),PRIM:ref(PRIM,b),"shot":ref("screenshot_sim",b)}
    for l in LA:
        ev[l]=np.asarray(L("none",l,b)); ev[l+"+P"]=np.asarray(L("none",l+"+P",b))
    X_t=[ref(c,b) for c in TMIX]
    p=cv_pred(X_t,y,fold,ev,C=0.01)
    pc=cv_pred([ref("clean",b)],y,fold,ev,C=0.01)
    pa=cv_pred(X_t+[np.asarray(L("none",l,b)) for l in ("noise8_q90","rot5")],y,fold,ev,C=0.01)
    print("ref clean",round(bacc(y,p["clean"]),3),"primary",round(bacc(y,p[PRIM]),3),"| clean-only probe: clean",round(bacc(y,pc["clean"]),3),"primary",round(bacc(y,pc[PRIM]),3))
    print("laundering | T-mix native [CI] | T-mix +P [CI] | clean-only probe native | clean-only +P")
    for l in LA:
        c1,_=bacc_ci(W,y,p[l]); c2,_=bacc_ci(W,y,p[l+"+P"])
        print(f"{l} | {c1[0]:.3f} [{c1[1]:.3f},{c1[2]:.3f}] | {c2[0]:.3f} [{c2[1]:.3f},{c2[2]:.3f}] | {bacc(y,pc[l]):.3f} | {bacc(y,pc[l+'+P']):.3f}",flush=True)
ms=["none","shuf32","eq","eqgray","meanstd","locnorm32","shuf14","blur3","blur6","blur12","blur24","shuf14_eqgray","shuf32_blur15_eqgray","shuf14_blur15_eqgray","phase_scr","amp_white"]
print("\n## B. manipulation of the 224 input after laundering; T-mix probe RETRAINED per manipulation; bacc@primary b6 [CI] | retained (none=C.01 same block) | b4 | b8 | b12 | @screenshot b6 | frozen none-probe tested on it (b6)")
Xn=[np.asarray(L("none",c)) if (CACHE/f"f_none__{RR.safe(c)}.npy").exists() else ref(c) for c in TMIX]
frozen={m:np.asarray(L(m,PRIM)) for m in ms}
pf=cv_pred(Xn,y,fold,frozen,C=0.01)
base=None
for m in ms:
    row={}
    pp=cv_pred([np.asarray(L(m,c)) for c in TMIX],y,fold,{PRIM:np.asarray(L(m,PRIM)),"shot":np.asarray(L(m,"screenshot_sim")),"none":ref(PRIM)},C=0.01)
    ci_,a=bacc_ci(W,y,pp[PRIM])
    if m=="none": base=a
    for b in (4,8,12):
        q=cv_pred([np.asarray(L(m,c,b)) for c in TMIX],y,fold,{PRIM:np.asarray(L(m,PRIM,b))},C=0.01); row[b]=bacc(y,q[PRIM])
    rc=RR.ci((a-.125)/(base-.125))
    print(f"{m} | {ci_[0]:.3f} [{ci_[1]:.3f},{ci_[2]:.3f}] | ret {rc[0]:.3f} [{rc[1]:.3f},{rc[2]:.3f}] | {row[4]:.3f} | {row[8]:.3f} | {row[12]:.3f} | {bacc(y,pp['shot']):.3f} | frozen {bacc(y,pf[m]):.3f}",flush=True)
    if m in("eqgray","shuf14_blur15_eqgray"):
        print("   recall",{CLASSES[c]:round((pp[PRIM][y==c]==c).mean(),3) for c in range(8)})
