"""Critic extraction: (A) launderings outside the JPEG+resize family (test only, T-mix probe never sees them) and (B) stronger input manipulations
(tone/histogram normalisation, blur dose, patch-aligned shuffle, phase/amplitude scrambles, combinations), for T-mix retraining. DINOv2-base; blocks 4,6,8,12 saved."""
import sys,os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from common import *
from PIL import Image, ImageFilter, ImageEnhance
from concurrent.futures import ThreadPoolExecutor
from scipy.ndimage import gaussian_filter
from perturb import apply_condition, jpeg75_resize05, jpeg, seed_from_key
BL=[4,6,8,12]
D=data(); paths,shas=D["paths"],D["shas"]
def arr(i): return np.asarray(i.convert("RGB")).astype(np.float32)
def toim(a): return Image.fromarray(np.clip(np.rint(a),0,255).astype(np.uint8))
# ---- launderings on the 512 image
def gamma(g): return lambda im,k: toim(255*(arr(im)/255)**g)
def filt(im,k): return ImageEnhance.Color(ImageEnhance.Contrast(im).enhance(1.3)).enhance(1.3)
def warm(im,k): a=arr(im); a[...,0]*=1.12; a[...,2]*=0.88; return toim(a)
def noise(im,k): r=np.random.default_rng(seed_from_key("n"+k)); return jpeg(toim(arr(im)+r.normal(0,8,(im.size[1],im.size[0],3))),90)
def rot(im,k): 
    return im.rotate(5,resample=Image.BICUBIC,reflect=False) if False else im.rotate(5,resample=Image.BICUBIC)
def unsharp(im,k): return im.filter(ImageFilter.UnsharpMask(radius=2,percent=150,threshold=0))
def webp(im,k):
    import io; b=io.BytesIO(); im.save(b,"WEBP",quality=50); b.seek(0); o=Image.open(b); o.load(); return o.convert("RGB")
def median(im,k): return im.filter(ImageFilter.MedianFilter(3))
def pixel(im,k): w,h=im.size; return im.resize((w//2,h//2),Image.NEAREST).resize((w,h),Image.NEAREST)
def blur_pre(im,k): return im.filter(ImageFilter.GaussianBlur(2))
LA={"gamma07":gamma(.7),"gamma14":gamma(1.4),"filter_cs13":filt,"warm":warm,"noise8_q90":noise,"rot5":rot,"unsharp":unsharp,"webp50":webp,"median3":median,"nearest_x05":pixel,"blur2_pre":blur_pre}
def launder(name,im,k):
    if name.endswith("+P"): return jpeg75_resize05(LA[name[:-2]](im.convert("RGB"),k))
    return LA[name](im.convert("RGB"),k)
# ---- manipulations on the 224 input
def eq_channel(c):
    h=np.bincount(c.ravel(),minlength=256); cdf=h.cumsum()/c.size; return (255*cdf)[c]
def eq(a): a=a.astype(np.uint8); return np.stack([eq_channel(a[...,i]) for i in range(3)],-1)
def gray(a): return np.repeat(np.asarray(toim(a).convert("L"))[...,None],3,-1).astype(np.float32)
def meanstd(a): m=a.mean((0,1),keepdims=True); s=a.std((0,1),keepdims=True)+1e-3; return (a-m)/s*50+128
def locnorm(a,t=32):
    o=a.copy()
    for y in range(0,224,t):
        for x in range(0,224,t):
            p=a[y:y+t,x:x+t]; m=p.mean((0,1),keepdims=True); s=p.std((0,1),keepdims=True)+1e-3; o[y:y+t,x:x+t]=(p-m)/s*50+128
    return o
def shuf(a,k,t):
    n=224//t; ti=a.reshape(n,t,n,t,3).transpose(0,2,1,3,4).reshape(n*n,t,t,3)
    p=np.random.default_rng(seed_from_key(f"tile{t}:{k}")).permutation(n*n)
    return ti[p].reshape(n,n,t,t,3).transpose(0,2,1,3,4).reshape(224,224,3)
def blur(a,s): return gaussian_filter(a,sigma=(s,s,0),mode="reflect",truncate=4.0)
def phase_scr(a,k):
    r=np.random.default_rng(seed_from_key("ph"+k)); ph=np.angle(np.fft.fft2(r.random((224,224))))        # same random phase in all channels keeps colour correlations
    o=np.zeros_like(a)
    for i in range(3):
        F=np.fft.fft2(a[...,i]); o[...,i]=np.real(np.fft.ifft2(np.abs(F)*np.exp(1j*(np.angle(F)+ph))))
    return o
def amp_white(a):
    o=np.zeros_like(a); fy=np.fft.fftfreq(224)[:,None]; fx=np.fft.fftfreq(224)[None,:]; f=np.sqrt(fx**2+fy**2); f[0,0]=1
    target=(1/f); target[0,0]=0
    for i in range(3):
        F=np.fft.fft2(a[...,i]-a[...,i].mean()); x=target*np.exp(1j*np.angle(F)); x=np.real(np.fft.ifft2(x)); o[...,i]=(x-x.mean())/(x.std()+1e-6)*50+128
    return o
MA={"none":lambda a,k:a,"eq":lambda a,k:eq(a),"eqgray":lambda a,k:eq(gray(a)),"meanstd":lambda a,k:meanstd(a),"locnorm32":lambda a,k:locnorm(a),
    "shuf14":lambda a,k:shuf(a,k,14),"shuf32":lambda a,k:shuf(a,k,32),"blur3":lambda a,k:blur(a,3),"blur6":lambda a,k:blur(a,6),"blur12":lambda a,k:blur(a,12),"blur24":lambda a,k:blur(a,24),
    "shuf14_eqgray":lambda a,k:shuf(eq(gray(a)),k,14),"shuf32_blur15_eqgray":lambda a,k:shuf(eq(gray(blur(a,1.5))),k,32),
    "shuf14_blur15_eqgray":lambda a,k:shuf(eq(gray(blur(a,1.5))),k,14),"phase_scr":phase_scr,"amp_white":lambda a,k:amp_white(a)}
def render_base(cond):
    f=CACHE/f"base_{RR.safe(cond)}.npy"
    if f.exists(): return np.load(f,mmap_mode="r")
    def one(i):
        with Image.open(paths[i]) as im:
            im.load()
            o=launder(cond,im,shas[i]) if cond.split("+")[0] in LA else apply_condition(im,cond,key=shas[i],seed=0)
        return np.asarray(RR.crop224(o))
    with ThreadPoolExecutor(12) as ex: X=np.stack(list(ex.map(one,range(len(paths)))))
    np.save(f,X); return X
emb=None
def embed(arrs):
    global emb
    emb=emb or RR.Embedder(batch=64)
    imgs=[Image.fromarray(np.clip(np.rint(a),0,255).astype(np.uint8)) for a in arrs]
    x=emb.embed(imgs); return x[:,[b-1 for b in BL],:].astype(np.float32)
def run(tag,cond,manip):
    f=CACHE/f"f_{manip}__{RR.safe(cond)}.npy"
    if f.exists(): return
    t=time.time(); B=render_base(cond)
    with ThreadPoolExecutor(8) as ex: A=list(ex.map(lambda i:MA[manip](B[i].astype(np.float32),shas[i]),range(len(shas))))
    np.save(f,embed(A)); log(tag,manip,cond,f"{time.time()-t:.0f}s")
if __name__=="__main__":
    run("base","clean","none")
    ref=np.asarray(RR.getfeat("none","clean"))[:,5,:]; mine=np.load(CACHE/"f_none__clean.npy")[:,1,:]
    log("parity vs r2r3 cache (block 6, max abs, scaled):",float(np.abs(ref-mine).max()),float((np.abs(ref-mine)/np.maximum(1,np.abs(ref).max(0,keepdims=True))).max()))
    conds=["clean",PRIM,"social_sim","screenshot_sim"]
    for l in LA:
        run("launder",l,"none"); run("launder",l+"+P","none")
    for m in MA:
        for c in conds: run("manip",c,m)
    log("done")
