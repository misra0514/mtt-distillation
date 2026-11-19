"""Execute the kernel's Python AST using NumPy pointer emulation.
Checks indexing/masks/scheduling only, NOT Triton compilation or GPU numerics.
Run with a Python environment containing NumPy; no torch/triton required.
"""
import ast
from pathlib import Path
import numpy as np

class Ptr:
    def __init__(self,a,offset=0): self.a,self.offset=a.reshape(-1),offset
    def __add__(self,other): return Ptr(self.a,self.offset+other)
    __radd__=__add__

class TL:
    float32=np.float64  # isolate address correctness from rounding
    ids=(0,0)
    @classmethod
    def program_id(cls,axis): return cls.ids[axis]
    @staticmethod
    def cdiv(a,b): return (a+b-1)//b
    minimum=staticmethod(np.minimum)
    arange=staticmethod(np.arange)
    zeros=staticmethod(np.zeros)
    static_range=staticmethod(range)
    @staticmethod
    def load(p,mask,other):
        off,mask=np.broadcast_arrays(p.offset,mask)
        out=np.full(off.shape,float(other))
        assert np.all((off[mask]>=0)&(off[mask]<p.a.size))
        out[mask]=p.a[off[mask]]
        return out
    @staticmethod
    def store(p,value,mask):
        off,mask=np.broadcast_arrays(p.offset,mask)
        assert np.all((off[mask]>=0)&(off[mask]<p.a.size))
        p.a[off[mask]]=value[mask]
    @staticmethod
    def dot(a,b,acc,input_precision): return a@b+acc


def main():
    tree=ast.parse(Path(__file__).with_name('group_first_conv.py').read_text())
    fun=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_conv')
    fun.decorator_list=[]
    for a in fun.args.args: a.annotation=None
    ns={'tl':TL}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fun],type_ignores=[])),'kernel','exec'),ns)
    rng=np.random.default_rng(42)
    count=0
    # O=145 covers multiple scheduling chunks; odd sizes cover tail masks.
    for g,b,c,o,h,w,r,s,sh,sw,ph,pw,dh,dw in [
        (3,2,5,7,9,11,3,2,2,1,2,1,2,1),
        (2,3,3,17,5,7,1,1,1,1,0,0,1,1),
        (1,1,1,1,3,3,3,3,1,1,1,1,1,1),
        (2,1,33,145,4,5,3,3,1,1,1,1,1,1)]:
        p=(h+2*ph-dh*(r-1)-1)//sh+1; q=(w+2*pw-dw*(s-1)-1)//sw+1
        x=rng.normal(size=(g,b,c,h,w)); wt=rng.normal(size=(g,o,c,r,s)); bias=rng.normal(size=(g,o))
        ref=np.zeros((g,b,o,p,q))
        for yy in range(p):
            for xx in range(q):
                for rr in range(r):
                    for ss in range(s):
                        iy,ix=yy*sh-ph+rr*dh,xx*sw-pw+ss*dw
                        if 0<=iy<h and 0<=ix<w:
                            ref[:,:,:,yy,xx]+=np.einsum('gbc,goc->gbo',x[:,:,:,iy,ix],wt[:,:,:,rr,ss])
        for has_bias in (False,True):
            expected=ref+bias[:,None,:,None,None] if has_bias else ref
            for rs in (False,True):
                for tm,tn,tk in [(16,64,32),(32,64,32),(32,128,32),(64,64,32),(32,64,64),(64,128,32)]:
                    y=np.full(expected.shape,np.nan)
                    for pid in range(TL.cdiv(o,tm)*TL.cdiv(p*q,tn)):
                        for gb in range(g*b):
                            TL.ids=(pid,gb)
                            ns['_conv'](Ptr(x),Ptr(wt),Ptr(bias),Ptr(y),g,b,c,h,w,o,r,s,p,q,
                                        sh,sw,ph,pw,dh,dw,has_bias,'ieee',tm,tn,tk,rs)
                    np.testing.assert_allclose(y,expected,rtol=1e-11,atol=1e-11)
                    count+=1
    print(f'PASS: {count} kernel indexing cases (NumPy emulation; GPU untested)')

if __name__=='__main__': main()
