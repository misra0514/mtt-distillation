"""Forward-only, contiguous group-first convolution. No silent autograd fallback."""
import torch
import triton
import triton.language as tl


def configs():
    # Search both CRS-flattened and spatial-loop/channel-blocked iterators.
    return [triton.Config(dict(TM=m, TN=n, TK=k, RS_LOOP=rs),
                          num_warps=warps, num_stages=stages)
            for rs in (False, True)
            for m,n,k,warps,stages in (
                (16,64,32,4,2), (32,64,32,4,3),
                (32,128,32,4,3), (64,64,32,4,3),
                (32,64,64,4,3), (64,128,32,8,3))]


@triton.autotune(configs=configs(), key=[
    'G','B','C','H','W','O','R','S','P','Q','SH','SW','PH','PW','DH','DW',
    'BIAS','PRECISION'])
@triton.jit
def _conv(X, Weight, Bias, Y, G:tl.constexpr, B:tl.constexpr,
          C:tl.constexpr,H:tl.constexpr,W:tl.constexpr,O:tl.constexpr,
          R:tl.constexpr,S:tl.constexpr,P:tl.constexpr,Q:tl.constexpr,
          SH:tl.constexpr,SW:tl.constexpr,PH:tl.constexpr,PW:tl.constexpr,
          DH:tl.constexpr,DW:tl.constexpr,BIAS:tl.constexpr,
          PRECISION:tl.constexpr,TM:tl.constexpr,TN:tl.constexpr,
          TK:tl.constexpr,RS_LOOP:tl.constexpr):
    # Each program owns one [output-channel, output-spatial] tile.
    # Batch stays outside GEMM to preserve contiguous NCHW spatial stores.
    pid = tl.program_id(0)
    gb = tl.program_id(1)
    g, b = gb // B, gb % B
    nm, nn = tl.cdiv(O,TM), tl.cdiv(P*Q,TN)
    # Reuse activation tiles across up to four output-channel tiles.
    chunk = pid // (4*nn)
    first = chunk*4
    actual = tl.minimum(nm-first,4)
    mi = first + (pid % (4*nn)) % actual
    ni = (pid % (4*nn)) // actual
    oc = mi*TM + tl.arange(0,TM)
    n = ni*TN + tl.arange(0,TN)
    ih0, iw0 = n//Q*SH-PH, n%Q*SW-PW
    kk = tl.arange(0,TK)
    xp = X + (g*B+b)*C*H*W
    wp = Weight + g*O*C*R*S
    acc = tl.zeros((TM,TN),tl.float32)
    if RS_LOOP:
        # A fixed filter position makes adjacent spatial loads contiguous.
        # Useful for large C; for C=3 the flattened variant wastes less work.
        for cb in range(tl.cdiv(C,TK)):
            ci = cb*TK+kk
            for r in tl.static_range(R):
                for s in tl.static_range(S):
                    ih, iw = ih0+r*DH, iw0+s*DW
                    a = tl.load(wp+oc[:,None]*(C*R*S)+ci[None,:]*(R*S)+r*S+s,
                                (oc[:,None]<O)&(ci[None,:]<C),0)
                    v = tl.load(xp+ci[:,None]*H*W+ih[None,:]*W+iw[None,:],
                                (ci[:,None]<C)&(n[None,:]<P*Q)&
                                (ih[None,:]>=0)&(ih[None,:]<H)&
                                (iw[None,:]>=0)&(iw[None,:]<W),0)
                    acc = tl.dot(a,v,acc,input_precision=PRECISION)
    else:
        for kb in range(tl.cdiv(C*R*S,TK)):
            k = kb*TK+kk
            ci, r, s = k//(R*S), (k//S)%R, k%S
            ih = ih0[None,:]+r[:,None]*DH
            iw = iw0[None,:]+s[:,None]*DW
            a = tl.load(wp+oc[:,None]*(C*R*S)+k[None,:],
                        (oc[:,None]<O)&(k[None,:]<C*R*S),0)
            v = tl.load(xp+ci[:,None]*H*W+ih*W+iw,
                        (k[:,None]<C*R*S)&(n[None,:]<P*Q)&
                        (ih>=0)&(ih<H)&(iw>=0)&(iw<W),0)
            acc = tl.dot(a,v,acc,input_precision=PRECISION)
    if BIAS:
        acc += tl.load(Bias+g*O+oc,oc<O,0)[:,None]
    tl.store(Y+((g*B+b)*O+oc[:,None])*(P*Q)+n[None,:],acc,
             (oc[:,None]<O)&(n[None,:]<P*Q))


def pair(v):
    if isinstance(v,int):
        return v,v
    if len(v)!=2 or any(not isinstance(i,int) for i in v):
        raise ValueError('Expected an int or pair of ints')
    return tuple(v)


def conv2d_group_first(x, weight, bias=None, stride=1, padding=0,
                       dilation=1, precision='ieee'):
    """X [G,B,C,H,W], W [G,O,C,R,S], bias [G,O] -> [G,B,O,P,Q].

    Forward only. Under grad mode, requires_grad inputs raise instead of
    silently detaching. Use no_grad for manual-backward integrations.
    precision: ieee (FP32 strict inputs), tf32, or tf32x3 (explicit opt-in).
    """
    if x.ndim!=5 or weight.ndim!=5:
        raise ValueError('Expected 5D x and weight')
    g,b,c,h,w=x.shape
    gw,o,cw,r,s=weight.shape
    if g!=gw or c!=cw or min(*x.shape,*weight.shape)<=0:
        raise ValueError('Invalid shapes or mismatched group/channel dimensions')
    sh,sw=pair(stride); ph,pw=pair(padding); dh,dw=pair(dilation)
    if min(sh,sw,dh,dw)<=0 or min(ph,pw)<0:
        raise ValueError('Invalid convolution parameters')
    p=(h+2*ph-dh*(r-1)-1)//sh+1
    q=(w+2*pw-dw*(s-1)-1)//sw+1
    if min(p,q)<=0 or g*b>65535:
        raise ValueError('Invalid output size or G*B exceeds CUDA grid.y limit')
    tensors=[x,weight]+([] if bias is None else [bias])
    if bias is not None and tuple(bias.shape)!=(g,o):
        raise ValueError('Bias must have shape [G,O]')
    if x.dtype not in (torch.float16,torch.bfloat16,torch.float32):
        raise TypeError('Supported dtypes: float16, bfloat16, float32')
    for t in tensors:
        if not t.is_cuda or t.device!=x.device or t.dtype!=x.dtype or not t.is_contiguous():
            raise ValueError('All tensors must be contiguous CUDA tensors on same device/dtype')
        if torch.is_grad_enabled() and t.requires_grad:
            raise RuntimeError('Forward-only kernel: autograd is not implemented')
    if precision not in ('ieee','tf32','tf32x3'):
        raise ValueError('Invalid precision')
    # Offsets in this kernel are int32. Reject oversized storage explicitly.
    if max(x.numel(),weight.numel(),g*b*o*p*q)>=2**31:
        raise ValueError('Tensor too large for int32 indexing')
    with torch.cuda.device(x.device):
        y=torch.empty((g,b,o,p,q),device=x.device,dtype=x.dtype)
        grid=lambda meta:(triton.cdiv(o,meta['TM'])*triton.cdiv(p*q,meta['TN']),g*b)
        _conv[grid](x,weight,bias if bias is not None else x,y,
                    g,b,c,h,w,o,r,s,p,q,sh,sw,ph,pw,dh,dw,bias is not None,precision)
    return y
