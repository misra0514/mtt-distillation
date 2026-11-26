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


@triton.jit
def _conv_kernel(X, Weight, Bias, Y, G:tl.constexpr, B:tl.constexpr,
          C:tl.constexpr,H:tl.constexpr,W:tl.constexpr,O:tl.constexpr,
          R:tl.constexpr,S:tl.constexpr,P:tl.constexpr,Q:tl.constexpr,
          SH:tl.constexpr,SW:tl.constexpr,PH:tl.constexpr,PW:tl.constexpr,
          DH:tl.constexpr,DW:tl.constexpr,BIAS:tl.constexpr,
          PRECISION:tl.constexpr,TM:tl.constexpr,TN:tl.constexpr,
          TK:tl.constexpr,RS_LOOP:tl.constexpr,MERGE_BATCH:tl.constexpr):
    # The batch-tile schedule puts consecutive images of one group in GEMM N.
    pid = tl.program_id(0)
    if MERGE_BATCH:
        g = tl.program_id(1)
        count:tl.constexpr = B*P*Q
    else:
        gb = tl.program_id(1)
        g = gb // B
        count:tl.constexpr = P*Q
    nm, nn = tl.cdiv(O,TM), tl.cdiv(count,TN)
    # Reuse activation tiles across up to four output-channel tiles.
    chunk = pid // (4*nn)
    first = chunk*4
    actual = tl.minimum(nm-first,4)
    mi = first + (pid % (4*nn)) % actual
    ni = (pid % (4*nn)) // actual
    oc = mi*TM + tl.arange(0,TM)
    n = ni*TN + tl.arange(0,TN)
    if MERGE_BATCH:
        image = g*B + n//(P*Q)
        pos = n%(P*Q)
        xp = X + image[None,:]*C*H*W
        yp = Y + image[None,:]*O*P*Q
    else:
        pos = n
        xp = X + gb*C*H*W
        yp = Y + gb*O*P*Q
    ih0, iw0 = pos//Q*SH-PH, pos%Q*SW-PW
    kk = tl.arange(0,TK)
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
                                (ci[:,None]<C)&(n[None,:]<count)&
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
                        (k[:,None]<C*R*S)&(n[None,:]<count)&
                        (ih>=0)&(ih<H)&(iw>=0)&(iw<W),0)
            acc = tl.dot(a,v,acc,input_precision=PRECISION)
    if BIAS:
        acc += tl.load(Bias+g*O+oc,oc<O,0)[:,None]
    tl.store(yp+oc[:,None]*(P*Q)+pos[None,:],acc,
             (oc[:,None]<O)&(n[None,:]<count))


_tune_key = [
    'G', 'B', 'C', 'H', 'W', 'O', 'R', 'S', 'P', 'Q', 'SH', 'SW',
    'PH', 'PW', 'DH', 'DW', 'BIAS', 'PRECISION',
]
# Separate tuners prevent an image-tile result from serving a batch tile.
_conv = triton.autotune(configs=configs(), key=_tune_key)(_conv_kernel)
_conv_batch_tile = triton.autotune(configs=configs(), key=_tune_key)(_conv_kernel)


def pair(v):
    if isinstance(v,int):
        return v,v
    if len(v)!=2 or any(not isinstance(i,int) for i in v):
        raise ValueError('Expected an int or pair of ints')
    return tuple(v)


def grouped_conv2d(x, weight, bias=None, stride=1, padding=0,
                   dilation=1, groups=1, precision='ieee', *,
                   algorithm='image_tile'):
    """X [G*B,C,H,W], W [G,O,C,R,S], bias [G,O] -> [G*B,O,P,Q].

    Leading dimension uses index = group * B + batch. Forward only.
    Use no_grad for manual-backward integrations. batch_tile is opt-in.
    """
    if x.ndim != 4 or weight.ndim != 5:
        raise ValueError('Expected x=[G*B,C,H,W], weight=[G,O,C,R,S]')
    if not isinstance(groups, int) or groups <= 0:
        raise ValueError('groups must be a positive integer')
    n, c, h, w = x.shape
    gw, o, cw, r, s = weight.shape
    g = groups
    if n % g != 0:
        raise ValueError('x.shape[0] must be divisible by groups')
    b = n // g
    if g != gw or c != cw or min(*x.shape, *weight.shape) <= 0:
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
    if algorithm not in ('image_tile', 'batch_tile'):
        raise ValueError('Expected algorithm=image_tile or batch_tile')
    if max(x.numel(),weight.numel(),n*o*p*q)>=2**31:
        raise ValueError('Tensor too large for int32 indexing')

    merge_batch = algorithm == 'batch_tile'
    kernel = _conv_batch_tile if merge_batch else _conv
    positions, slices = (b*p*q, g) if merge_batch else (p*q, n)
    with torch.cuda.device(x.device):
        y = torch.empty((n,o,p,q),device=x.device,dtype=x.dtype)
        grid = lambda meta: (
            triton.cdiv(o,meta['TM'])*triton.cdiv(positions,meta['TN']), slices,
        )
        kernel[grid](x,weight,bias if bias is not None else x,y,
                     g,b,c,h,w,o,r,s,p,q,sh,sw,ph,pw,dh,dw,bias is not None,precision,
                     MERGE_BATCH=merge_batch)
    return y


def conv2d_group_first(x, weight, bias=None, stride=1, padding=0,
                       dilation=1, precision='ieee', *, algorithm='image_tile'):
    """5D view of the same interface, preserving the original wrapper."""
    if x.ndim != 5 or not x.is_contiguous():
        raise ValueError('Expected contiguous x=[G,B,C,H,W]')
    g, b, c, h, w = x.shape
    y = grouped_conv2d(x.view(g*b,c,h,w),weight,bias,stride,padding,dilation,
                       g,precision,algorithm=algorithm)
    return y.view(g,b,*y.shape[1:])
