"""Target-GPU smoke test for the v34 unified auto-selector interface."""
import torch

import test_script.stateless_conv_double_bwd.conv_double_bwd as kernel
from test_script.stateless_conv_double_bwd.benchmark import aten_group_first,check,make_args


_BALANCED_BG=((1,1),(1,4),(1,16),(1,64),(10,1),(10,4),(10,16),(10,64),
              (100,1),(100,2),(100,4),(1000,1),(1000,2),(1000,4))


def _static_signature(g,b,c,o,h,p,r,s,stride,padding,bias=False):
    """Construct a selector-only signature without allocating CUDA tensors."""
    return kernel.ConvDoubleBackwardSignature(
        device=('NVIDIA RTX A5000',8,6),dtype='torch.float32',
        runtime=(str(torch.__version__),str(kernel.triton.__version__)),
        shape=(g,b,c,h,h,o,r,s,p,p),
        convolution=(stride,stride,padding,padding,1,1),
        optional_inputs=(True,True,bias),compute=(True,True,True),
        precision='ieee',workspace_bytes=128*1024**2,split_k=None)


def validate_v34_selector_matrix():
    """Check all 196 planned v33 points against the accepted v34 table."""
    pointwise=((64,128,32,16),(128,256,16,8),(256,512,8,4),
               (64,128,16,8),(128,256,8,4),(256,512,4,2))
    for c,o,h,p in pointwise:
        for b,g in _BALANCED_BG:
            sig=_static_signature(g,b,c,o,h,p,1,1,2,0)
            assert kernel._select_algorithm_cached(sig)==(
                'pointwise_stride2','measured_pointwise_v26')

    # Rows follow CIFAR s2/s3/s4 then TinyImageNet s2/s3/s4.  Each row is
    # ordered by _BALANCED_BG and includes the seven sparse exact overrides.
    dense_shapes=((64,128,32,16),(128,256,16,8),(256,512,8,4),
                  (64,128,16,8),(128,256,8,4),(256,512,4,2))
    dense_expected=(
        ('implicit','stride2_packed_gi_tail','stride2_joint_onebuf',
         'stride2_joint_onebuf_wide_gi','stride2_wide_chunks',
         'stride2_joint_onebuf','stride2_joint_onebuf','stride2_joint_onebuf',
         'stride2_joint_split_gw8','stride2_joint_split_gw8',
         'stride2_joint_split_gw8','stride2_joint_split_gw8',
         'stride2_joint_split_gw8','stride2_joint_split_gw8'),
        ('stride2_static_weight','stride2_static_weight','stride2_packed_gi_tail',
         'stride2_joint_onebuf_wide_gi','stride2_ggo_single',
         'stride2_joint_onebuf_wide_gi','stride2_joint_onebuf',
         'stride2_joint_onebuf_wide_gi','stride2_joint_onebuf_wide_gi',
         'stride2_joint_onebuf','stride2_joint_onebuf','stride2_joint_split_gw8',
         'stride2_joint_split_gw8','stride2_joint_split_gw8'),
        ('stride2_static_weight','stride2_packed_gi_tail','implicit','implicit',
         'stride2_packed_gi_tail','stride2_joint_onebuf',
         'stride2_joint_onebuf_wide_gi','stride2_joint_onebuf_wide_gi',
         'stride2_joint_onebuf','stride2_joint_onebuf','stride2_joint_onebuf',
         'stride2_joint_onebuf','stride2_joint_onebuf','stride2_joint_onebuf'),
        ('implicit','implicit','stride2_static_weight','stride2_packed_gi_tail',
         'stride2_static_weight','stride2_ggo_single','stride2_joint_onebuf',
         'stride2_joint_onebuf','stride2_wide_chunks','stride2_wide_chunks',
         'stride2_ggo_single','stride2_joint_split_gw8',
         'stride2_joint_split_gw8','stride2_joint_split_gw8'),
        ('stride2_static_weight','stride2_static_weight','stride2_packed_gi_tail',
         'implicit','stride2_static_weight','stride2_packed_gi_tail',
         'stride2_joint_onebuf_wide_gi','stride2_joint_onebuf_wide_gi',
         'stride2_joint_onebuf_wide_gi','stride2_joint_onebuf',
         'stride2_joint_onebuf','stride2_packed_gi_tail','stride2_ggo_single',
         'stride2_joint_onebuf'),
        ('stride2_static_weight','implicit','implicit','implicit',
         'stride2_static_weight','stride2_packed_gi_tail',
         'stride2_joint_onebuf_wide_gi','stride2_joint_onebuf_wide_gi',
         'stride2_joint_onebuf','stride2_joint_onebuf','stride2_joint_onebuf',
         'stride2_joint_onebuf','stride2_joint_onebuf','stride2_joint_onebuf'))
    exact_count=0
    for (c,o,h,p),expected_row in zip(dense_shapes,dense_expected):
        for (b,g),expected in zip(_BALANCED_BG,expected_row):
            sig=_static_signature(g,b,c,o,h,p,3,3,2,1)
            selected,source=kernel._select_algorithm_cached(sig)
            assert selected==expected,(sig.shape,selected,source,expected)
            exact_count+=source=='builtin_exact_table'
    assert exact_count==7,exact_count

    stems=((3,64,64,32),(3,64,128,64))
    for c,o,h,p in stems:
        for b,g in _BALANCED_BG:
            total=g*b*p*p
            expected=('implicit' if total<=1024 else
                      'stem7_gi' if b==1 and total<=4096 else
                      'stem7_gi_ggo')
            sig=_static_signature(g,b,c,o,h,p,7,7,2,3)
            selected,source=kernel._select_algorithm_cached(sig)
            assert (selected,source)==(expected,'stem7_family_v34')
    print('v34 selector-matrix PASS: 196 planned points, 7 exact overrides')


def validate_group_first(shape,stride=(1,1),padding=(1,1),bias=False,expected=None):
    args=make_args(shape,torch.float32,'cuda',st=stride,pad=padding,
                   opts=(True,True,bias))
    kwargs=dict(precision='ieee',workspace_limit_mb=128,algorithm='auto')
    signature=kernel.convolution_double_backward_signature(
        *args,precision='ieee',workspace_limit_mb=128)
    algorithm,source=kernel._select_algorithm_cached(signature)
    if expected is not None:assert algorithm==expected,(algorithm,source,expected)
    auto=kernel.convolution_double_backward(*args,**kwargs)
    explicit=kernel.convolution_double_backward(
        *args,**dict(kwargs,algorithm=algorithm))
    ref=aten_group_first(args)
    check(auto,ref,torch.float32,'ieee',args=args)
    check(explicit,ref,torch.float32,'ieee',args=args)
    for a,b in zip(auto,explicit):
        if a is not None:torch.testing.assert_close(a,b,rtol=0,atol=0)
    print('5D PASS',shape,'algorithm='+algorithm,'source='+source)
    return args


def validate_aten_4d(group_args):
    args=tuple(None if t is None else t[0].contiguous() for t in group_args[:6])+group_args[6:]
    kwargs=dict(precision='ieee',workspace_limit_mb=128,algorithm='auto')
    got=kernel.convolution_double_backward(*args,**kwargs)
    ref=torch.ops.aten._convolution_double_backward(*args)
    check(got,ref,torch.float32,'ieee')
    assert all(x is None or x.ndim in (1,4) for x in got)
    signature=kernel.convolution_double_backward_signature(
        *args,precision='ieee',workspace_limit_mb=128)
    algorithm,source=kernel._select_algorithm_cached(signature)
    print('4D PASS','algorithm='+algorithm,'source='+source,
          'output_shapes='+str([None if x is None else tuple(x.shape) for x in got]))


def validate_exact_table(group_args):
    kwargs=dict(precision='ieee',workspace_limit_mb=128,algorithm='auto')
    signature=kernel.convolution_double_backward_signature(
        *group_args,precision='ieee',workspace_limit_mb=128)
    fallback,_=kernel._select_algorithm_cached(signature)
    kernel.register_convolution_double_backward_algorithms(
        {signature:fallback})
    exact,source=kernel._select_algorithm_cached(signature)
    assert exact==fallback and source=='exact_table'
    kernel.register_convolution_double_backward_algorithms({},replace=True)
    print('exact-table PASS')


def validate_call_exact_table(group_args):
    signature=kernel.convolution_double_backward_signature(
        *group_args,precision='ieee',workspace_limit_mb=128)
    table={signature:'implicit'}
    got=kernel.convolution_double_backward(
        *group_args,precision='ieee',workspace_limit_mb=128,
        algorithm='auto',exact_algorithm_table=table)
    ref=aten_group_first(group_args)
    check(got,ref,torch.float32,'ieee',args=group_args)
    print('per-call exact-table PASS')


def validate_fallback_exceptions():
    cases=[
        ((1,10,128,128,16,16,3,3),(1,1),False,'winograd_exact'),
        ((1,10,128,128,16,16,3,3),(1,1),True,'winograd_exact'),
        ((4,10,128,128,8,8,3,3),(1,1),True,'winograd_weight'),
        ((4,1,64,64,32,32,3,3),(1,1),False,'implicit'),
        ((1,1,512,512,4,4,3,3),(1,1),False,'winograd_weight'),
        ((16,1,64,128,32,32,3,3),(2,2),False,'stride2_joint_onebuf'),
        ((64,1,64,128,32,32,3,3),(2,2),False,'stride2_joint_onebuf_wide_gi'),
        ((64,1,128,256,16,16,3,3),(2,2),False,'stride2_joint_onebuf_wide_gi'),
        ((1,10,64,128,32,32,3,3),(2,2),False,'stride2_wide_chunks'),
        ((4,10,64,128,32,32,3,3),(2,2),False,'stride2_joint_onebuf'),
        ((4,10,128,256,16,16,3,3),(2,2),False,'stride2_joint_onebuf_wide_gi'),
        ((4,10,256,512,8,8,3,3),(2,2),False,'stride2_joint_onebuf'),
    ]
    for shape,stride,bias,expected in cases:
        args=make_args(shape,torch.float32,'cuda',st=stride,pad=(1,1),
                       opts=(True,True,bias))
        signature=kernel.convolution_double_backward_signature(
            *args,precision='ieee',workspace_limit_mb=128)
        selected,source=kernel._select_algorithm_cached(signature)
        assert (selected,source)==(expected,'builtin_exact_table'),(selected,source)
        del args
    print('fallback-exception PASS:',len(cases),'measured signatures')


def main():
    validate_v34_selector_matrix()
    if not torch.cuda.is_available():
        print('CUDA unavailable: selector-only tests passed; GPU execution skipped')
        return
    torch.manual_seed(20261118)
    dense=validate_group_first((2,10,32,64,16,16,3,3))
    validate_group_first((2,10,64,128,16,16,3,3),stride=(2,2))
    validate_group_first((2,10,128,256,16,16,1,1),stride=(2,2),
                         padding=(0,0),expected='pointwise_stride2')
    validate_aten_4d(dense)
    validate_exact_table(dense)
    validate_call_exact_table(dense)
    validate_fallback_exceptions()
    print('ALL AUTO-SELECTOR TESTS PASSED')


if __name__=='__main__':
    with torch.no_grad():main()
