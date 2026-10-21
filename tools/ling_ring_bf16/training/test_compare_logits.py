
import numpy as np
import pytest
from qualify import compare_logits

def original(a,b):
    d=np.abs(a-b)
    argmax=bool(np.array_equal(a.argmax(-1),b.argmax(-1)))
    tol=bool(np.all(d<=.001+.0001*np.abs(b)))
    return dict(max_abs=float(d.max()),argmax_equal=argmax,tolerance_pass=tol,passed=argmax and tol)

def test_chunked_comparison_matches_decision_across_chunk_boundaries():
    rng=np.random.default_rng(42)
    for rows in (1,7,8,9,23):
        a=rng.normal(size=(rows,51)).astype(np.float32)
        for scale in (0.,.00001,.01):
            b=a+rng.normal(scale=scale,size=a.shape).astype(np.float32)
            expected=original(a,b);got=compare_logits(a,b)
            assert all(got[k]==v for k,v in expected.items())

def test_chunked_comparison_ties_and_tolerance_boundary():
    for eps in (0.,np.float32(.001),np.nextafter(np.float32(.001),np.float32(1.))):
        b=np.array([[0.,0.,2.]],np.float32);a=b.copy();a[0,0]=eps
        expected=original(a,b);got=compare_logits(a,b)
        assert all(got[k]==v for k,v in expected.items())
    assert compare_logits([[1.,1.,0.]],[[1.,1.,0.]])['argmax_equal']

@pytest.mark.parametrize('a,b',[
    ([[np.nan]],[[0.]]),([[np.inf]],[[np.inf]]),
    ([[1.,2.]],[[1.]]),([],[]),(1.,1.)])
def test_chunked_comparison_invalid_input(a,b):
    with pytest.raises(ValueError):compare_logits(a,b)
