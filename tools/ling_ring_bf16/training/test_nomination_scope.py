"""CPU controls for scoped nomination."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import torch,pytest
from capability_runner import nomination
def rows(family,expert):
 return [dict(family=family,group=family,grade={'reward':i%2},features=torch.full((2,32),float(expert))) for i in range(4)]
def test_one_family_nomination():
 chosen,meta=nomination(rows('arithmetic',7),8,('arithmetic',))
 assert chosen==7 and meta['active_families']==['arithmetic']
def test_strict_nomination_requires_both():
 with pytest.raises(ValueError):nomination(rows('arithmetic',7),8)
def test_control_family_does_not_change_active_nomination():
 chosen,_=nomination(rows('arithmetic',7)+rows('extraction',3),8,('arithmetic',))
 assert chosen==7
@pytest.mark.parametrize('scope',[(),('other',),('arithmetic','arithmetic')])
def test_bad_scope(scope):
 with pytest.raises(ValueError):nomination(rows('arithmetic',7),8,scope)
