import json
import numpy as np
import pytest
import torch
from arfm.data import task_zscore,LiberoChunks
from arfm.objective import weighted_loss

def test_task_population_standardization_and_order():
    values=np.array([-.004,-.002,.001,.005])
    z,stats=task_zscore(values)
    assert abs(z.mean())<1e-7
    assert np.isclose(z.std(),values.std()/(values.std()+1e-8))
    assert np.array_equal(np.argsort(z),np.argsort(values))
    assert stats['std']==values.std(ddof=0)

def test_constant_task_is_uniform_and_finite():
    z,stats=task_zscore(np.full(20,.01))
    assert np.isfinite(z).all() and np.allclose(z,0,atol=1e-8)
    _,m=weighted_loss(torch.ones(16),torch.from_numpy(z[:16]))
    assert m['ess']==16.

@pytest.mark.parametrize('values,eps',[([],1e-8),([0,np.nan],1e-8),([0,1],0)])
def test_invalid_statistics(values,eps):
    with pytest.raises(ValueError):task_zscore(values,eps)

def test_dataset_normalizes_each_task_not_combined_and_preserves_files(tmp_path):
    filenames=[]; originals=[]
    for task,values in enumerate([np.array([-2.,0.,2.])*.001,np.array([8.,10.,12.])]):
        p=tmp_path/f'task_{task}.npz'
        np.savez(p,lengths=[3],episodes=['demo_0'],advantage=values.astype('float32'),source=f'task_{task}_demo.hdf5')
        filenames.append(str(p)); originals.append(p.read_bytes())
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps({'files':filenames}))
    raw=LiberoChunks(manifest); scaled=LiberoChunks(manifest,advantage_normalization='task_zscore')
    for i in range(2):
        assert np.array_equal(raw.tasks[i]['advantage'],scaled.tasks[i]['raw_advantage'])
        assert abs(scaled.tasks[i]['advantage'].mean())<1e-6
        assert abs(scaled.tasks[i]['advantage'].std()-1)<1e-5
        assert originals[i]==(tmp_path/f'task_{i}.npz').read_bytes()

def test_fixed_alpha_has_nontrivial_weights_after_zscore():
    z,_=task_zscore(np.linspace(-.005,.005,16))
    _,m=weighted_loss(torch.linspace(.05,.1,16),torch.tensor(z),'rwr',.5)
    assert 8<m['ess']<14
    assert m['weight_max']>1/16 and m['weight_min']<1/16
