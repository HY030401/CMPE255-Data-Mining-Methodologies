import sys
from pathlib import Path
import importlib.util
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
def load(relative):
    spec=importlib.util.spec_from_file_location('project',ROOT/relative)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def test_bank_excludes_post_call_duration(tmp_path):
    p=tmp_path/'bank.csv';p.write_text('age;duration;job;y\n30;100;admin.;yes\n40;200;services;no\n')
    X,y=load('01_CRISP_DM/src/run.py').load_bank(p)
    assert 'duration' not in X and list(y)==[1,0]

def test_semma_excludes_refunds():
    mod=load('02_SEMMA/src/run.py')
    df=pd.DataFrame({'CustomerID':[1,1,2],'InvoiceNo':['1','C2','3'],'Quantity':[2,-1,1],'UnitPrice':[10,10,5],'InvoiceDate':['2021-01-01']*3})
    rfm=mod.prepare(df)
    assert rfm.loc[1,'monetary']==20 and rfm.loc[1,'frequency']==1

def test_fraud_target_removed():
    X,y=load('03_KDD/src/run.py').prepare(pd.DataFrame({'V1':[0,1,2],'Class':[0,1,0]}))
    assert 'Class' not in X and y.sum()==1
