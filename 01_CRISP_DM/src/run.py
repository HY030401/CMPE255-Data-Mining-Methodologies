from pathlib import Path
import argparse, json, sys
import pandas as pd
from joblib import dump
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from shared.ml_utils import supervised_compare

def load_bank(path):
    df=pd.read_csv(path,sep=';' if str(path).endswith('.csv') and ';' in Path(path).read_text(encoding='utf-8-sig').splitlines()[0] else ',')
    if 'y' not in df.columns: raise ValueError('Bank dataset must contain y column')
    df=df.drop_duplicates()
    y=df.pop('y').map({'yes':1,'no':0})
    if y.isna().any(): raise ValueError('Unexpected target values')
    # 'duration' is known only after the call; exclude for pre-call targeting to prevent leakage.
    X=df.drop(columns=['duration'],errors='ignore')
    return X,y

def run(path,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    X,y=load_bank(path)
    results,models,_=supervised_compare(X,y)
    (out/'metrics.json').write_text(json.dumps({'rows':len(X),'features':X.columns.tolist(),'positive_rate':float(y.mean()),'models':results},indent=2))
    best=max(results,key=lambda x:x['roc_auc'])['model'];dump(models[best],out/'model.joblib')
    print('Saved',out/'metrics.json','best:',best)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',default='01_CRISP_DM/artifacts');a=p.parse_args();run(a.data,a.out)
