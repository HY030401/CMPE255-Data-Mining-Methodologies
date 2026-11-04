from pathlib import Path
import argparse,json
import numpy as np,pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from joblib import dump

def prepare(df):
    required={'CustomerID','InvoiceNo','Quantity','UnitPrice','InvoiceDate'}
    if not required.issubset(df.columns): raise ValueError('Missing columns: '+str(required-set(df.columns)))
    df=df.copy();df['InvoiceDate']=pd.to_datetime(df['InvoiceDate'],errors='coerce')
    df=df.dropna(subset=['CustomerID','InvoiceDate'])
    df=df[(df['Quantity']>0)&(df['UnitPrice']>0)&(~df['InvoiceNo'].astype(str).str.startswith('C'))].copy()
    df['Amount']=df['Quantity']*df['UnitPrice']
    ref=df['InvoiceDate'].max()+pd.Timedelta(days=1)
    rfm=df.groupby('CustomerID').agg(recency=('InvoiceDate',lambda x:(ref-x.max()).days),frequency=('InvoiceNo','nunique'),monetary=('Amount','sum'))
    return rfm

def run(path,out):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    df=pd.read_csv(path,encoding='latin-1');rfm=prepare(df)
    if len(rfm)<10: raise ValueError('Need >=10 customers for segmentation')
    features=np.log1p(rfm[['recency','frequency','monetary']]);scaler=StandardScaler();X=scaler.fit_transform(features)
    candidates=[]
    for k in range(2,min(7,len(rfm)-1)):
        km=KMeans(n_clusters=k,random_state=42,n_init=10);labels=km.fit_predict(X)
        score=silhouette_score(X,labels,sample_size=min(2000,len(X)),random_state=42)
        candidates.append((k,float(score)))
    k=max(candidates,key=lambda v:v[1])[0];model=KMeans(n_clusters=k,random_state=42,n_init=10).fit(X)
    rfm['cluster']=model.labels_;rfm.to_csv(out/'customer_segments.csv')
    (out/'metrics.json').write_text(json.dumps({'customers':len(rfm),'silhouette_by_k':candidates,'selected_k':k,'cluster_profiles':rfm.groupby('cluster')[['recency','frequency','monetary']].median().round(2).to_dict(orient='index')},indent=2))
    dump({'scaler':scaler,'model':model},out/'model.joblib');print('Saved segmentation artifacts')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--out',default='02_SEMMA/artifacts');a=p.parse_args();run(a.data,a.out)
