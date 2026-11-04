import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def supervised_compare(X, y, random_state=42, stratify=True):
    y = pd.Series(y).reset_index(drop=True)
    X = X.reset_index(drop=True)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.2,random_state=random_state,stratify=y if stratify else None)
    numeric=X.select_dtypes(include='number').columns.tolist()
    categorical=[c for c in X.columns if c not in numeric]
    prep=ColumnTransformer([('num',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),numeric),('cat',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),('encode',OneHotEncoder(handle_unknown='ignore'))]),categorical)])
    results=[]; models={}
    for name,est in [('LogisticRegression',LogisticRegression(max_iter=1500,class_weight='balanced',random_state=random_state)),('RandomForest',RandomForestClassifier(n_estimators=100,min_samples_leaf=3,class_weight='balanced_subsample',random_state=random_state,n_jobs=-1))]:
        pipe=Pipeline([('preprocess',prep),('model',est)])
        pipe.fit(X_train,y_train)
        scores=pipe.predict_proba(X_test)[:,1]
        pred=(scores>=.5).astype(int)
        results.append({'model':name,'roc_auc':roc_auc_score(y_test,scores),'pr_auc':average_precision_score(y_test,scores),'confusion_matrix':confusion_matrix(y_test,pred).tolist(),'classification_report':classification_report(y_test,pred,output_dict=True,zero_division=0)})
        models[name]=pipe
    return results,models,(X_test,y_test)
