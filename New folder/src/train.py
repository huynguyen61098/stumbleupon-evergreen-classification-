import os
import pandas as pd
import joblib
from sklearn import metrics
from sklearn import model_selection
from sklearn import feature_extraction
from sklearn.decomposition import TruncatedSVD
from sklearn import pipeline
from sklearn import linear_model
import numpy as np
FOLD_MAPPING = {
    0: [1,2,3,4],
    1: [0,2,3,4],
    2: [0,1,3,4],
    3: [0,1,2,4],
    4: [0,1,2,3]
}

FOLD= int(os.environ.get("FOLD"))
TRAINING_DATA= os.environ.get("TRAINING_DATA")
TEST_DATA= os.environ.get("TEST_DATA")
MODEL_OUTPUT = "F:\data learn/stumbleupon/"
MODEL = os.environ.get("MODEL")

    
if __name__ == "__main__":
    
    df = pd.read_csv(TRAINING_DATA)
    
    train_df = df[df.kfold.isin(FOLD_MAPPING.get(FOLD))].dropna(how="any", axis =0)
    valid_df = df[df.kfold == FOLD].dropna(how="any", axis = 0)
    
    ytrain = train_df.label.values
    yvalid = valid_df.label.values
    
    train_df = list(np.array(train_df["boilerplate"]))
    valid_df = list(np.array(valid_df["boilerplate"]))
    
    vectorizer = feature_extraction.text.TfidfVectorizer()
    
        
    print("fitting pipeline")
    vectorizer.fit(train_df)
    print("transforming data")
    train_df = vectorizer.transform(train_df)
    valid_df = vectorizer.transform(valid_df)
    
    
    logistic = linear_model.LogisticRegression() 
    svd = TruncatedSVD()
    clf = pipeline.Pipeline([("svd", svd),("logistic", logistic)])
    param_grid = {"svd__n_components": [100,200],
                   "logistic__penalty": ['l2'],
                    "logistic__C": [0.01, 0.1, 1, 10]
                 }
    
    model = model_selection.RandomizedSearchCV(estimator = clf, param_distributions = param_grid,n_iter = 7,scoring="roc_auc", cv = 5, random_state = 0, n_jobs = -1)
    model.fit(train_df,ytrain)
    best_model = model.best_estimator_
    
    preds = best_model.predict_proba(valid_df)[:,1]
    
    roc_auc_score = metrics.roc_auc_score(yvalid, preds)
    print(f"Fold = {FOLD}, score = {roc_auc_score}")
    
    joblib.dump(vectorizer, f"{MODEL_OUTPUT}model/{MODEL}_{FOLD}_vectorizer.pkl")
    joblib.dump(best_model, f"{MODEL_OUTPUT}model/{MODEL}_{FOLD}.pkl")
    #joblib.dump(train_df.columns, f"{MODEL_OUTPUT}model/{MODEL}_{FOLD}_columns.pkl")
    
    
    
