import os
import joblib
import numpy as np
import pandas as pd



TEST_DATA= "F:\data learn/stumbleupon/input/modified_test.csv"
MODEL=os.environ.get("MODEL")
MODEL_PATH = "F:\data learn/stumbleupon/model/"
OUTPUT_PATH = "F:\data learn/stumbleupon/"



def predict():
    df = pd.read_csv(TEST_DATA)
    predictions = None

    for FOLD in range(5):
        df = pd.read_csv(TEST_DATA)
        df = df.dropna(how="any", axis =0)

        test_idx = df["urlid"].values

        df = list(np.array(df.boilerplate))

        vectorizer = joblib.load(os.path.join(MODEL_PATH, f"{MODEL}_{FOLD}_vectorizer.pkl"))
        df = vectorizer.transform(df)
        
        #data is ready to train
        clf = joblib.load(os.path.join(MODEL_PATH, f"{MODEL}_{FOLD}.pkl"))
        preds = clf.predict_proba(df)[:, 1]
        
        if FOLD == 0:
            predictions = preds
        else:
            predictions += preds
    predictions /= 5
    
    sub = pd.DataFrame(np.column_stack((test_idx, predictions)), columns = ["urlid","label"])
    sub.urlid = sub.urlid.astype("int32")
    return sub

if __name__ == "__main__":
    submission = predict()
    submission.to_csv(f"{OUTPUT_PATH}submission.csv", index = False)