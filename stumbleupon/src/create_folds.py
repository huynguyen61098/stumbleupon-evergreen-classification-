import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn import model_selection
import string


def clean_text(s):
    ps = PorterStemmer()
    s = s.split()
    s = " ".join(s)
    s = re.sub(f'[{re.escape(string.punctuation)}]',"",s)
    s = s.lower()
    if not s in set(stopwords.words("english")):
    	return ps.stem(s)

if __name__== "__main__":
    #read the training data
    df = pd.read_csv("F:\data learn/stumbleupon/input/train.tsv",sep = "\t")
    df.boilerplate = df.boilerplate.apply(clean_text)
    #we create a new column called kfold and fill it with -1
    df["kfold"] = -1
    
    #the next step is to randomize the rows of the data
    df = df.sample(frac=1).reset_index(drop=True)
    
    #initiate the kfold class from model_selection module
    kf = model_selection.StratifiedKFold(n_splits = 5)
    
    #fill the new kfold column
    for fold, (train_idx, val_idx) in enumerate(kf.split(X=df, y = df.label.values)):
        print(len(train_idx), len(val_idx))
        df.loc[val_idx,"kfold"] = fold
        
    df.to_csv("F:\data learn/stumbleupon/input/train_folds.csv", index = False)