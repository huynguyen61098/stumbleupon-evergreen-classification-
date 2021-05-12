import os
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
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
    df = pd.read_csv("F:\data learn/stumbleupon/input/test.tsv",sep = "\t")
    df.boilerplate = df.boilerplate.apply(clean_text)
    df = df[["urlid","boilerplate"]]
    
    df.to_csv("F:\data learn/stumbleupon/input/modified_test.csv", index = False)