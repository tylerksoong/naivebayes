import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
import email


def get_fj(X: ArrayLike, label_idx) -> ArrayLike:
    #since we know that are classes are binary, will simply use parameterized version instead of ECDF
    # mle estimator for Bernoulli p is simply the average
    df = pd.DataFrame(X)
    label_name = df.columns[label_idx]
    average_given_label = df.groupby(label_name).mean()
    return average_given_label.to_numpy()




def get_pij(X: ArrayLike, label_idx) -> dict:
    df = pd.DataFrame(X)
    probabilites = df.iloc[:,label_idx].value_counts(normalize=True).to_dict()
    return probabilites

def bernoulli_pdf(p, x):
    return p if x == 1 else 1 - p

def get_probofx(fj, x: list, j):
    nparr = np.array(x)
    retval = 1
    for i, p in enumerate(fj[j]):
        retval *= bernoulli_pdf(p, x[i])
    return retval


y
