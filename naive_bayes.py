import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
import email

class NaiveBayesBernoulli:
    def NaiveBayesBernoulli(self):
        self.fj
        self.pij

    def get_fj(self, X: ArrayLike, y) -> ArrayLike:
        #since we know that are classes are binary, will simply use parameterized version instead of ECDF
        # mle estimator for Bernoulli p is simply the average
        df = pd.DataFrame(X)
        average_given_label = df.groupby(y).mean()
        return average_given_label.to_numpy()

    def get_pij(self, y) -> dict:
        probabilites = pd.Series(y).value_counts(normalize=True).to_dict()
        return probabilites

    def bernoulli_pdf(self, p, x):
        return p if x == 1 else 1 - p

    def fj_x(self, j, x: list):
        retval = 1
        for i, p in enumerate(self.fj[j]):
            retval *= self.bernoulli_pdf(p, x[i])
        return retval

    def fit(self, X: ArrayLike, y):
        self.fj = self.get_fj(X, y)
        self.pij = self.get_pij(y)


    def predict(self, x):
        x = np.array(x)
        numerator = self.pij[1] * self.fj_x(1, x)

        denom = 0
        for j in [0, 1]:
            denom += self.pij[j] * self.fj_x(j, x)
        
        return numerator/denom
