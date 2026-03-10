import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
from email.parser import BytesParser
from email import policy
import os

class NaiveBayesBernoulli:
    def NaiveBayesBernoulli(self):
        self.fj
        self.pij

    def get_fj(self, X: ArrayLike, y) -> ArrayLike:
        #since we know that are classes are binary, will simply use parameterized version instead of ECDF
        # mle estimator for Bernoulli p is simply the average
        # returns [y, x_i]
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


        denom = 0
        for j in [0, 1]:
            denom += self.pij[str(j)] * self.fj_x(j, x)

        output = []
        for j in [0, 1]:
            numerator = self.pij[str(j)] * self.fj_x(j, x)
            output.append(numerator/denom)
        return (np.argmax(output), np.max(output))

def create_matrix(training_dir="TRAINING/",features=['you',
 'your',
 'the',
 'that',
 'but',
 'our',
 'and',
 'Our',
 'not',
 'will',
 'price',
 'with',
 'THE',
 'some',
 'use',
 'this',
 'receive',
 'Contact',
 'money',
 'YOUR']
):
    features_set = set(features)
    labels = {}
    with open('SPAMTrain.label', 'r') as f:
        for line in f.readlines():
            splitted = line.split()
            labels[splitted[1]] = splitted[0]
            

    rows = []
    for file_name in os.listdir(training_dir):

        row = {feat: 0 for feat in features}
        row['file_name'] = file_name
        row['label'] = labels[file_name]
        try:
            with open(os.path.join(training_dir, file_name), 'rb') as fp:
                msg = BytesParser(policy=policy.default).parse(fp)

            body = msg.get_body(preferencelist=('plain'))
            if body:
                words = body.get_content().split()
                for word in words:
                    if word in features_set:
                        row[word] = 1
        except Exception as e:
             print(f"Skipping {file_name}: {e}")
        rows.append(row)
    
    df = pd.DataFrame(rows, columns=['file_name'] + features + ['label'])

    return df

