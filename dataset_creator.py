"""
Sample Dataset Creator
======================
Generates a synthetic dataset with 20 binary features and a binary label.

Each feature is independently Bernoulli-distributed, with its own probability
conditioned on the label:

    P(feature_i = 1 | label = 0) = prob_matrix[i, 0]
    P(feature_i = 1 | label = 1) = prob_matrix[i, 1]

Labels are split 50/50 between class 0 and class 1.
"""

import numpy as np
import pandas as pd
from typing import Optional


N_FEATURES = 20


def make_dataset(
    n_samples: int = 1000,
    prob_matrix: Optional[np.ndarray] = None,
    random_state: Optional[int] = None,
    feature_prefix: str = "feature_",
) -> pd.DataFrame:
    """
    Generate a synthetic binary-feature dataset.

    Parameters
    ----------
    n_samples : int
        Total number of samples to generate. Must be even (split 50/50 by label).
        If odd, it is rounded down to the nearest even number.
        Default: 1000.

    prob_matrix : np.ndarray of shape (20, 2), optional
        Entry [i, 0] = P(feature_i = 1 | label = 0)
        Entry [i, 1] = P(feature_i = 1 | label = 1)
        All values must be in [0.0, 1.0].
        If None, a random matrix is generated using a seeded RNG so the
        defaults are reproducible when random_state is set.

    random_state : int or None, optional
        Seed for NumPy's random number generator. Use for reproducibility.
        Default: None (non-deterministic).

    feature_prefix : str
        Column name prefix for the 20 feature columns.
        Default: "feature_".

    Returns
    -------
    pd.DataFrame
        DataFrame with columns  [feature_0, ..., feature_19, label].
        All feature columns and the label column contain integer values (0 or 1).
        Rows are shuffled so the two classes are interleaved.

    Raises
    ------
    ValueError
        If prob_matrix has the wrong shape or contains values outside [0, 1].

    Examples
    --------
    # --- Quick start (random defaults) ---
    df = make_dataset(n_samples=500, random_state=42)

    # --- Custom probability matrix ---
    import numpy as np
    probs = np.full((20, 2), 0.5)          # baseline: 50% everywhere
    probs[:5, 1] = 0.9                     # first 5 features strongly predict label=1
    probs[:5, 0] = 0.1                     # and are rare when label=0
    df = make_dataset(n_samples=1000, prob_matrix=probs, random_state=0)
    print(df.head())
    print(df["label"].value_counts())
    """

    rng = np.random.default_rng(random_state)

    # ------------------------------------------------------------------ #
    #  Probability matrix                                                  #
    # ------------------------------------------------------------------ #
    if prob_matrix is None:
        # Sensible random defaults: draw from a Beta(2,2) distribution so
        # probabilities cluster around 0.5 but vary meaningfully.
        prob_matrix = rng.beta(a=2, b=2, size=(N_FEATURES, 2))
    else:
        prob_matrix = np.asarray(prob_matrix, dtype=float)
        if prob_matrix.shape != (N_FEATURES, 2):
            raise ValueError(
                f"prob_matrix must have shape ({N_FEATURES}, 2), "
                f"got {prob_matrix.shape}."
            )
        if not (np.all(prob_matrix >= 0.0) and np.all(prob_matrix <= 1.0)):
            raise ValueError(
                "All values in prob_matrix must be in the range [0.0, 1.0]."
            )

    # ------------------------------------------------------------------ #
    #  Class sizes (strict 50/50 split)                                   #
    # ------------------------------------------------------------------ #
    n_samples = int(n_samples)
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2.")
    # Round down to nearest even number so split is exact
    n_samples = (n_samples // 2) * 2
    n_per_class = n_samples // 2

    # ------------------------------------------------------------------ #
    #  Generate features                                                   #
    # ------------------------------------------------------------------ #
    feature_names = [f"{feature_prefix}{i}" for i in range(N_FEATURES)]

    # Class 0 block  — shape (n_per_class, 20)
    probs_class0 = prob_matrix[:, 0]          # (20,)
    block0 = rng.random(size=(n_per_class, N_FEATURES)) < probs_class0
    labels0 = np.zeros(n_per_class, dtype=int)

    # Class 1 block  — shape (n_per_class, 20)
    probs_class1 = prob_matrix[:, 1]          # (20,)
    block1 = rng.random(size=(n_per_class, N_FEATURES)) < probs_class1
    labels1 = np.ones(n_per_class, dtype=int)

    # ------------------------------------------------------------------ #
    #  Assemble DataFrame and shuffle                                      #
    # ------------------------------------------------------------------ #
    features = np.vstack([block0, block1]).astype(int)
    labels   = np.concatenate([labels0, labels1])

    df = pd.DataFrame(features, columns=feature_names)
    df["label"] = labels

    # Shuffle rows
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return df


# ------------------------------------------------------------------ #
#  Quick demo when run directly                                        #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import numpy as np
    from naive_bayes import NaiveBayesBernoulli

  
    df1 = make_dataset(n_samples=10000)
    train=df1.iloc[:8000]
    test = df1.iloc[8000:]
    nbayes = NaiveBayesBernoulli()
    nbayes.fit(X=train.iloc[:,:-1], y=train.iloc[:,-1])
   

    meow = []
    test = test.reset_index()
    for index, x in test.iterrows():
        sample = np.array(x)
        sample = sample[1:21]
        print(f'Guess: {nbayes.predict(sample):.2f}, True: {x.iloc[-1]}')
        meow.append(int(np.round(nbayes.predict(sample))) == x.iloc[-1])

    print(np.mean(meow))
