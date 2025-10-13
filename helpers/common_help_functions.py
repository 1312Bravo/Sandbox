# Libraries
import numpy as np
import pandas as pd

# -----------------------------------------------------------
# Calculate log-likelihood
# -----------------------------------------------------------
def log_likelihood(y_true, y_pred):
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    ll = np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return ll

# -----------------------------------------------------------
# Bucket feature to percentiles
# -----------------------------------------------------------
def percentile_bin_feature(data, bin_feature, percentiles):

    df = data.copy()
    my_percentiles = percentiles
    my_percentile_values = np.percentile(df[bin_feature], my_percentiles)
    df[bin_feature+"_bins_percentiles"] = pd.cut(df[bin_feature], bins=my_percentile_values, include_lowest=True, duplicates="drop")
    df[bin_feature+"_bins_percentiles"] = df[bin_feature+"_bins_percentiles"].map({cat: i for i, cat in enumerate(df[bin_feature+"_bins_percentiles"].cat.categories)})
    df[bin_feature+"_bins_percentiles_labels"] = df.groupby(bin_feature+"_bins_percentiles", observed=True)[bin_feature].transform(lambda x: f"[{round(x.min(), 3)}, {round(x.max(), 3)}]")
    df[bin_feature + "_bins_percentiles_labels"] = pd.Categorical(df[bin_feature + "_bins_percentiles_labels"], categories=df.sort_values(bin_feature+"_bins_percentiles", ascending=True)[bin_feature + "_bins_percentiles_labels"].unique(), ordered=True)

    return df[[bin_feature, bin_feature+"_bins_percentiles", bin_feature+"_bins_percentiles_labels"]]

# -----------------------------------------------------------
# Transfrom [0,1] to logit space and logit back to [0,1]
# -----------------------------------------------------------
def zeroOne_to_logit(value):
    epsilon = 1e-3 
    value = np.clip(value, epsilon, 1 - epsilon)  
    return np.log(value / (1-value))

def logit_to_zeroOne(value):
    return 1 / (1 + np.exp(-value))
