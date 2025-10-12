import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from typing import Dict
from IPython.display import display

import statsmodels.api as sm
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.calibration import calibration_curve
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import chi2

from helpers.common_help_functions import log_likelihood
from helpers.common_help_functions import percentile_bin_feature

# -----------------------------------------------------------
# Univariate Logistic regression [Logit] on Numeric feature
# -----------------------------------------------------------
def univariate_logit_numeric(
        target: str, target_name: str, 
        feature: str, feature_name: str, 
        data: pd.DataFrame, 
        alpha: float=1.0,
        print_metrics: bool=True,
        show_plot: bool=True
        ) -> pd.DataFrame:
    
    if (print_metrics or show_plot) == True:
        print("-----------------------------------------------------")
        print(f"{target_name} ~ {feature_name} ~> Univariate Logistic Regression")
        print("-----------------------------------------------------\n")

    # Prepare data for modeling
    data = data.copy()

    data_clean = data[[target, feature]].replace([np.inf, -np.inf], np.nan).dropna()
    X = sm.add_constant(data_clean[[feature]])
    y = data_clean[target]

    # Model
    model = sm.Logit(y, X)
    fit = model.fit(disp=False)
    y_pred_prob = fit.predict(X)

    # Metrics
    scaling_factor = np.std(data_clean[feature]) / np.std(data_clean[target])
    metrics = {
        "coef": fit.params[feature],
        "scaled coef": fit.params[feature] * scaling_factor,
        "coef OR": np.exp(fit.params[feature]),
        "scaled coef OR": np.exp(fit.params[feature] * scaling_factor),
        "p-value": fit.pvalues[feature],
        "standard error": fit.bse[feature],
        "scaled standard error": fit.bse[feature] * scaling_factor,
        "confidence interval": round(fit.conf_int().loc[feature], 4).tolist(),
        "scaled confidence interval": round(fit.conf_int().loc[feature] * scaling_factor, 4).tolist(),
        "confidence interval OR": round(np.exp(fit.conf_int().loc[feature]), 4).tolist(),
        "scaled confidence interval OR": round(np.exp(fit.conf_int().loc[feature] * scaling_factor), 4).tolist(),
        "Pseudo R2": fit.prsquared,
        "Log-Likelihood": fit.llf,
        "Likelihood Ratio (Chi2)": fit.llr,
        "Chi2 test p-value": fit.llr_pvalue ,
        "AIC": fit.aic,
        "BIC": fit.bic,
        "AUC": roc_auc_score(y, y_pred_prob),
    }

    metrics_df = pd.DataFrame([metrics])
    metrics_df.insert(0, "feature", feature)

    if print_metrics:
        for key, value in metrics.items():
            if isinstance(value, list):
                print(f"{key} ~> {[f'{v:.4f}' for v in value]}")
            else:
                print(f"{key} ~> {value:.4f}" if isinstance(value, float) else f"{key} ~> {value}")

    # Plot
    if show_plot:
        fig, ax = plt.subplots(2, 3, figsize=(25, 10))

        # Feature distribution
        ax[0,0].set_title(f"{feature_name} Distribution", fontsize=14)
        ax[0,0].hist(data_clean[feature], bins=30, density=True, color="grey", edgecolor="black", alpha=1.0)
        sns.kdeplot(data_clean[feature], ax=ax[0,0], color="black", linewidth=2)
        ax[0,0].set_xlabel(feature_name)

        # Binned feature & Average target values
        binned_feature_df = percentile_bin_feature(data = data_clean, bin_feature = feature, percentiles = np.arange(0,105,5))
        binned_feature_df[target] = data_clean[target]
        binned_feature_agg_df = binned_feature_df.groupby([feature+"_bins_percentiles", feature+"_bins_percentiles_labels"], observed=False)[target].mean().reset_index()
        ax[0,1].set_title(f"{target_name} ~ {feature_name} ~> Percentile bins & Average target values", fontsize=14)
        ax[0,1].stem(binned_feature_agg_df[feature+"_bins_percentiles"], binned_feature_agg_df[target], basefmt=" ", linefmt="--k", markerfmt="ok")
        ax[0,1].set_ylim(binned_feature_agg_df[target].min() - binned_feature_agg_df[target].std(), binned_feature_agg_df[target].max() + binned_feature_agg_df[target].std())
        ax[0,1].xaxis.set_major_locator(MaxNLocator(nbins=21))
        ax[0,1].set_xlabel(feature_name + " Percentiles [0, 5, 10, ..., 95, 100]")
        ax[0,1].set_ylabel(target_name + " Average values")

        # Predicted probabilities distribution in comparison to actual values
        ax[0,2].set_title(f"Predicted Probability Distribution ~ {target_name} ", fontsize=14, alpha=.8)
        sns.violinplot(
            x=y, y=y_pred_prob, hue=y,
            split = False, inner="quartile", palette={0: "#3898FF", 1: "#FF4040"}, ax=ax[0,2]
        )
        ax[0,2].set_xlabel("Actual Boolean")
        ax[0,2].set_ylabel("Predicted probability")
        ax[0,2].get_legend().remove()

        # Logistic curve
        ax[1,0].set_title(f"{target_name} ~ {feature_name} ~ Logistic curve")
        sorted_idx = np.argsort(data_clean[feature])
        ax[1,0].scatter(data_clean[feature], y, color="black", alpha=alpha, label="Observed")
        ax[1,0].plot(data_clean[feature].iloc[sorted_idx], y_pred_prob.iloc[sorted_idx], color="red", linewidth=2, label="Logistic fit")
        ax[1,0].set_xlabel(feature_name)
        ax[1,0].set_ylabel("Predicted Probability = 1")
        ax[1,0].legend()

        # ROC and AUC
        ax[1,1].set_title("ROC Curve", fontsize=14)
        fpr, tpr, thresholds = roc_curve(y, y_pred_prob)
        auc_score = roc_auc_score(y, y_pred_prob)
        ax[1,1].plot(fpr, tpr, color="black", label=f"ROC curve, AUC = {auc_score:.3f}")
        ax[1,1].plot([0, 1], [0, 1], color="blue", linestyle="--", label="Random classifier")
        ax[1,1].set_xlabel("False Positive Rate")
        ax[1,1].set_ylabel("True Positive Rate")
        ax[1,1].legend(fontsize=14)

        # Calibration plot
        ax[1,2].set_title("Calibration Plot", fontsize=14)
        prob_true, prob_pred = calibration_curve(y, y_pred_prob, n_bins=20)
        ax[1,2].plot(prob_pred, prob_true, "o-", color="black")
        ax[1,2].plot([0,1], [0,1], "--", color="blue") 
        ax[1,2].set_xlabel("Predicted probability")
        ax[1,2].set_ylabel("Observed frequency")
        ax[1,2].grid(alpha=.5)

        for i in [0,1]:
            for j in [0,1,2]:
                ax[i,j].grid(alpha=.5)

        plt.tight_layout()
        plt.show()

    return metrics_df

# -----------------------------------------------------------
# Univariate Logistic regression [Logit] on categorical feature
# -----------------------------------------------------------
def univariate_logit_categorical(
        target: str, target_name: str, 
        feature: str, feature_name: str, 
        data: pd.DataFrame, 
        alpha: float=1.0,
        print_metrics: bool=True,
        show_plot: bool=True
        ) -> pd.DataFrame:
    
    if (print_metrics or show_plot) == True:
        print("-----------------------------------------------------")
        print(f"{target_name} ~ {feature_name} ~> Univariate Logistic Regression")
        print("-----------------------------------------------------\n")

    # Prepare data for modeling
    data = data.copy()

    # Dummies
    dummies = pd.get_dummies(data[feature], prefix=feature, drop_first=True).astype(int)
    data = pd.concat([data, dummies], axis=1)
    dummies_names = list(dummies.columns)

    data_clean = data[[target] + dummies_names].replace([np.inf, -np.inf], np.nan).dropna()
    X = sm.add_constant(data_clean[dummies_names])
    y = data_clean[target]

    # Model
    model = sm.Logit(y, X)
    fit = model.fit(disp=False)
    y_pred_prob = fit.predict(X)

    # Metrics
    metrics_full = {}
    for dummy in dummies_names:
        scaling_factor = np.std(data_clean[dummy]) / np.std(data_clean[target])
        metrics_dummy = {
            "coef": fit.params[dummy],
            "scaled coef": fit.params[dummy] * scaling_factor,
            "coef OR": np.exp(fit.params[dummy]),
            "scaled coef OR": np.exp(fit.params[dummy] * scaling_factor),
            "p-value": fit.pvalues[dummy],
            "standard error": fit.bse[dummy],
            "scaled standard error": fit.bse[dummy] * scaling_factor,
            "confidence interval": round(fit.conf_int().loc[dummy], 4).tolist(),
            "scaled confidence interval": round(fit.conf_int().loc[dummy] * scaling_factor, 4).tolist(),
            "confidence interval OR": round(np.exp(fit.conf_int().loc[dummy]), 4).tolist(),
            "scaled confidence interval OR": round(np.exp(fit.conf_int().loc[dummy] * scaling_factor), 4).tolist(),
            "Pseudo R2": fit.prsquared,
            "Log-Likelihood": fit.llf,
            "Likelihood Ratio (Chi2)": fit.llr,
            "Chi2 test p-value": fit.llr_pvalue ,
            "AIC": fit.aic,
            "BIC": fit.bic,
            "AUC": roc_auc_score(y, y_pred_prob),
        }

        metrics_full[dummy] = metrics_dummy

    if print_metrics:
        for metric in metrics_full[dummies_names[0]]:
            values = [metrics_full[dummy][metric] for dummy in dummies_names]

            if isinstance(values[0], (list, tuple)):
                row_str = " ".join([
                    f"[{dummy} = {[f'{v:.4f}' for v in val]}]" 
                    for dummy, val in zip(dummies_names, values)
                    ])
                
            else:  
                row_str = ", ".join([
                    f"[{dummy} = {[f'{v:.4f}' for v in metrics_full[dummy][metric]]}]"
                    if isinstance(metrics_full[dummy][metric], (list, tuple))
                    else f"[{dummy} = {metrics_full[dummy][metric]:.4f}]"
                    for dummy in dummies_names
                ])

            print(f"{metric} ~> {row_str}")

    # Plot
    if show_plot:
        fig, ax = plt.subplots(2, 3, figsize=(25, 10))

        # Feature distribution
        ax[0,0].set_title(f"{feature_name} Distribution", fontsize=14)
        prop_df = data[feature].value_counts(normalize=True) .reset_index()
        sns.barplot(x=feature, y="proportion", data=prop_df, color="grey", edgecolor="black", ax=ax[0,0])
        ax[0,0].set_xlabel(feature_name)

        # Binned feature & Average target values
        binned_feature_df = percentile_bin_feature(data = data_clean, bin_feature = feature, percentiles = np.arange(0,105,5))
        binned_feature_df[target] = data_clean[target]
        binned_feature_agg_df = binned_feature_df.groupby([feature+"_bins_percentiles", feature+"_bins_percentiles_labels"], observed=False)[target].mean().reset_index()
        ax[0,1].set_title(f"{target_name} ~ {feature_name} ~> Percentile bins & Average target values", fontsize=14)
        ax[0,1].stem(binned_feature_agg_df[feature+"_bins_percentiles"], binned_feature_agg_df[target], basefmt=" ", linefmt="--k", markerfmt="ok")
        ax[0,1].set_ylim(binned_feature_agg_df[target].min() - binned_feature_agg_df[target].std(), binned_feature_agg_df[target].max() + binned_feature_agg_df[target].std())
        ax[0,1].xaxis.set_major_locator(MaxNLocator(nbins=21))
        ax[0,1].set_xlabel(feature_name + " Percentiles [0, 5, 10, ..., 95, 100]")
        ax[0,1].set_ylabel(target_name + " Average values")

        # Predicted probabilities distribution in comparison to actual values
        ax[0,2].set_title(f"{target_name} ~ {feature_name} ~> Distribution", fontsize=14, alpha=.8)
        sns.violinplot(
            x=y, y=y_pred_prob, hue=y,
            split = False, inner="quartile", palette={0: "#3898FF", 1: "#FF4040"}, ax=ax[0,2]
        )
        ax[0,2].set_xlabel("Actual Boolean")
        ax[0,2].set_ylabel("Predicted probability")
        ax[0,2].get_legend().remove()

        # "Logistic curve"
        ax[1,0].set_title(f"Predicted Probability Distribution ~ {target_name} ")
        pred_prob_by_cat = pd.concat([data[feature], pd.Series(y_pred_prob, index=data.index)], axis=1)
        sns.pointplot(x=feature, y=y_pred_prob, data=pred_prob_by_cat, ax=ax[1,0], color="red")
        ax[1,0].set_xlabel(feature_name)
        ax[1,0].set_ylabel("Predicted Probability = 1")
        ax[1,0].legend()

        # ROC and AUC
        ax[1,1].set_title("ROC Curve", fontsize=14)
        fpr, tpr, thresholds = roc_curve(y, y_pred_prob)
        auc_score = roc_auc_score(y, y_pred_prob)
        ax[1,1].plot(fpr, tpr, color="black", label=f"ROC curve, AUC = {auc_score:.3f}")
        ax[1,1].plot([0, 1], [0, 1], color="blue", linestyle="--", label="Random classifier")
        ax[1,1].set_xlabel("False Positive Rate")
        ax[1,1].set_ylabel("True Positive Rate")
        ax[1,1].legend(fontsize=14)

        # Calibration plot
        ax[1,2].set_title("Calibration Plot", fontsize=14)
        prob_true, prob_pred = calibration_curve(y, y_pred_prob, n_bins=20)
        ax[1,2].plot(prob_pred, prob_true, "o-", color="black")
        ax[1,2].plot([0,1], [0,1], "--", color="blue")  # Perfect calibration line
        ax[1,2].set_xlabel("Predicted probability")
        ax[1,2].set_ylabel("Observed event frequency")
        ax[1,2].grid(alpha=.5)

        for i in [0,1]:
            for j in [0,1,2]:
                ax[i,j].grid(alpha=.5)

        plt.tight_layout()
        plt.show()

    metrics_df = pd.DataFrame(metrics_full).T.reset_index().rename(columns={"index": "feature"})
    return metrics_df

# -----------------------------------------------------------
# Full Train & Test Logistic regression [Logit] model
# -----------------------------------------------------------
def full_logit_model_analysis(
        target_dict: Dict[str, str],
        features_dict: Dict[str, str],
        train_data: pd.DataFrame, 
        test_data: pd.DataFrame,
        alpha: float=1.0,
        print_metrics: bool=True,
        show_plot: bool=True
        ) -> Dict[str, any]:

    target = list(target_dict.keys())[0]
    target_name = list(target_dict.values())[0]

    features = list(features_dict.keys())
    features_names = list(features_dict.values())
        
    if (print_metrics or show_plot) == True:
        print("-----------------------------------------------------")
        print(f"{target_name} ~\n{" + \n".join(features_names)} \n~> Logistic regression")
        print("-----------------------------------------------------\n")

    # Prepare data for modeling
    train_data = train_data.copy()
    test_data = test_data.copy()

    train_data_clean = train_data[[target] + features].replace([np.inf, -np.inf], np.nan).dropna()
    X_train = sm.add_constant(train_data_clean[features])
    y_train = train_data_clean[target]

    test_data_clean = test_data[[target] + features].dropna() 
    X_test = sm.add_constant(test_data_clean[features])
    y_test = test_data_clean[target]

    # Model
    model = sm.Logit(y_train, X_train)
    fit = model.fit(disp=False)

    y_pred_prob_train = fit.predict(X_train)
    y_pred_prob_test = fit.predict(X_test)

    # Metrics
    X_train_no_const = X_train.drop(columns="const", errors="ignore").copy()
    if X_train_no_const.shape[1] > 1:
        vif_train = pd.DataFrame({
            "feature": X_train_no_const.columns,
            "VIF": [variance_inflation_factor(X_train_no_const.values, i)
                    for i in range(X_train_no_const.shape[1])]
        })
    else:
        vif_train = pd.DataFrame({
            "feature": X_train_no_const.columns,
            "VIF": [np.nan]
        })

    train_model_metrics = {
        "Pseudo R2": fit.prsquared,
        "Log-Likelihood": fit.llf,
        "Likelihood Ratio (Chi2)": fit.llr,
        "Chi2 test p-value": fit.llr_pvalue ,
        "AIC": fit.aic,
        "BIC": fit.bic,
        "AUC": roc_auc_score(y_train, y_pred_prob_train),
        "mean VIF": vif_train["VIF"].mean(),
        "max VIF": vif_train["VIF"].max(),
    }
    train_model_metrics_df = pd.DataFrame([train_model_metrics])

    ll_model_test = log_likelihood(y_test, y_pred_prob_test)
    ll_null_test = log_likelihood(y_test, np.full_like(y_test, y_test.mean()))
    nr_features = len(features)
    nr_samples_test = len(y_test)
    test_model_metrics = {
        "Pseudo R2": 1 - (ll_model_test / ll_null_test),
        "Log-Likelihood": ll_model_test,
        "Likelihood Ratio (Chi2)": 2 * (ll_model_test - ll_null_test),
        "Chi2 test p-value": 1 - chi2.cdf(2 * (ll_model_test - ll_null_test), df=nr_features),
        "AIC": 2 * nr_features - 2 * ll_model_test,
        "BIC": np.log(nr_samples_test) * nr_features - 2 * ll_model_test,
        "AUC": roc_auc_score(y_test, y_pred_prob_test),
    }
    test_model_metrics_df = pd.DataFrame([test_model_metrics])

    features_metrics = []
    for single_feature in features:
        scaling_factor = np.std(train_data_clean[single_feature]) / np.std(train_data_clean[target])
        single_feature_metrics = {
            "feature": single_feature,
            "feature name": features_dict[single_feature],
            "coef": fit.params[single_feature],
            "scaled coef": fit.params[single_feature] * scaling_factor,
            "coef OR": np.exp(fit.params[single_feature]),
            "scaled coef OR": np.exp(fit.params[single_feature] * scaling_factor),
            "p-value": fit.pvalues[single_feature],
            "standard error": fit.bse[single_feature],
            "scaled standard error": fit.bse[single_feature] * scaling_factor,
            "confidence interval": round(fit.conf_int().loc[single_feature], 4).tolist(),
            "scaled confidence interval": round(fit.conf_int().loc[single_feature] * scaling_factor, 4).tolist(),
            "confidence interval OR": round(np.exp(fit.conf_int().loc[single_feature]), 4).tolist(),
            "scaled confidence interval OR": round(np.exp(fit.conf_int().loc[single_feature] * scaling_factor), 4).tolist(),
            "VIF": vif_train.loc[vif_train["feature"] == single_feature, "VIF"].values[0]
        }
        features_metrics.append(single_feature_metrics)
    features_metrics_df = pd.DataFrame(features_metrics)

    if print_metrics:

        print("\nModel metrics on Train data:")
        print("------------------------------------")
        for key, value in train_model_metrics.items():
            if isinstance(value, list):
                print(f"{key} ~> {[f'{v:.4f}' for v in value]}")
            else:
                print(f"{key} ~> {value:.4f}" if isinstance(value, float) else f"{key} ~> {value}")

        print("\nModel metrics on Test data:")
        print("------------------------------------")
        for key, value in test_model_metrics.items():
            if isinstance(value, list):
                print(f"{key} ~> {[f'{v:.4f}' for v in value]}")
            else:
                print(f"{key} ~> {value:.4f}" if isinstance(value, float) else f"{key} ~> {value}")

        print("\nFeatures metrics:")
        display(features_metrics_df)

    # Plot
    if show_plot:
        fig, ax = plt.subplots(2, 2, figsize=(25, 10))

        # Predicted proability distribution
        ax[0,0].set_title("Predicted Probability Distribution ~ Test data")
        ax[0,0].hist(y_pred_prob_test, bins=30, color="grey", edgecolor="black", density=True)
        sns.kdeplot(y_pred_prob_test, ax=ax[0,0], color="black", linewidth=2)
        ax[0,0].set_xlabel("Predicted probability")
        ax[0,0].set_ylabel("Density")

        # Predicted probabilities distribution in comparison to actual values
        ax[0,1].set_title(f"Predicted Probability Distribution ~ {target_name} ~ Test data", fontsize=14, alpha=.8)
        sns.violinplot(
            x=y_test, y=y_pred_prob_test, hue=y_test,
            split = False, inner="quartile", palette={0: "#3898FF", 1: "#FF4040"}, ax=ax[0,1]
        )
        ax[0,1].set_xlabel("Actual Boolean")
        ax[0,1].set_ylabel("Predicted probability")
        ax[0,1].get_legend().remove()

        # ROC and AUC
        ax[1,0].set_title("ROC Curve ~ Test data", fontsize=14)
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob_test)
        auc_score = roc_auc_score(y_test, y_pred_prob_test)
        ax[1,0].plot(fpr, tpr, color="black", label=f"ROC curve, AUC = {auc_score:.3f}")
        ax[1,0].plot([0, 1], [0, 1], color="blue", linestyle="--", label="Random classifier")
        ax[1,0].set_xlabel("False Positive Rate")
        ax[1,0].set_ylabel("True Positive Rate")
        ax[1,0].legend(fontsize=14)

        # Deviance Residuals plot
        ax[1,1].set_title("Residuals vs Predicted ~ Train data")
        residuals_train = np.sign(y_train - y_pred_prob_train) * np.sqrt(-2 * (y_train * np.log(y_pred_prob_train) + (1 - y_train) * np.log(1 - y_pred_prob_train)))
        ax[1,1].scatter(y_pred_prob_train, residuals_train, color="black", alpha=0.5)
        ax[1,1].axhline(0, color="red", linestyle="--")
        ax[1,1].set_xlabel("Predicted probability")
        ax[1,1].set_ylabel("Residuals")

        for i in [0,1]:
            for j in [0,1]:
                ax[i,j].grid(alpha=.5)

        plt.tight_layout()
        plt.show()

    # Return
    metrics = {
        "fit": fit,
        "features": features,
        "train_model_metrics": train_model_metrics_df,
        "test_model_metrics": test_model_metrics_df,
        "features_metrics": features_metrics_df,
    }

    return metrics
