# Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns
from typing import Dict
from IPython.display import display

import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr
from scipy.special import inv_boxcox
from scipy.stats import boxcox, boxcox_normmax
from statsmodels.stats.outliers_influence import variance_inflation_factor

from helpers.common_help_functions import percentile_bin_feature

# -----------------------------------------------------------
# Univariate Linear regression [OLS] on Numeric feature
# -----------------------------------------------------------
def univariate_ols_numeric(
        target: str, target_name: str, 
        feature: str, feature_name: str, 
        data: pd.DataFrame, 
        alpha: float=1.0,
        box_cox_transformation: bool=False, 
        print_metrics: bool=True,
        show_plot: bool=True
        ) -> pd.DataFrame:
    
    if (print_metrics or show_plot) == True:
        print("-----------------------------------------------------")
        print(f"{target_name} ~ {feature_name} ~> Univariate Linear Regression")
        print("-----------------------------------------------------\n")

    # Prepare data for modeling
    data = data.copy()

    if box_cox_transformation:
        target_name = target_name + " [Box-Cox Transformed]"
        target_bc = data[target]
        lambda_bc = boxcox_normmax(target_bc)
        data[target+"_bc"] = boxcox(target_bc, lmbda=lambda_bc)
        target = target+"_bc"

    data_clean = data[[target, feature]].replace([np.inf, -np.inf], np.nan).dropna()
    X = sm.add_constant(data_clean[[feature]])
    y = data_clean[target]

    # Model
    model = sm.OLS(y, X)
    fit = model.fit()
    y_pred = fit.predict(X)

    if box_cox_transformation:
        prop_clipped = round((y_pred < 1e-6).mean(), 2)
        y_pred = np.where(y_pred < 1e-6, 1e-6, y_pred)
        
        y_pred = inv_boxcox(y_pred, lambda_bc)
        y = inv_boxcox(y, lambda_bc)

    # Metrics
    scaling_factor = np.std(data_clean[feature]) / np.std(data_clean[target])
    metrics = {
        "proportion of clipped Box-Cox predicted values": prop_clipped if box_cox_transformation else None,
        "coef": fit.params[feature],
        "scaled coef": fit.params[feature] * scaling_factor,
        "p-value": fit.pvalues[feature],
        "standard error": fit.bse[feature],
        "scaled standard error": fit.bse[feature] * scaling_factor,
        "confidence interval": round(fit.conf_int().loc[feature], 4).tolist(),
        "scaled confidence interval": round(fit.conf_int().loc[feature] * scaling_factor, 4).tolist(),
        "R2": fit.rsquared,
        "R2 adjusted": fit.rsquared_adj,
        "MAE": mean_absolute_error(y, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y, y_pred)),
        "Pearson correlation coefficient": pearsonr(y, y_pred)[0]
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
            
        # Scatter plot with regression line
        ax[0,2].set_title(f"{target_name} ~ {feature_name} ~> Scatter plot & Regression line", fontsize=14)
        ax[0,2].scatter(data_clean[feature], data_clean[target], color="black", s=15, alpha=alpha)
        sns.regplot(x=data_clean[feature], y=data_clean[target], ax=ax[0,2], scatter=False, line_kws={"color":"red", "linewidth":2})
        ax[0,2].set_xlabel(feature_name)
        ax[0,2].set_ylabel(target_name)

        # Influence / Leverage plot
        ax[1,0].set_title("Influence plot", fontsize=14)
        influence = fit.get_influence()
        leverage = influence.hat_matrix_diag
        standard_resid = influence.resid_studentized_internal
        ax[1,0].scatter(leverage, standard_resid, color="black", s=15, alpha=alpha)
        ax[1,0].axhline(0, color="red", linestyle="--")
        ax[1,0].set_xlabel("Leverage ~ Outliers in {feature}")
        ax[1,0].set_ylabel("Standardized Residuals ~ Affect slope")

        # Residuals scatter ~ Linearity assumption, Homoscedasticity, Outliers, Centering around zero
        ax[1,1].set_title("Predicted values vs. Residuals", fontsize=14)
        residuals = y - y_pred
        ax[1,1].scatter(y_pred, residuals, color="black", s=15, alpha=alpha)
        ax[1,1].axhline(0, color="red", linestyle="--")
        ax[1,1].set_xlabel("Predictions")
        ax[1,1].set_ylabel("Residuals")

        # Residuals KDE ~ Normality assumption
        ax[1,2].set_title("Residuals distribution", fontsize=14)
        ax[1,2].hist(residuals, bins=30, density=True, color="grey", edgecolor="black", alpha=1.0)
        sns.kdeplot(residuals, ax=ax[1,2], color="black", linewidth=2)
        ax[1,2].set_xlabel("Residuals")
        ax[1,2].set_ylabel("Density")

        for i in [0,1]:
            for j in [0,1,2]:
                ax[i,j].grid(alpha=.5)

        plt.tight_layout()
        plt.show()

    return metrics_df

# -----------------------------------------------------------
# Univariate Linear regression [OLS] on Categorical feature
# -----------------------------------------------------------
def univariate_ols_categorical(
        target: str, target_name: str, 
        feature: str, feature_name: str, 
        data: pd.DataFrame, 
        alpha: float=1.0,
        box_cox_transformation: bool=False, 
        print_metrics: bool=True,
        show_plot: bool=True
        ) -> pd.DataFrame:
    
    if (print_metrics or show_plot) == True:
        print("-----------------------------------------------------")
        print(f"{target_name} ~ {feature_name} ~> Univariate Linear Regression")
        print("-----------------------------------------------------\n")

    # Prepare data for modeling
    data = data.copy()

    if box_cox_transformation:
        target_name = target_name + " [Box-Cox Transformed]"
        target_bc = data[target]
        lambda_bc = boxcox_normmax(target_bc)
        data[target+"_bc"] = boxcox(target_bc, lmbda=lambda_bc)
        target = target+"_bc"

    dummies = pd.get_dummies(data[feature], prefix=feature, drop_first=True).astype(int)
    data = pd.concat([data, dummies], axis=1)
    dummies_names = list(dummies.columns)

    data_clean = data[[target] + dummies_names].replace([np.inf, -np.inf], np.nan).dropna()
    X = sm.add_constant(data_clean[dummies_names])
    y = data_clean[target]

    # Model
    model = sm.OLS(y, X)
    fit = model.fit()
    y_pred = fit.predict(X)

    if box_cox_transformation:
        prop_clipped = round((y_pred < 1e-6).mean(), 2)
        y_pred = np.where(y_pred < 1e-6, 1e-6, y_pred)
        
        y_pred = inv_boxcox(y_pred, lambda_bc)
        y = inv_boxcox(y, lambda_bc)

    # Metrics
    metrics_full = {}
    for dummy in dummies_names:
        scaling_factor = np.std(data_clean[dummy]) / np.std(data_clean[target])
        metrics_dummy = {
            "proportion of clipped Box-Cox predicted values": prop_clipped if box_cox_transformation else None,
            "coef": fit.params[dummy],
            "scaled coef": fit.params[dummy] * scaling_factor,
            "p-value": fit.pvalues[dummy],
            "standard error": fit.bse[dummy],
            "scaled standard error": fit.bse[dummy] * scaling_factor,
            "confidence interval": round(fit.conf_int().loc[dummy], 4).tolist(),
            "scaled confidence interval": round(fit.conf_int().loc[dummy] * scaling_factor, 4).tolist(),
            "R2": fit.rsquared,
            "R2 adjusted": fit.rsquared_adj,
            "MAE": mean_absolute_error(y, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y, y_pred)),
            "Pearson correlation coefficient": pearsonr(y, y_pred)[0]
        }

        metrics_full[dummy] = metrics_dummy

    def format_metric(val):
        if isinstance(val, (list, tuple)):
            return f"[{', '.join(f'{v:.4f}' for v in val)}]"
        elif val is None:
            return "None"
        else:
            return f"{val:.4f}"

    if print_metrics:
        for metric in metrics_full[dummies_names[0]]:
            row_str = ", ".join([f"[{dummy} = {format_metric(metrics_full[dummy][metric])}]" for dummy in dummies_names])
            print(f"{metric} ~> {row_str}")

    # Plot
    if show_plot:
        fig, ax = plt.subplots(2, 3, figsize=(25, 10))

        # Feature distribution
        ax[0,0].set_title(f"{feature_name} Distribution", fontsize=14)
        prop_df = data[feature].value_counts(normalize=True) .reset_index()
        sns.barplot(x=feature, y="proportion", data=prop_df, color="grey", edgecolor="black", ax=ax[0,0])
        ax[0,0].set_xlabel(feature_name)

        # Average target values
        feature_agg_df = data.groupby(feature, observed=False)[target].mean().reset_index()
        ax[0,1].set_title(f"Average {target_name} values ~ {feature_name} category", fontsize=14)
        ax[0,1].stem(feature_agg_df[feature], feature_agg_df[target], basefmt=" ", linefmt="--k", markerfmt="ok")
        ax[0,1].set_ylim(feature_agg_df[target].min() - feature_agg_df[target].std(), feature_agg_df[target].max() + feature_agg_df[target].std())
        ax[0,1].set_xlabel(feature_name)
        ax[0,1].set_ylabel(target_name + " Average values")

        # Scatter plot with regression line
        ax[0,2].set_title(f"{target_name} ~ {feature_name} ~> Scatter plot & Regression line", fontsize=14)
        ax[0,2].scatter(data_clean[dummies_names], data_clean[target], color="black", s=15, alpha=alpha)
        sns.regplot(x=data_clean[dummies_names], y=data_clean[target], ax=ax[0,2], scatter=False, line_kws={"color":"red", "linewidth":2})
        ax[0,2].set_xlabel(feature_name)
        ax[0,2].set_ylabel(target_name)

        # Influence / Leverage plot
        ax[1,0].set_title("Influence plot", fontsize=14)
        influence = fit.get_influence()
        leverage = influence.hat_matrix_diag
        standard_resid = influence.resid_studentized_internal
        ax[1,0].scatter(leverage, standard_resid, color="black", s=15, alpha=alpha)
        ax[1,0].axhline(0, color="red", linestyle="--")
        ax[1,0].set_xlabel("Leverage ~ Outliers in {feature}")
        ax[1,0].set_ylabel("Standardized Residuals ~ Affect slope")

        # Residuals scatter ~ Linearity assumption, Homoscedasticity, Outliers, Centering around zero
        ax[1,1].set_title("Predicted values vs. Residuals", fontsize=14)
        residuals = y - y_pred
        ax[1,1].scatter(y_pred, residuals, color="black", s=15, alpha=alpha)
        ax[1,1].axhline(0, color="red", linestyle="--")
        ax[1,1].set_xlabel("Predictions")
        ax[1,1].set_ylabel("Residuals")

        # Residuals KDE ~ Normality assumption
        ax[1,2].set_title("Residuals distribution", fontsize=14)
        ax[1,2].hist(residuals, bins=30, density=True, color="grey", edgecolor="black", alpha=1.0)
        sns.kdeplot(residuals, ax=ax[1,2], color="black", linewidth=2)
        ax[1,2].set_xlabel("Residuals")
        ax[1,2].set_ylabel("Density")

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
def full_ols_model_analysis(
        target_dict: Dict[str, str],
        features_dict: Dict[str, str],
        train_data: pd.DataFrame, 
        test_data: pd.DataFrame,
        alpha: float=1.0,
        box_cox_transformation: bool=False, 
        print_metrics: bool=True,
        show_plot: bool=True
        ) -> Dict[str, any]:

    target = list(target_dict.keys())[0]
    target_name = list(target_dict.values())[0]

    features = list(features_dict.keys())
    features_names = list(features_dict.values())
        
    if (print_metrics or show_plot) == True:
        print("-----------------------------------------------------")
        print(f"{target_name} ~\n{" + \n".join(features_names)} \n~> Linear regression")
        print("-----------------------------------------------------\n")

    # Prepare data for modeling
    train_data = train_data.copy()
    test_data = test_data.copy()

    if box_cox_transformation:
        target_name = target_name + " [Box-Cox Transformed]"
        target_bc = train_data[target]
        lambda_bc = boxcox_normmax(target_bc)
        train_data[target+"_bc"] = boxcox(train_data[target], lmbda=lambda_bc)
        test_data[target+"_bc"] = boxcox(test_data[target], lmbda=lambda_bc)
        target = target+"_bc"

    train_data_clean = train_data[[target] + features].replace([np.inf, -np.inf], np.nan).dropna()
    X_train = sm.add_constant(train_data_clean[features])
    y_train = train_data_clean[target]

    test_data_clean = test_data[[target] + features].dropna() 
    X_test = sm.add_constant(test_data_clean[features])
    y_test = test_data_clean[target]

    # Model
    model = sm.OLS(y_train, X_train)
    fit = model.fit()

    y_pred_train = fit.predict(X_train)
    y_pred_test = fit.predict(X_test)

    if box_cox_transformation:
        prop_clipped_train = round((y_pred_train < 1e-6).mean(), 2)
        y_pred_train = np.where(y_pred_train < 1e-6, 1e-6, y_pred_train)
        y_pred_train = inv_boxcox(y_pred_train, lambda_bc)
        y_train = inv_boxcox(y_train, lambda_bc)

        prop_clipped_test = round((y_pred_test < 1e-6).mean(), 2)
        y_pred_test = np.where(y_pred_test < 1e-6, 1e-6, y_pred_test)
        y_pred_test = inv_boxcox(y_pred_test, lambda_bc)
        y_test = inv_boxcox(y_test, lambda_bc)

    # Metrics
    X_train_no_const = X_train.drop(columns="const", errors="ignore").copy()
    if X_train_no_const.shape[1] > 1:
        vif_train = pd.DataFrame({
            "feature": X_train_no_const.columns,
            "VIF": [variance_inflation_factor(X_train_no_const.values, i) for i in range(X_train_no_const.shape[1])]
        })
    else:
        vif_train = pd.DataFrame({
            "feature": X_train_no_const.columns,
            "VIF": [np.nan]
        })
        
    train_model_metrics = {
        "proportion of clipped Box-Cox predicted values": prop_clipped_train if box_cox_transformation else None,
        "R2": fit.rsquared,
        "R2 adjusted": fit.rsquared_adj,
        "MAE": mean_absolute_error(y_train, y_pred_train),
        "RMSE": np.sqrt(mean_squared_error(y_train, y_pred_train)),
        "Pearson correlation coefficient": pearsonr(y_train, y_pred_train)[0],
        "mean VIF": vif_train["VIF"].mean(),
        "max VIF": vif_train["VIF"].max(),
    }
    train_model_metrics_df = pd.DataFrame([train_model_metrics])

    test_model_metrics = {
        "proportion of clipped Box-Cox predicted values": prop_clipped_test if box_cox_transformation else None,
        "R2": r2_score(y_test, y_pred_test),
        "MAE": mean_absolute_error(y_test, y_pred_test),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_test)),
        "Pearson correlation coefficient": pearsonr(y_test, y_pred_test)[0],
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
            "p-value": fit.pvalues[single_feature],
            "standard error": fit.bse[single_feature],
            "scaled standard error": fit.bse[single_feature] * scaling_factor,
            "confidence interval": round(fit.conf_int().loc[single_feature], 4).tolist(),
            "scaled confidence interval": round(fit.conf_int().loc[single_feature] * scaling_factor, 4).tolist(),
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

        # Scatter plot with regression line
        ax[0,0].set_title("Scatter plot & Regression line ~ Test data", fontsize=14)
        ax[0,0].scatter(y_pred_test, y_test, color="black", s=15, alpha=alpha)
        sns.regplot(x=y_pred_test, y=y_test, ax=ax[0,0], scatter=False, line_kws={"color":"red", "linewidth":2}, label="Regression line")
        ax[0,0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="blue", linewidth=2, linestyle="--", label="Perfect line [x=y]")
        ax[0,0].set_xlabel("Predictions")
        ax[0,0].set_ylabel("Actual values")
        ax[0,0].legend()

        # Predicted values and actual values distribution on Test data
        ax[0,1].set_title("Predicted & Actual values distribution ~ Test data", fontsize=14)
        residuals_train = y_train - y_pred_train
        ax[0,1].hist(y_pred_test, bins=30, density=True, color="red", edgecolor="red", alpha=0.3)
        sns.kdeplot(y_pred_test, ax=ax[0,1], color="red", linewidth=2, label="Predictions")
        ax[0,1].hist(y_test, bins=30, density=True, color="blue", edgecolor="blue", alpha=0.3)
        sns.kdeplot(y_test, ax=ax[0,1], color="blue", linewidth=2, label="Actuals")
        ax[0,1].set_xlabel("Residuals")
        ax[0,1].set_ylabel("Density")
        ax[0,1].legend()

        # Residuals scatter on Train data ~ Linearity assumption, Homoscedasticity, Outliers, Centering around zero
        ax[1,0].set_title("Predicted values vs. Residuals ~ Train data", fontsize=14)
        residuals_train = y_train - y_pred_train
        ax[1,0].scatter(y_pred_train, residuals_train, color="black", s=15, alpha=alpha)
        ax[1,0].axhline(0, color="red", linestyle="--")
        ax[1,0].set_xlabel(f"Predicted")
        ax[1,0].set_ylabel("Residuals")

        # Residuals KDE ~ Normality assumption
        ax[1,1].set_title("Residuals distribution ~ Train data", fontsize=14)
        residuals_train = y_train - y_pred_train
        ax[1,1].hist(residuals_train, bins=30, density=True, color="grey", edgecolor="black", alpha=1.0)
        sns.kdeplot(residuals_train, ax=ax[1,1], color="black", linewidth=2)
        ax[1,1].set_xlabel("Residuals")
        ax[1,1].set_ylabel("Density")

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
