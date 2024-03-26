# Here will be the auxiliar functions.
import pandas as pd
from imblearn.over_sampling import SMOTE
from scipy.stats import ttest_ind
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def convert_vars_to_factors(data:pd.DataFrame, colnames:list):
    for col in colnames:
        data[col] = data[col].astype('category')

    return data

def assign_categorical_levels(data: pd.DataFrame):
  categories_dict = {
    "credit_risk": {
      "categories": ["bad", "good"],
      "current_values": [0, 1]
    },
    "status": {
      "categories": ["no checking account", "... < 0 DM", "0<= ... < 200 DM", "... >= 200 DM / salary for at least 1 year"],
      "current_values": [1, 2, 3, 4]
    },
    "credit_history": {
      "categories": [
          "delay in paying off in the past",
          "critical account/other credits elsewhere",
          "no credits taken/all credits paid back duly",
          "existing credits paid back duly till now",
          "all credits at this bank paid back duly"
      ],
      "current_values": [0, 1, 2, 3, 4]
    },
    "purpose": {
      "categories": [
          "others", "car (new)", "car (used)", "furniture/equipment",
          "radio/television", "domestic appliances", "repairs", "education", 
          "vacation", "retraining", "business"
      ],
      "current_values": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    },
    "savings": {
      "categories": ["unknown/no savings account", "... <  100 DM", "100 <= ... <  500 DM", "500 <= ... < 1000 DM", "... >= 1000 DM"],
      "current_values": [1, 2, 3, 4, 5]
    },
    "employment_duration": {
      "categories": ["unemployed", "< 1 yr", "1 <= ... < 4 yrs", "4 <= ... < 7 yrs", ">= 7 yrs"],
      "current_values": [1, 2, 3, 4, 5]
    },
    "installment_rate": {
      "categories": [">= 35", "25 <= ... < 35", "20 <= ... < 25", "< 20"],
      "current_values": [1, 2, 3, 4]
    },
    "other_debtors": {
      "categories": ["none", "co-applicant", "guarantor"],
      "current_values": [1, 2, 3]
    },
    "personal_status_sex": {
      "categories": [
          "male : divorced/separated",
          "female : non-single or male : single",
          "male : married/widowed",
          "female : single"
      ],
      "current_values": [1, 2, 3, 4]
    },
    "present_residence": {
      "categories": ["< 1 yr", "1 <= ... < 4 yrs", "4 <= ... < 7 yrs", ">= 7 yrs"],
      "current_values": [1, 2, 3, 4]
    },
    "property": {
      "categories": [
          "unknown / no property", "car or other",
          "building soc. savings agr./life insurance", "real estate"
      ],
      "current_values": [1, 2, 3, 4]
    },
    "other_installment_plans": {
      "categories": ["bank", "stores", "none"],
      "current_values": [1, 2, 3]
    },
    "housing": {
      "categories": ["for free", "rent", "own"],
      "current_values": [1, 2, 3]
    },
    "number_credits": {
      "categories": ["1", "2-3", "4-5", ">= 6"],
      "current_values": [1, 2, 3, 4]
    },
    "job": {
      "categories": [
          "unemployed/unskilled - non-resident",
          "unskilled - resident",
          "skilled employee/official",
          "manager/self-empl./highly qualif. employee"
      ],
      "current_values": [1, 2, 3, 4]
    },
    "people_liable": {
       "categories": ["3 or more", "0 to 2"],
       "current_values": [1, 2]
    },
    "telephone": {
       "categories": ["no", "yes (under customer name)"],
       "current_values": [1, 2]
    },
    "foreign_worker": {
       "categories": ["yes", "no"],
       "current_values": [1, 2]
    }
  }

  for column, categories_data in categories_dict.items():
     data[column] = data[column].replace(to_replace=categories_data["current_values"], value=categories_data["categories"])

  return data

def create_data():
  names = [
    "status", "duration", "credit_history", "purpose", "amount",
    "savings", "employment_duration", "installment_rate",
    "personal_status_sex", "other_debtors",
    "present_residence", "property",
    "age", "other_installment_plans",
    "housing", "number_credits",
    "job", "people_liable", "telephone", "foreign_worker",
    "credit_risk"]

  data = pd.read_table('./data/SouthGermanCredit.asc', sep=' ', names=names, skiprows=1)


  return data

def stratified_subsampling(df, stratification_variables, number_of_samples):

    df_stratified = df.groupby(stratification_variables)
    sample = pd.DataFrame()
    for i, group in df_stratified:
        sample_size_stratum = round((number_of_samples / df.shape[0]) * group.shape[0])
        while sample_size_stratum > group.shape[0]:
            number_of_samples -= 1
            sample_size_stratum = round((number_of_samples / df.shape[0]) * group.shape[0])
        sample = pd.concat([sample, group.sample(sample_size_stratum, random_state=42)])
    return sample

def smote_oversampling(X_vars, y_var):
   smote = SMOTE(random_state=5)
   x_resampled, y_resampled = smote.fit_resample(X_vars, y_var)
   return pd.concat([x_resampled, y_resampled], axis=1)

def merge_two_subsets(subset_1: pd.DataFrame, subset_2: pd.DataFrame):
   return pd.concat([subset_1, subset_2], ignore_index=True)

def create_balanced_sample(full_data: pd.DataFrame, subsampling_features:list, subsampling_size:int):
   subsample = stratified_subsampling(full_data.loc[full_data.credit_risk.isin([1])], subsampling_features, subsampling_size)
   subsample = merge_two_subsets(subsample, full_data.loc[full_data.credit_risk.isin([0])])
   subsample = smote_oversampling(X_vars=subsample.drop("credit_risk", axis=1), y_var=subsample.credit_risk)
   return subsample

def verify_sample_belongs_to_population(data:pd.DataFrame, subset: pd.DataFrame):
   return ttest_ind(data, subset)

def standardize_continuous_variables(df: pd.DataFrame, column_list: list):
    scaler = StandardScaler()
    df_continuous_columns = df[column_list]
    df_standardized = pd.DataFrame(scaler.fit_transform(df_continuous_columns), columns=column_list)
    df.update(df_standardized)
    return df

def remove_outliers(df, column_list, threshold=1.5):
    df_outliers = pd.DataFrame()

    for col in column_list:
        if df[col].dtype != 'object':  # Check if the column is numeric
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            df_outliers = pd.concat([df_outliers, outliers])

    df_outliers = df_outliers.drop_duplicates()
    df_no_outliers = df[~df.index.isin(df_outliers.index)]

    return df_no_outliers, df_outliers

def remove_highly_correlated_variables(df, correlation_threshold):

  correlation_matrix = df.corr()
  correlated_pairs = []
  for i in range(len(correlation_matrix)):
    for j in range(i + 1, len(correlation_matrix)):
      if abs(correlation_matrix.iloc[i, j]) > correlation_threshold:
        correlated_pairs.append((i, j))

  for i, j in correlated_pairs:
    df = df.drop(df.columns[i], axis=1)

  return df

def principal_components_analysis(data:pd.DataFrame):
  n_components = len(data.columns)
  # Realizar PCA
  pca = PCA(n_components=n_components)  # Puedes ajustar el número de componentes según tus necesidades
  df_pca = pca.fit_transform(data)

  # Crear un DataFrame con los resultados del PCA
  df_pca_result = pd.DataFrame(data=df_pca, columns=[f'PC{x}' for x in range(n_components) ])

  return df_pca_result, pca

def get_representative_vars_by_pca(data, pca_instance):
  df_loadings = pd.DataFrame(pca_instance.components_, columns=data.columns)

  top_variables = df_loadings.iloc[0].abs().sort_values(ascending=False).index
  return top_variables