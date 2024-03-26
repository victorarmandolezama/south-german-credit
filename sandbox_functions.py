# Here will be the auxiliar functions.
import pandas as pd
from imblearn.over_sampling import SMOTE

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

  data = convert_vars_to_factors(data=data, colnames=[
    'credit_risk',
    'status',
    'credit_history',
    'purpose',
    'savings',
    'employment_duration',
    'installment_rate',
    'other_debtors',
    'personal_status_sex',
    'present_residence',
    'property',
    'other_installment_plans',
    'housing',
    'number_credits',
    'job',
    'people_liable',
    'telephone',
    'foreign_worker',
  ])

  data = assign_categorical_levels(data=data)

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
   return smote.fit_resample(X_vars, y_var)