# Here will be the auxiliar functions.
import pandas as pd

def convert_vars_to_factors(data:pd.DataFrame, colnames:list):
  for col in data.columns.difference(colnames):
    data[col] = data[col].astype('category')

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

  data = convert_vars_to_factors(data=data, colnames=["duration", "amount", "age", "people_liable"])

  return data