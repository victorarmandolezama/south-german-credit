import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import chi2_contingency


def get_statistics(df: pd.DataFrame, columnas_continuas, columnas_categoricas):

    # Estadísticos para variables continuas
    estadisticos_continuas = df[columnas_continuas].describe()

    # Estadísticos para variables categóricas
    estadisticos_categoricas = df[columnas_categoricas].describe()

    # Frecuencias relativas para variables categóricas
    frecuencias_relativas = {col: df[col].value_counts(
        normalize=True) for col in columnas_categoricas}

    return estadisticos_continuas, estadisticos_categoricas, frecuencias_relativas


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

    dtypes = {
        "status": "category",
        "duration": "int",
        "credit_history": "category",
        "purpose": "category",
        "amount": "float",
        "savings": "category",
        "employment_duration": "category",  # Asumido como categórico por ser ordinal
        "installment_rate": "category",     # Asumido como categórico por ser ordinal
        "personal_status_sex": "category",
        "other_debtors": "category",
        "present_residence": "category",     # Asumido como categórico por ser ordinal
        "property": "category",              # Asumido como categórico por ser ordinal
        "age": "int",
        "other_installment_plans": "category",
        "housing": "category",
        "number_credits": "category",             # Asumido como entero
        "job": "category",                    # Asumido como categórico por ser ordinal
        "people_liable": "category",
        # Asumido como categórico (binario)
        "telephone": "category",
        # Asumido como categórico (binario)
        "foreign_worker": "category",
        # Asumido como categórico (binario)
        "credit_risk": "category"
    }

    data = pd.read_table('./data/SouthGermanCredit.asc',
                         sep=' ', names=names, skiprows=1, dtype=dtypes)

    return data


def plot_frequencies(df: pd.DataFrame, columns: list, n_per_row: int = 3):
    """
    Generates bar charts for the frequency of specified categorical columns in the DataFrame.

    Args:
    df (pd.DataFrame): The DataFrame containing the data.
    columns (list): A list of column names for which to plot the frequency.
    n_per_row (int): Number of plots per row in the figure (default is 3).
    """
    total_columns = len(columns)
    # Calculate the number of rows needed
    n_rows = (total_columns + n_per_row - 1) // n_per_row

    fig, axes = plt.subplots(
        n_rows, n_per_row, figsize=(n_per_row * 5, n_rows * 4))
    axes = axes.flatten()  # Flatten the axes array for easier indexing

    for i, col in enumerate(columns):
        if col in df.columns:
            # Calculate frequencies
            frequencies = df[col].value_counts()

            # Plot the frequencies
            axes[i].bar(frequencies.index.astype(str),
                        frequencies.values, color='skyblue')
            axes[i].set_title(f'Frequency of {col}')
            axes[i].set_ylabel('Frequency')
            axes[i].set_xlabel(col)
            axes[i].tick_params(axis='x', rotation=45)  # Rotate x-axis labels
        else:
            axes[i].remove()  # Remove the plot if the column does not exist

    # Hide empty axes if there are fewer columns than spaces
    for j in range(i + 1, len(axes)):
        axes[j].remove()

    plt.tight_layout()  # Adjust spacing between plots
    plt.show()


def plot_boxplot(data, column, figsize=(10, 6)):
    plt.figure(figsize=figsize)
    plt.boxplot(data[column], patch_artist=True,
                boxprops=dict(facecolor='skyblue'))
    plt.title(f'Boxplot of {column}')
    plt.ylabel(column)
    # plt.grid(axis='y', alpha=0.75)
    plt.show()

def remove_outliers(df, column_list, threshold=1.5):
  df_outliers = pd.DataFrame()

  for col in column_list:
      if df[col].dtype != 'object':
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


def generate_synthetic_variables(df: pd.DataFrame, rules: list) -> pd.DataFrame:
    """
    Generates synthetic variables based on specified rules for string values
    and returns a new DataFrame containing only the synthetic variables.

    Args:
    df (pd.DataFrame): The original DataFrame containing the data.
    rules (list): A list of dictionaries containing the rules for generating synthetic variables.

    Returns:
    pd.DataFrame: A new DataFrame with only the synthetic variables added.
    """
    synthetic_df = pd.DataFrame()  # Create an empty DataFrame for synthetic variables

    for rule in rules:
        name = rule['name']
        conditions = rule['conditions']
        operator = rule['operator']

        # Initialize combined_condition based on the first condition
        first_condition = conditions[0]
        col = first_condition['col']
        values = first_condition['values']
        combined_condition = df[col].isin(values)  # Start with the first condition

        # Combine remaining conditions
        for condition in conditions[1:]:
            col = condition['col']
            values = condition['values']
            current_condition = df[col].isin(values)  # True if any value matches

            # Combine with the existing condition using the specified operator
            if operator == 'AND':
                combined_condition = combined_condition & current_condition
            elif operator == 'OR':
                combined_condition = combined_condition | current_condition
            else:
                raise ValueError("Operator must be 'AND' or 'OR'.")

        # Add the new column to the synthetic DataFrame
        synthetic_df[name] = combined_condition.astype(int)

    return synthetic_df

def merge_two_subsets(subset_1: pd.DataFrame, subset_2: pd.DataFrame, axis, ignore_index):
  return pd.concat([subset_1, subset_2], ignore_index=ignore_index, axis=axis)

def plot_correlation_matrix(df: pd.DataFrame, use_mask: bool, threshold: float, columns: list = None):
    """
    Plots a correlation matrix for the given DataFrame and returns the most correlated column pairs
    above a specified threshold.

    Args:
    df (pd.DataFrame): The input DataFrame.
    use_mask (bool): Whether to use a mask to hide the upper triangle of the correlation matrix.
    threshold (float): The minimum correlation coefficient to consider.
    columns (list): Optional list of column names to include in the correlation matrix.

    Returns:
    list: A list of tuples containing the names of the most correlated column pairs and their correlation values.
    """
    
    currDf = pd.DataFrame(df)

    # Select specified columns if provided
    if columns is not None:
        currDf = currDf[columns]

    # Calculate the correlation matrix
    corr = currDf.corr()

    # Create a mask for the upper triangle
    mask = None
    if use_mask:
        mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    plt.figure(figsize=(10, 8))

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=True, fmt=".2f", square=True, 
                linewidths=.5, cbar_kws={"shrink": .8})

    # Title the heatmap
    plt.title('Correlation Matrix', fontsize=16)
    plt.show()

    # Find column pairs with correlation above the threshold
    correlated_pairs = []
    for col1 in corr.columns:
        for col2 in corr.columns:
            if col1 != col2:
                corr_value = corr[col1][col2]
                if abs(corr_value) > threshold:
                    correlated_pairs.append((col1, col2, corr_value))

    # Sort pairs by correlation value in descending order
    correlated_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    return correlated_pairs

# Example usage
# data = {'A': [1, 2, 3, 4], 'B': [2, 3, 4, 5], 'C': [5, 4, 3, 2]}
# df = pd.DataFrame(data)
# result = plot_correlation_matrix(df, use_mask=True, threshold=0.7)
# print(result)

def create_dummy_variables(data, categorical_column, target_column):
    # Crear variables dummy a partir de la columna categórica
    dummies = pd.get_dummies(data[categorical_column], prefix=categorical_column, drop_first=False)
    
    # Concatenar las variables dummy con la columna objetivo
    new_data = pd.concat([pd.get_dummies(data[[target_column]], drop_first=True), dummies], axis=1)
    
    return new_data

def find_correlations(dummy_data: pd.DataFrame, target_column):
    # Calcular la correlación entre las variables dummy y la variable objetivo
    correlation = dummy_data.corr('pearson')

    if target_column not in correlation.columns:
        raise ValueError(f"Column '{target_column}' not found in the correlation matrix.")

    correlation = correlation[target_column]
    
    # # Obtener la variable con la mayor correlación
    # max_correlation_variable = correlation.idxmax()
    # max_correlation_value = correlation.max()
    
    return correlation

def filter_data_by_value(data, categorical_column, target_column, value_to_compare):
    # Filtrar el conjunto de datos según el valor especificado en la columna categórica
    filtered_data = data[data[categorical_column] == value_to_compare]
    
    return filtered_data[[target_column] + [col for col in data.columns if col != target_column]]

def cramers_v(data, categorical_column, target_column):
    confusion_matrix = pd.crosstab(data[categorical_column], data[target_column])  # Crear tabla de contingencia
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    return np.sqrt(phi2 / min(k - 1, r - 1))

def chi_squared_test(dummy_data, target_column):
    # Inicializar un diccionario para almacenar los resultados
    results = {}
    
    # Obtener el nombre de las columnas dummy
    dummy_columns = [col for col in dummy_data.columns if col != target_column]
    
    # Realizar la prueba Chi-Cuadrado para cada variable dummy
    for col in dummy_columns:
        # Crear una tabla de contingencia
        contingency_table = pd.crosstab(dummy_data[target_column], dummy_data[col])
        
        # Realizar la prueba Chi-Cuadrado
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        
        # Almacenar los resultados en el diccionario
        results[col] = {
            'chi2_statistic': chi2,
            'p_value': p,
            'degrees_of_freedom': dof,
            'expected_frequencies': expected
        }
    
    return results
