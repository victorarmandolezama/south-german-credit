import pandas as pd
import matplotlib.pyplot as plt


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
