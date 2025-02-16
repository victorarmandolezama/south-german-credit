import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

legends = {
    'status': {
        '1': 'No checking account',
        '2': 'Balance less than 0 DM',
        '3': 'Balance between 0 and 200 DM',
        '4': 'Balance greater than or equal to 200 DM / salary for at least 1 year'
    },
    'credit_history': {
        '0': 'Delay in paying off in the past',
        '1': 'Critical account/other credits elsewhere',
        '2': 'No credits taken/all credits paid back duly',
        '3': 'Existing credits paid back duly till now',
        '4': 'All credits at this bank paid back duly'
    },
    'purpose': {
        '0': 'Others',
        '1': 'Car (new)',
        '2': 'Car (used)',
        '3': 'Furniture/equipment',
        '4': 'Radio/television',
        '5': 'Domestic appliances',
        '6': 'Repairs',
        '7': 'Education',
        '8': 'Vacation',
        '9': 'Retraining',
        '10': 'Business'
    },
    'savings': {
        '1': 'Unknown/no savings account',
        '2': 'Balance less than 100 DM',
        '3': 'Balance between 100 and 500 DM',
        '4': 'Balance between 500 and 1000 DM',
        '5': 'Balance greater than or equal to 1000 DM'
    },
    'employment_duration': {
        '1': 'Unemployed',
        '2': 'Less than 1 year',
        '3': 'Between 1 and 4 years',
        '4': 'Between 4 and 7 years',
        '5': 'Greater than or equal to 7 years'
    },
    'installment_rate': {
        '1': 'Greater than or equal to 35',
        '2': 'Between 25 and 35',
        '3': 'Between 20 and 25',
        '4': 'Less than 20'
    },
    'personal_status_sex': {
        '1': 'Male: divorced/separated',
        '2': 'Female: non-single or male: single',
        '3': 'Male: married/widowed',
        '4': 'Female: single'
    },
    'other_debtors': {
        '1': 'None',
        '2': 'Co-applicant',
        '3': 'Guarantor'
    },
    'present_residence': {
        '1': 'Less than 1 year',
        '2': 'Between 1 and 4 years',
        '3': 'Between 4 and 7 years',
        '4': 'Greater than or equal to 7 years'
    },
    'property': {
        '1': 'Unknown / no property',
        '2': 'Car or other',
        '3': 'Building society savings agreement/life insurance',
        '4': 'Real estate'
    },
    'other_installment_plans': {
        '1': 'Bank',
        '2': 'Stores',
        '3': 'None'
    },
    'housing': {
        '1': 'For free',
        '2': 'Rent',
        '3': 'Own'
    },
    'number_credits': {
        '1': '1',
        '2': '2-3',
        '3': '4-5',
        '4': '6 or more'
    },
    'job': {
        '1': 'Unemployed/unskilled - non-resident',
        '2': 'Unskilled - resident',
        '3': 'Skilled employee/official',
        '4': 'Manager/self-employed/highly qualified employee'
    },
    'people_liable': {
        '1': '3 or more',
        '2': '0 to 2'
    },
    'telephone': {
        '1': 'No',
        '2': 'Yes (under customer name)'
    },
    'foreign_worker': {
        '1': 'Yes',
        '2': 'No'
    },
    'credit_risk': {
        '0': 'Bad',
        '1': 'Good'
    }
}

metric_translation = {
    "amount": "Monto de crédito",
    "duration": "Duración en meses",
    "age": "Edad"
}

categorical_translation = {
    "status": "Estatus",
    "credit_history": "Historial de crédito",
    "purpose": "Propósito",
    "savings": "Ahorros",
    "employment_duration": "Duración del empleo",
    "installment_rate": "Tasa de cuota",
    "personal_status_sex": "Estado personal y sexo",
    "other_debtors": "Otros deudores",
    "present_residence": "Residencia actual",
    "property": "Propiedad",
    "other_installment_plans": "Otros planes de cuotas",
    "housing": "Vivienda",
    "number_credits": "Número de créditos",
    "job": "Trabajo",
    "people_liable": "Personas responsables",
    "telephone": "Teléfono",
    "foreign_worker": "Trabajador extranjero",
    "credit_risk": "Riesgo de crédito",
}

def plot_histogram_and_metrics(data, tab, column_name, title, xlabel, ylabel, min_bins=1, max_bins=100, default_bins=30):
    # Selección del número de bins, usando el rango dinámico
    bins = tab.slider(f'Selecciona el número de bins para {title}', min_value=min_bins, max_value=max_bins, value=default_bins, key=f"bins_{column_name}")

    # Dividir la columna en dos
    col1, col2 = tab.columns([3, 1], vertical_alignment="center")

    # Gráfico de histograma
    with col1:
        plt.figure(figsize=(10, 6))
        sns.histplot(data[column_name], bins=bins, kde=True, color='blue', alpha=0.6)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(axis='y')
        col1.pyplot(plt)

    # Cálculo de métricas descriptivas
    mean = data[column_name].mean()
    median = data[column_name].median()
    mode = data[column_name].mode()[0]  # mode() devuelve una serie, tomamos el primer valor
    std_dev = data[column_name].std()

    # Mostrar métricas descriptivas
    with col2:
        col2.markdown('### Métricas Descriptivas')
        col2.metric(label='Media', value=f"{mean:.2f}")
        col2.metric(label='Desviación Estándar', value=f"{std_dev:.2f}")
        col2.metric(label='Mediana', value=f"{median:.2f}")
        col2.metric(label='Moda', value=f"{mode:.2f}")

def generate_plot(data, tab, plot_type, x_axis, y_axis, hue=None, title='', xlabel='', ylabel=''):
    plt.figure(figsize=(12, 6))
    
    if plot_type == 'bar':
        sns.barplot(data, x=x_axis, y=y_axis, hue=hue, errorbar=None, palette='pastel')
    elif plot_type == 'box':
        sns.boxplot(data, x=x_axis, y=y_axis, hue=hue, palette='pastel')
    elif plot_type == 'violin':
        sns.violinplot(data, x=x_axis, y=y_axis, hue=hue, palette='pastel', width=0.9)
    elif plot_type == 'count':
        sns.countplot(data, x=x_axis, hue=hue, palette='pastel', width=0.9)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if hue:
        plt.legend(title=categorical_translation[hue])
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    
    tab.pyplot(plt)

def funcion_carrusel(data, selected_column1, selected_column2, hue_column):
    st.title("Descripción de variables categóricas cruzadas")

    # Crear DataFrames para las leyendas basándose en las categorías
    df_legends_column1 = pd.DataFrame(list(legends[selected_column1].items()), columns=['Categoría', 'Descripción'])
    df_legends_column2 = pd.DataFrame(list(legends[selected_column2].items()), columns=['Categoría', 'Descripción'])

    # Mostrar leyendas en columnas
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"### Leyenda de {categorical_translation[selected_column1]}")
        st.dataframe(df_legends_column1)

    with col2:
        st.markdown(f"### Leyenda de {categorical_translation[selected_column2]}")
        st.dataframe(df_legends_column2)

    # Seleccionar categorías de las columnas
    selected_category1 = st.selectbox(f"Selecciona una categoría de {categorical_translation[selected_column1]}", options=df_legends_column1['Categoría'], key=f'selected_category_{selected_column1}_1')
    selected_category2 = st.selectbox(f"Selecciona una categoría de {categorical_translation[selected_column2]}", options=df_legends_column2['Categoría'], key=f'selected_category_{selected_column2}_2')

    # Filtrar datos según la selección
    filtered_data = data[(data[selected_column1] == selected_category1) & (data[selected_column2] == selected_category2)]

    # Calcular media y desviación estándar
    if not filtered_data.empty:
        mean_value = filtered_data[hue_column].mean()
        std_value = filtered_data[hue_column].std()
        st.write(f"**Media de los valores:** {mean_value:.2f}")
        st.write(f"**Desviación Estándar de los valores:** {std_value:.2f}")
    else:
        st.write("No hay datos para las categorías seleccionadas.")

