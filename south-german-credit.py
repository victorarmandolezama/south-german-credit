import streamlit as st
import pandas as pd
import numpy as np
from main_functions import create_data, create_dummy_variables, chi_squared_test
import seaborn as sns
import matplotlib.pyplot as plt

south_german_credit = create_data()

st.markdown('## Descripción del conjunto de datos de créditos alemanes en un banco de 1994')

data_shape_col1, data_shape_col2 = st.columns(2, vertical_alignment="center")

data_shape_col1.metric(label="Registros", value=south_german_credit.shape[0])
data_shape_col2.metric(label="Caracterìsticas", value=south_german_credit.shape[1])

# st.bar_chart(south_german_credit, x='credit_history', y=['age'])

st.markdown('## Datos descriptivos')

def plot_histogram_and_metrics(tab, column_name, title, xlabel, ylabel, min_bins=1, max_bins=100, default_bins=30):
    # Selección del número de bins, usando el rango dinámico
    bins = tab.slider(f'Selecciona el número de bins para {title}', min_value=min_bins, max_value=max_bins, value=default_bins, key=f"bins_{column_name}")

    # Dividir la columna en dos
    col1, col2 = tab.columns([3, 1], vertical_alignment="center")

    # Gráfico de histograma
    with col1:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=south_german_credit[column_name], bins=bins, kde=True, color='blue', alpha=0.6)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(axis='y')
        col1.pyplot(plt)

    # Cálculo de métricas descriptivas
    mean = south_german_credit[column_name].mean()
    median = south_german_credit[column_name].median()
    mode = south_german_credit[column_name].mode()[0]  # mode() devuelve una serie, tomamos el primer valor
    std_dev = south_german_credit[column_name].std()

    # Mostrar métricas descriptivas
    with col2:
        col2.markdown('### Métricas Descriptivas')
        col2.metric(label='Media', value=f"{mean:.2f}")
        col2.metric(label='Mediana', value=f"{median:.2f}")
        col2.metric(label='Moda', value=f"{mode:.2f}")
        col2.metric(label='Desviación Estándar', value=f"{std_dev:.2f}")

# Crear las pestañas
descriptive_tab1, descriptive_tab2, descriptive_tab3 = st.tabs(["Monto de crédito solicitado", "Edad de los clientes", "Duración del crédito (en meses)"])

# Llamadas a la función para cada pestaña con parámetros de slider y etiquetas de ejes diferentes
plot_histogram_and_metrics(
    descriptive_tab1,
    "amount",
    'Histograma de Monto de crédito solicitado',
    xlabel='Valor del Crédito',
    ylabel='Densidad',
     min_bins=10, 
     max_bins=100, 
     default_bins=50,
)

plot_histogram_and_metrics(
    descriptive_tab2,
    "age",
    'Histograma de Edad de los clientes',
    xlabel='Edad (años)',
    ylabel='Densidad',
    min_bins=10, 
    max_bins=50, 
    default_bins=30
)

plot_histogram_and_metrics(
    descriptive_tab3,
    "duration",
    'Histograma de Duración del crédito (en meses)',
    xlabel='Duración (meses)',
    ylabel='Densidad',
    min_bins=5, 
    max_bins=30, 
    default_bins=15
)

import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Diccionario de traducción para las métricas
metric_translation = {
    "amount": "Monto de crédito",
    "duration": "Duración en meses",
    "age": "Edad"
}

# Diccionario de traducción para las variables categóricas
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

# Crear las pestañas
bar_plot_tab1, bar_plot_tab2, bar_plot_tab3 = st.tabs(['Gráfico de barras', 'Gráfico de caja', 'Gráfico de violín'])

# Función para generar gráficos
def generate_plot(tab, plot_type, x_axis, y_axis, hue=None, title='', xlabel='', ylabel=''):
    plt.figure(figsize=(12, 6))
    
    if plot_type == 'bar':
        sns.barplot(data=south_german_credit, x=x_axis, y=y_axis, hue=hue, errorbar=None, palette='pastel')
    elif plot_type == 'box':
        sns.boxplot(data=south_german_credit, x=x_axis, y=y_axis, hue=hue, palette='pastel')
    elif plot_type == 'violin':
        sns.violinplot(data=south_german_credit, x=x_axis, y=y_axis, hue=hue, palette='pastel', width=0.9)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if hue:
        plt.legend(title=hue)
    plt.xticks(rotation=45)
    plt.grid(axis='y')
    
    tab.pyplot(plt)

# Gráfico de barras
barplot_x_axis_option = bar_plot_tab1.selectbox(
    "Seleccione la categoría para el gráfico",
    list(categorical_translation.values()),  # Usar las traducciones
    0,
    key="barplot_x_axis_option"
)

barplot_y_axis_option = bar_plot_tab1.selectbox(
    "Seleccione la métrica a visualizar",
    list(metric_translation.values()),  # Usar las traducciones
    key="barplot_y_axis_option"
)

barplot_hue_option = bar_plot_tab1.selectbox(
    "Seleccione la variable para la segmentación",
    list(categorical_translation.values()),  # Usar las traducciones
    1,
    key="barplot_hue_option"
)

generate_plot(
    bar_plot_tab1,
    plot_type='bar',
    x_axis=list(categorical_translation.keys())[list(categorical_translation.values()).index(barplot_x_axis_option)],  # Buscar la llave
    y_axis=list(metric_translation.keys())[list(metric_translation.values()).index(barplot_y_axis_option)],  # Buscar la llave
    hue=list(categorical_translation.keys())[list(categorical_translation.values()).index(barplot_hue_option)],  # Buscar la llave
    title='Análisis de crédito: Gráfico de barras',
    xlabel='Categoría',
    ylabel=barplot_y_axis_option
)

# Gráfico de caja
box_plot_x_axis_option = bar_plot_tab2.selectbox(
    "Seleccione la categoría para el gráfico",
    list(categorical_translation.values()),  # Usar las traducciones
    0,
    key="box_plot_x_axis_option"
)

box_plot_y_axis_option = bar_plot_tab2.selectbox(
    "Seleccione la métrica a visualizar",
    list(metric_translation.values()),  # Usar las traducciones
    key="box_plot_y_axis_option"
)

boxplot_hue_option = bar_plot_tab2.selectbox(
    "Seleccione la variable para la segmentación",
    list(categorical_translation.values()),  # Usar las traducciones
    1,
    key="boxplot_hue_option"
)

generate_plot(
    bar_plot_tab2,
    plot_type='box',
    x_axis=list(categorical_translation.keys())[list(categorical_translation.values()).index(box_plot_x_axis_option)],  # Buscar la llave
    y_axis=list(metric_translation.keys())[list(metric_translation.values()).index(box_plot_y_axis_option)],  # Buscar la llave
    hue=list(categorical_translation.keys())[list(categorical_translation.values()).index(boxplot_hue_option)],  # Buscar la llave
    title='Análisis de crédito: Gráfico de caja',
    xlabel='Categoría',
    ylabel=box_plot_y_axis_option
)

# Gráfico de violín
violin_x_axis_option = bar_plot_tab3.selectbox(
    "Seleccione la categoría para el gráfico",
    list(categorical_translation.values()),  # Usar las traducciones
    0,
    key="violin_x_axis_option"
)

violin_y_axis_option = bar_plot_tab3.selectbox(
    "Seleccione la métrica a visualizar",
    list(metric_translation.values()),  # Usar las traducciones
    key="violin_y_axis_option"
)

violin_hue_option = bar_plot_tab3.selectbox(
    "Seleccione la variable para la segmentación",
    list(categorical_translation.values()),  # Usar las traducciones
    1,
    key="violin_hue_option"
)

generate_plot(
    bar_plot_tab3,
    plot_type='violin',
    x_axis=list(categorical_translation.keys())[list(categorical_translation.values()).index(violin_x_axis_option)],  # Buscar la llave
    y_axis=list(metric_translation.keys())[list(metric_translation.values()).index(violin_y_axis_option)],  # Buscar la llave
    hue=list(categorical_translation.keys())[list(categorical_translation.values()).index(violin_hue_option)],  # Buscar la llave
    title='Análisis de crédito: Gráfico de violín',
    xlabel='Categoría',
    ylabel=violin_y_axis_option
)

container = st.container(border=True)

dummy_status_data = create_dummy_variables(south_german_credit, 'status', 'credit_risk')
dummy_credit_history_data = create_dummy_variables(south_german_credit, 'credit_history', 'credit_risk')
dummy_purpose_data = create_dummy_variables(south_german_credit, 'purpose', 'credit_risk')
dummy_savings_data = create_dummy_variables(south_german_credit, 'savings', 'credit_risk')
dummy_personal_status_sex_data = create_dummy_variables(south_german_credit, 'personal_status_sex', 'credit_risk')
chi_squared_df = pd.DataFrame.from_dict(
    {
        **chi_squared_test(dummy_status_data, 'credit_risk_1'),
        **chi_squared_test(dummy_credit_history_data, 'credit_risk_1'),
        **chi_squared_test(dummy_purpose_data, 'credit_risk_1'),
        **chi_squared_test(dummy_savings_data, 'credit_risk_1'),
        **chi_squared_test(dummy_personal_status_sex_data, 'credit_risk_1'),

    }, 
    orient='index')

container.dataframe(chi_squared_df)
