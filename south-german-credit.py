import streamlit as st
import pandas as pd
import numpy as np
from main_functions import create_data, create_dummy_variables, chi_squared_test
import seaborn as sns
import matplotlib.pyplot as plt
from streamlit_utils import plot_histogram_and_metrics, generate_plot, funcion_carrusel, categorical_translation, metric_translation

south_german_credit = create_data()

st.markdown('## Descripción del conjunto de datos de créditos alemanes en un banco de 1994')

data_shape_col1, data_shape_col2 = st.columns(2, vertical_alignment="center")

data_shape_col1.metric(label="Registros", value=south_german_credit.shape[0])
data_shape_col2.metric(label="Caracterìsticas", value=south_german_credit.shape[1])

st.divider()

# st.bar_chart(south_german_credit, x='credit_history', y=['age'])

st.markdown('## Datos descriptivos')

st.divider()

st.markdown('### Histogramas')

# Crear las pestañas
descriptive_tab1, descriptive_tab2, descriptive_tab3 = st.tabs(["Monto de crédito solicitado", "Edad de los clientes", "Duración del crédito (en meses)"])

# Llamadas a la función para cada pestaña con parámetros de slider y etiquetas de ejes diferentes
plot_histogram_and_metrics(
    south_german_credit,
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
    south_german_credit,
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
    south_german_credit,
    descriptive_tab3,
    "duration",
    'Histograma de Duración del crédito (en meses)',
    xlabel='Duración (meses)',
    ylabel='Densidad',
    min_bins=5, 
    max_bins=30, 
    default_bins=15
)

st.divider()

st.markdown('### Barras, cajas y violines')

x_axis_option = st.selectbox(
    "Seleccione la categoría para el gráfico",
    list(categorical_translation.values()),  # Usar las traducciones
    0,
    key="x_axis_option"
)

y_axis_option = st.selectbox(
    "Seleccione la métrica a visualizar",
    list(metric_translation.values()),  # Usar las traducciones
    key="y_axis_option"
)

hue_option = st.selectbox(
    "Seleccione la variable para la segmentación",
    list(categorical_translation.values()),  # Usar las traducciones
    1,
    key="hue_option"
)

# Crear las pestañas
bar_plot_tab1, bar_plot_tab2, bar_plot_tab3, bar_plot_tab4 = st.tabs(['Gráfico de barras', 'Gráfico de caja', 'Gráfico de violín', 'Conteos'])

# Gráfico de barras
x_axis_name=list(categorical_translation.keys())[list(categorical_translation.values()).index(x_axis_option)]
y_axis_name=list(metric_translation.keys())[list(metric_translation.values()).index(y_axis_option)]
hue_name=list(categorical_translation.keys())[list(categorical_translation.values()).index(hue_option)]

generate_plot(
    south_german_credit,
    bar_plot_tab1,
    plot_type='bar',
    x_axis=x_axis_name,
    y_axis=y_axis_name,
    hue=hue_name,
    title='Análisis de crédito: Gráfico de barras',
    xlabel=f'Categoría: {categorical_translation[x_axis_name]}',
    ylabel=y_axis_option,
)

generate_plot(
    south_german_credit,
    bar_plot_tab2,
    plot_type='box',
    x_axis=x_axis_name,
    y_axis=y_axis_name,
    hue=hue_name,
    title='Análisis de crédito: Gráfico de caja',
    xlabel=f'Categoría: {categorical_translation[x_axis_name]}',
    ylabel=y_axis_option
)

generate_plot(
    south_german_credit,
    bar_plot_tab3,
    plot_type='violin',
    x_axis=x_axis_name,
    y_axis=y_axis_name,
    hue=hue_name,
    title='Análisis de crédito: Gráfico de violín',
    xlabel=f'Categoría: {categorical_translation[x_axis_name]}',
    ylabel=y_axis_option
)

generate_plot(
    south_german_credit,
    bar_plot_tab4,
    plot_type='count',
    x_axis=x_axis_name,
    y_axis=y_axis_name,
    hue=hue_name,
    title='Análisis de crédito: Conteos',
    xlabel=f'Categoría: {categorical_translation[x_axis_name]}',
    ylabel='Conteos'
)

funcion_carrusel(data=south_german_credit, selected_column1=x_axis_name, selected_column2=hue_name, hue_column=y_axis_name)

st.divider()

st.markdown('### Pruebas Xi cuadrada')

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
