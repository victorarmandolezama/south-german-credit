import streamlit as st
import pandas as pd
import numpy as np
from main_functions import create_data
import seaborn as sns
import matplotlib.pyplot as plt

south_german_credit = create_data()

st.markdown('## Descripción del conjunto de datos de créditos alemanes en un banco de 1994')

data_shape_col1, data_shape_col2 = st.columns(2, vertical_alignment="center")

data_shape_col1.metric(label="Registros", value=south_german_credit.shape[0])
data_shape_col2.metric(label="Caracterìsticas", value=south_german_credit.shape[1])

# st.bar_chart(south_german_credit, x='credit_history', y=['age'])

st.markdown('## Histograma')

column_name = st.selectbox('Selecciona una columna numérica:', ["amount", "age", "duration"],)

bins = st.slider('Selecciona el número de bins', min_value=1, max_value=100, value=30)

col1, col2 = st.columns([3, 1], vertical_alignment="center")

with col1:
    plt.figure(figsize=(10, 6))
    sns.histplot(data=south_german_credit[column_name], bins=bins, kde=True, color='blue', alpha=0.6)
    plt.title('Histograma de Datos Aleatorios')
    plt.xlabel('Valor')
    plt.ylabel('Densidad')
    plt.grid(axis='y')
    col1.pyplot(plt)

with col2:
    mean = south_german_credit[column_name].mean()
    median = south_german_credit[column_name].median()
    mode = south_german_credit[column_name].mode()[0]  # mode() devuelve una serie, tomamos el primer valor
    std_dev = south_german_credit[column_name].std()

    # Mostrar métricas descriptivas
    col2.markdown('### Métricas Descriptivas')
    col2.metric(label='Media', value=f"{mean:.2f}")
    col2.metric(label='Mediana', value=f"{median:.2f}")
    col2.metric(label='Moda', value=f"{mode:.2f}")
    col2.metric(label='Desviación Estándar', value=f"{std_dev:.2f}")

tab1, tab2, tab3 = st.tabs(['Grafico de barras', 'Grafico de caja', 'Grafico de violin'])

categorical_columns_tuple = (
        "status",
        "credit_history",
        "purpose",
        "savings",
        "employment_duration",
        "installment_rate",
        "personal_status_sex",
        "other_debtors",
        "present_residence",
        "property",
        "other_installment_plans",
        "housing",
        "number_credits",
        "job",
        "people_liable",
        "telephone",
        "foreign_worker",
        "credit_risk",
    )

barplot_x_axis_option = tab1.selectbox(
    "Variables en el eje x",
    categorical_columns_tuple,
    0,
    key="barplot_x_axis_option",
)

barplot_y_axis_option = tab1.selectbox(
    "Variables en el eje y",
    ("amount", "age", "duration"),
    key="barplot_y_axis_option",
)

barplot_hue_option = tab1.selectbox(
    "Variables en el hue",
    categorical_columns_tuple,
    1,
    key="barplot_hue_option",
)

plt.figure(figsize=(12, 6))
sns.barplot(data=south_german_credit, x=barplot_x_axis_option, y=barplot_y_axis_option, hue=barplot_hue_option, errorbar=None, palette='pastel')
plt.title('Promedio de monto de crédito solicitado por la combinación de estatus e historial de crédito (full data)')
plt.xlabel('Historial de crédito')
plt.ylabel('Promedio de monto solicitado')
plt.legend(title='Estatus')
plt.xticks(rotation=45)
plt.grid(axis='y')


tab1.pyplot(plt)

box_plot_x_axis_option = tab2.selectbox(
    "Variables en el eje x",
    categorical_columns_tuple,
    0,
    key="box_plot_x_axis_option",
)

box_plot_y_axis_option = tab2.selectbox(
    "Variables en el eje y",
    ("amount", "age", "duration"),
    key="box_plot_y_axis_option",
)

plt.figure(figsize=(12, 6))
sns.boxplot(data=south_german_credit, x=box_plot_x_axis_option, y=box_plot_y_axis_option, palette='pastel')
plt.title('Promedio de monto de crédito solicitado por la combinación de estatus e historial de crédito (full data)')
plt.xlabel('Historial de crédito')
plt.ylabel('Promedio de monto solicitado')

tab2.pyplot(plt)

violin_x_axis_option = tab3.selectbox(
    "Variables en el eje x",
    categorical_columns_tuple,
    0,
    key="violin_x_axis_option",
)

violin_y_axis_option = tab3.selectbox(
    "Variables en el eje y",
    ("amount", "age", "duration"),
    key="violin_y_axis_option",
)

violin_hue_option = tab3.selectbox(
    "Variables en el hue",
    categorical_columns_tuple,
    1,
    key="violin_hue_option",
)

plt.figure(figsize=(15, 10))
sns.violinplot(data=south_german_credit, x=violin_x_axis_option, y=violin_y_axis_option, hue=violin_hue_option, palette='pastel', width=0.9)
plt.title('Gráfico de violín de historial de crédito por monto de crédito solicitado y estatus')
plt.xlabel('Historial de crédito')
plt.ylabel('Promedio de monto solicitado')
plt.legend(title='Estatus')

tab3.pyplot(plt)