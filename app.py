import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from category_encoders import TargetEncoder
from sklearn.preprocessing import StandardScaler
import re

menu = st.sidebar.selectbox("Escolha uma opção", [
    "Entenda os dados",
    "Calcule o salário a ser ofertado"
])

if menu == "Entenda os dados":
    st.subheader("Entenda os dados")
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("<div style='text-align: justify'><h5>Os dados são referentes a vagas da área de dados e empresas avaliadas no <i>site</i> do Glassdoor. Para cada uma das vagas, é possível obter seu título, descrição, salário <span style='color:#E07A5F;'>estimado</span>, nome e setor da empresa, assim como sua avaliação. Desenvolver um modelo de regressão ou classificação que ajude as empresas a estimar o salário que deve ser oferecido em seus anúncios de vagas é interessante, tanto para que não paguem muito a mais que a média do mercado em salários, quanto para que não percam talentos oferecendo salários muito baixos.</h5></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("<div style='text-align: justify'><h5>Assim, foi determinado um modelo de regressão XXX, que mostrou RMSE médio de XXX para dados nunca vistos pelo modelo (valor obtido do croos-validation com cv=XXX para o conjunto geral dos dados). Por conta do <i>target</i> se tratar de salários estimados, na forma de <span style='color:#E07A5F;'>ranges</span> e não valores exatos, já era de se esperar grande variação e maior dificuldade para traçar um modelo que acertasse todas as previsões. Ainda assim, é de se considerar como bom o desempenho do modelo, já que servirá apenas de ponto de partida para negociações salariais e não como valor final.</h5></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("<h5>O salário médio ofertado é <br>$78K<br> anuais.</h5>", unsafe_allow_html=True)

    with col2:
        st.markdown("<h5>Entre os profissionais da área de dados, os <br>Cientistas de Dados são os que recebem melhor, <br>seguidos dos Engenheiros de Dados e Analistas de Dados.</h5>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    col3, col4 = st.columns([1.5, 1])

    with col3:
        st.markdown("<h5>Candidatos que <span style='color:#E07A5F;'>saíram do <br> seu último emprego há mais tempo</span>, <br> têm <span style='color:#E07A5F;'><b>menor chance</b></span> de trocar de emprego</h5>", unsafe_allow_html=True)

        # Dados
        x = ['Cientista de Dados', 'Engenheiro de Dados', 'Analista de Dados']
        x_range = range(len(x))

        y_todos = [92, 83, 58]
        y_nao_senior = [86, 80, 53]
        y_senior = [114, 99, 84]

        # Cores personalizadas
        cor_todos = '#f15050ff'
        cor_nao_senior = '#f77c7c'
        cor_senior = '#bb3737ff'

        # Criando o gráfico de linhas
        fig, ax = plt.subplots(figsize=(10, 6))

        # Linhas verticais mostrando a diferença
        for i in range(len(x)):
            ax.vlines(x=x_range[i], ymin=y_senior[i], ymax=y_nao_senior[i], color='gray', linestyle='--', linewidth=1)

        # Linhas
        ax.plot(x_range, y_nao_senior, marker='o', label='Não Sênior', color=cor_nao_senior)
        ax.plot(x_range, y_senior, marker='o', label='Sênior', color=cor_senior)

        # Labels nas linhas
        ax.text(x_range[-1] + 0.05, y_nao_senior[-1], 'Não sênior', va='center', ha='left', fontsize=10, color=cor_nao_senior)
        ax.text(x_range[-1] + 0.05, y_senior[-1], 'Sênior', va='center', ha='left', fontsize=10, color=cor_senior)

        for i in x_range:
            y_diff = y_senior[i] - y_nao_senior[i]
            y_medio = (y_senior[i] + y_nao_senior[i]) / 2
            ax.text(i + 0.03, y_medio, f'${y_diff}k', va='center', ha='left', fontsize=10, color='gray')

        # Ajustes de eixos
        ax.set_ylabel('Salário Anual')
        ax.set_xticks(x_range)
        ax.set_xticklabels(x)
        ax.set_xlim(-0.05, len(x) - 0.7)
        ax.set_ylim(50, 120)
        ax.set_yticks([])

        # Mostra o gráfico no Streamlit
        st.pyplot(fig)

    with col4:
        st.markdown("<br><br><br><br><br><br><br><br><br><br>", unsafe_allow_html=True)
        
        st.markdown("<h5>Os analistas de dados são os que têm <span style='color:#E07A5F;'>maior aumento de salário</span> ao se tornarem sêniors</h5>", unsafe_allow_html=True)
        
elif menu == "Calcule o salário a ser ofertado":
    st.subheader("Calcule o salário a ser ofertado")
