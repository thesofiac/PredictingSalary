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
    
elif menu == "Calcule o salário a ser ofertado":
    st.subheader("Calcule o salário a ser ofertado")
