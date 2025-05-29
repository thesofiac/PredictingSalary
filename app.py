import joblib
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from category_encoders import TargetEncoder
from sklearn.preprocessing import StandardScaler
import re

def filtrar_palavras(texto):
    palavras = [p.strip().lower() for p in re.split('[,\\s]+', texto) if p]
    relevantes = {p for p in palavras if p in top_words}
    return ', '.join(sorted(relevantes))


def extrair_media(s):
    numeros = re.findall(r'\d[\d.,]*', s)  # captura números com ponto ou vírgula
    numeros = [float(n.replace('.', '').replace(',', '.')) for n in numeros]
    if len(numeros) == 0:
        return None
    elif len(numeros) == 1:
        return numeros[0]
    else:
        return sum(numeros) / len(numeros)


def treat_columns(df):
    # Dropar colunas pouco informativas
    df = df.drop('Unnamed: 0', axis=1)
    df = df.drop('Company Name', axis=1)
    df = df.drop('Founded', axis=1)
    df = df.drop('Sector', axis=1)
    df = df.drop('Competitors', axis=1)
    df = df.drop('Rating', axis=1)
    df = df.drop('Revenue', axis=1)
    df = df.drop('Headquarters', axis=1)
    df = df.drop('Type of ownership', axis=1)
    df = df.drop('Size', axis=1)

    top_words = ['scientist', 'data', 'engineer', 'senior', 'analyst', 'sr', 'analytics']

    df['Filtered Titles'] = df['Job Title'].apply(filtrar_palavras)
    df = df.drop('Job Title', axis=1)

    top_words = ['data', 'experience', 'work', 'business', 'development', 'team', 'skills']

    df['Filtered Description'] = df['Job Description'].apply(filtrar_palavras)
    df = df.drop('Job Description', axis=1)

    df['salario_medio'] = df['Salary Estimate'].apply(extrair_media)
    df = df.drop('Salary Estimate', axis=1)

    X = df.drop('salario_medio', axis=1)
    y = df['salario_medio']

    # Tratar salario_medio
    limite_inferior = -81.9375
    limite_superior = 231.5625

    # Aplica o clip
    y = y.clip(lower=limite_inferior, upper=limite_superior)

    # Calcula a média ignorando os 1s
    media_sem_1 = y.loc[y != 1].mean() # 1 = Null
    y = y.replace(1, media_sem_1)

    # Tratar Location, Headquarters, Type of ownership, Industry e Filtered Titles
    X['Location'] = X['Location'].str.split(',').str[1]

    encoder = joblib.load('target_encoder.pkl')
    X_encoded = encoder.transform(X, y)

    return X_encoded, y

# Carregar modelo e dados
df = pd.read_csv('glassdoor_jobs.csv')
model = joblib.load('PredictSalary_model.pkl')

# Processar
X, y = treat_columns(df)
y_pred = model.predict(X)

st.set_page_config(layout="wide")
st.title("Classificador Binário")

menu = st.sidebar.selectbox("Escolha uma opção", [
    "Entenda os dados",
    "Calcule o salário a ser ofertado"
])

states_options = ['AK', 'AL', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'MI', 'MN', 'MO', 'NC', 'NE', 'NJ', 'NM', 'NY', 'OH', 'OR', 'PA', 'PR', 'RI', 'SC', 'TN', 'TX', 'UT', 'VA', 'WA', 'WI']

dic_sizes = {
        'De 1 a 50 funcionários' : '1 to 50 employees',
        'De 50 a 200 funcionários' : '51 to 200 employees',
        'De 200 a 500 funcionários' : '201 to 500 employees',
        'De 500 a 1000 funcionários' : '501 to 1000 employees',
        'De 1000 a 5000 funcionários' : '1001 to 5000 employees',
        'De 5000 a 10000 funcionários' : '5001 to 10000 employees'
        }


dic_industry = {
        'Alimentação' : 'Food & Beverage Manufacturing',
        'Bancos e Finanças' : 'Banks & Credit Unions',
        'Consultoria' : 'Consulting',
        'Energia' : 'Energy',
        'Farmacêutica' : 'Biotech & Pharmaceuticals',
        'Manufatura' : 'Consumer Products Manufacturing',
        'Marketing' : 'Advertising & Marketing',
        'Pesquisa e Desenvolvimento' : 'Research & Development',
        'Prestação de Serviços' : 'Staffing & Outsourcing',
        'Saúde' : 'Health Care Services & Hospitals',
        'Seguros' : 'Insurance Carriers',
        'Tecnologia' : 'IT Services',
        'Universidade' : 'Colleges & Universities',
        }

sizes_options = list(dic_sizes.keys())
industry_options = list(dic_industry.keys())


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

['Unnamed: 0', 'Job Title', 'Salary Estimate', 'Job Description',
       'Rating', 'Company Name', 'Location', 'Headquarters', 'Size', 'Founded',
       'Type of ownership', 'Industry', 'Sector', 'Revenue', 'Competitors']
    
    f1 = '0'
    f2 = st.text_area("Título da Vaga")
    f3 = st.text_area("Descrição da Vaga")
    f4 = st.number_input("Avaliação da Empresa", value=0.0)
    f5 = st.text_area("Nome da Empresa")
    f6 = st.selectbox("Estado da Empresa", states_options) 
    f7 = '0'
    f8 = st.selectbox("Tamanho da Empresa", sizes_options)
    f9 = '0'
    f10 = '0'
    f11 = st.selectbox("Área da Empresa", industry_options)
    f12 = '0'
    f13 = '0'
    f14 = '0'

    if st.button("Prever"):
        input_df = pd.DataFrame([[str(f1), str(f2), str(f3), float(f4), str(f5), str(f6), str(f7), dic_sizes[f8], str(f9), str(f10), dic_industry[f11], str(f12), str(f13), str(f14)]], columns=['Unnamed: 0', 'Job Title', 'Salary Estimate', 'Job Description', 'Rating', 'Company Name', 'Location', 'Headquarters', 'Size', 'Founded', 'Type of ownership', 'Industry', 'Sector', 'Revenue', 'Competitors'])
        X, y = treat_columns(input_df)
        y_pred = model.predict(X)
        st.write(f"O salário estimado para essa vaga é: ${y_pred[0]:.0f}")
