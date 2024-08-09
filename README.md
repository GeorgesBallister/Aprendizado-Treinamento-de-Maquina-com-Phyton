# Aprendizado Treinamento de Maquina com Phyton

**Tratamento de dados e Treinamento de Aprendizado de Máquina para Identificação de Doenças com Base em** **Hemograma**

### Integrantes do Projeto

| Nome | Email |
| --- | --- |
| Georges Ballister de Oliveira | georgesballister.profissional@gmail.com |
| Daniel de Melo Arantes Cabral | 12dancabral@gmail.com |
| Douglas Numeriano Marinho Falcão | douglasnumeriano11@hotmail.com |
| Davi Albnes Vasconcellos Pires | davipires03@gmail.com |
| Bruno Vinícius Araújo de Mesquita | brunovinicius2002@hotmail.com |
| Gabriel Sobral Santos Silva | gabrielprofessional12@gmail.com |

***Abstract. The aim of this study is to demonstrate the data analysis process conducted on the "Multiple Disease Prediction" dataset using Python. The primary focus was to train machine learning models to identify, based on a blood test including information on 25 blood proteins, two major blood-related conditions: diabetes and anemia, as well as to distinguish other health conditions.***

***Resumo. Este estudo visa demonstrar o processo de análise de dados realizado no dataset "Multiple Disease Prediction", utilizando a linguagem Python. O objetivo principal é treinar um modelo de aprendizado de máquina capaz de identificar, a partir de um hemograma com 25 proteínas sanguíneas, duas doenças sanguíneas principais: diabetes e anemia, além de detectar outras possíveis condições não relacionadas a essas doenças.***

# **Introdução**

O crescente número de doenças que afetam a sociedade humana é uma preocupação significativa. A identificação precoce dessas doenças é essencial para o tratamento eficaz e a melhoria da qualidade de vida. Um hemograma, acompanhado pela análise de proteínas totais e suas frações, pode fornecer uma avaliação valiosa do estado nutricional e a detecção de diversas doenças hepáticas, renais e hematológicas.

Com base nesse contexto, este estudo realizou uma análise de dados utilizando um conjunto de dados que simula milhares de hemogramas, contendo informações sobre 25 proteínas sanguíneas. O objetivo foi desenvolver um diagnóstico que interprete os valores presentes em cada coluna do registro, focando na identificação de diabetes, anemia e outras condições de saúde.

Para alcançar esse objetivo, foram treinados dois modelos de inteligência artificial distintos projetados para identificar padrões nos valores medianos dessas proteínas, levando em consideração o desvio padrão. Este projeto busca avançar no campo da análise de dados médicos e contribuir para a melhoria das técnicas de diagnóstico precoce por meio do uso de algoritmos de aprendizado de máquina.

# 1. Contextualização do Problema Encontrado:

## **1.1 Conjunto de Dados**

O estudo utiliza o conjunto de dados ["Multiple Disease Prediction"](https://www.kaggle.com/datasets/ehababoelnaga/multiple-disease-prediction) extraído do site Kaggle, composto por informações de amostras de sangue utilizadas para prever diversas doenças. O objetivo é realizar uma análise detalhada dos dados para diagnosticar doenças como diabetes e anemia, empregando dois modelos de treinamento distintos.

## **1.2 Dados Encontrados**

O dataset é normalizado no intervalo de 0 a 1 para facilitar a análise. A seguir, apresentamos as principais características e seus respectivos intervalos de referência:

| Parâmetro | Intervalo de Referência | Unidade |
| --- | --- | --- |
| Glicose | 70-140 | mg/dL |
| Colesterol | 125-200 | mg/dL |
| Hemoglobina | 13,5-17,5 | g/dL |
| Plaquetas | 150.000-450.000 | por microlitro de sangue |
| Leucócitos (glóbulos brancos) | 4.000-11.000 | por milímetro cúbico de sangue |
| Eritrócitos (glóbulos vermelhos) | 4,2-5,4 | milhões por microlitro de sangue |
| Hematócrito | 38-52 | % |
| Volume Corpuscular Médio (MCV) | 80-100 | femtolitros |
| Hemoglobina Corpuscular Média (MCH) | 27-33 | picogramas |
| Concentração de Hemoglobina Corpuscular Média (MCHC) | 32-36 | g/dL |
| Insulina | 5-25 | microU/mL |
| Troponina | 0-0,04 | ng/mL |
| Índice de Massa Corporal (IMC) | 18,5-24,9 | kg/m² |
| Pressão Arterial Sistólica | 90-120 | mmHg |
| Pressão Arterial Diastólica | 60-80 | mmHg |
| Triglicerídeos | 50-150 | mg/dL |
| Hemoglobina Glicada (HbA1c) | 4-6 | % |
| Colesterol LDL | 70-130 | mg/dL |
| Colesterol HDL | 40-60 | mg/dL |
| ALT (Alanina Aminotransferase) | 10-40 | U/L |
| AST (Aspartato Aminotransferase) | 10-40 | U/L |
| Frequência Cardíaca | 60-100 | batimentos por minuto |
| Creatinina | 0,6-1,2 | mg/dL |
| Proteína C-reativa | 0-3 | mg/L |

Esses parâmetros são fundamentais para a detecção de diversas condições de saúde, como diabetes, anemia, dislipidemias e doenças cardíacas, entre outras. Ao analisar esses dados, podemos identificar padrões e anomalias que auxiliam no diagnóstico e na proposição de intervenções preventivas ou terapêuticas.

# **2. Ferramentas e Bibliotecas**

### Ferramentas Utilizadas

| Ferramenta | Descrição |
| --- | --- |
| Google Colab | Ambiente de desenvolvimento integrado (IDE) na nuvem para escrever e executar código Python. |
| Google Drive | Ferramenta de armazenamento na nuvem, permitindo colaboração e sincronização de arquivos. |

### Bibliotecas Python Utilizadas

| Biblioteca | Descrição |
| --- | --- |
| Pandas | Biblioteca poderosa para acessar e manipular tabelas. |
| Random e Numpy | Bibliotecas para funções matemáticas e manipulação de arrays e matrizes. |
| Matplotlib e Seaborn | Bibliotecas de visualização e criação de gráficos. |
| Sklearn (Scikit-learn) | Biblioteca de aprendizado de máquina em Python com várias ferramentas para modelagem preditiva. |
| Sklearn.preprocessing.MinMaxScaler | Ferramenta para normalizar dados. |
| Sklearn.model_selection.train_test_split | Ferramenta para dividir um conjunto de dados em conjuntos de treinamento e teste. |
| Sklearn.tree.DecisionTreeClassifier | Algoritmo de árvore de decisão para classificação e regressão. |
| Sklearn.ensemble.RandomForestClassifier | Modelo de florestas aleatórias para reduzir o overfitting. |
| Sklearn.svm.SVC | Modelo de vetores de suporte para espaços de alta dimensão. |
| Sklearn.metrics.accuracy_score | Ferramenta para calcular a acurácia do modelo de classificação. |
| Sklearn.metrics.classification_report e Sklearn.metrics.confusion_matrix | Ferramentas para avaliar o desempenho do modelo. |
| Sklearn.impute.SimpleImputer | Ferramenta para lidar com dados faltantes. |
| Sklearn.preprocessing.LabelEncoder e Sklearn.preprocessing.OneHotEncoder | Ferramentas para converter variáveis categóricas em numéricas. |
| Imblearn.over_sampling.RandomOverSampler | Ferramenta para lidar com conjuntos de dados desbalanceados. |
| Collections.Counter | Biblioteca de contagem de dados para analisar a distribuição das classes. |

# 3. Como Executar o Projeto

Caso você queira executar o projeto, siga as seguintes etapas:

1. O projeto foi estruturado para que as bibliotecas sejam importadas antes do CSV. Ao abrir o projeto em sua máquina ou no Google Colab, preste atenção ao segundo e terceiro blocos de código:

![Aprendizado%20Treinamento%20de%20Maquina%20com%20Phyton%20c6711311974346b6b17ef9c0f2bc4bfc/Untitled.png](Aprendizado%20Treinamento%20de%20Maquina%20com%20Phyton%20c6711311974346b6b17ef9c0f2bc4bfc/Untitled.png)

1. Este código deve ser alterado para que você adicione o caminho correto onde o CSV está localizado.
    1. Caso você esteja utilizando o Colab, após fazer o upload do projeto no Drive, copie o caminho da pasta e adicione-o no local apropriado dentro da função `read_csv("/content....")`.
    2. Caso você esteja utilizando sua máquina, exclua a segunda linha e substitua o caminho dentro da função `read_csv("/content....")`.

<aside>
💡 Caso algum problema persista, abra uma Issue no repositório do projeto.

</aside>

# **Bibliografia**

| Autor | Título | Data de Publicação | Disponível em | Acessado em |
| --- | --- | --- | --- | --- |
| Blog EngDB | Análises Preditivas | 02/08/2023 | [https://blog.engdb.com.br/analises-preditivas/] | 05/05/2024 |
| Escola DNC | Como Identificar e Tratar Outliers em Data Science | 14/05/2024 | [https://www.escoladnc.com.br/blog/como-identificar-e-tratar-outliers-em-data-science/] | 28/05/2024 |
| Awari | Tratamento de Dados com Python | 31/07/2023 | [https://awari.com.br/tratamento-de-dados-com-python-o-tratamento-de-dados-com-python/] | 20/05/2024 |
| Ale George Lustosa | Métodos de Tratamento para Dados Categóricos em Python | 28/12/2018 | [https://medium.com/@alegeorgelustosa/m%C3%A9todos-de-tratamento-para-dados-categ%C3%B3ricos-em-python-a66f910215c7] | 20/05/2024 |
| http://aquare.la/ | O que São Outliers e Como Tratá-los em uma Análise de Dados | 25/09/2017 | [https://aquare.la/o-que-sao-outliers-e-como-trata-los-em-uma-analise-de-dados/] | 25/05/2024 |
| Heitor Catunda, Hashtag Treinamentos | Datasets Desbalanceados Ciência de Dados | 31/10/2022 | [https://www.hashtagtreinamentos.com/datasets-desbalanceados-ciencia-dados] | 22/05/2024 |
| Mirla Costa, leticiapyres, Alura | Machine Learning | 19/01/2024 | [https://www.alura.com.br/artigos/machine-learning] | 12/05/2024 |
| Carlos Melo, http://sigmoidal.ai/ | Como Lidar com Dados Desbalanceados | 24/12/2019 | [https://sigmoidal.ai/como-lidar-com-dados-desbalanceados/] | 05/06/2024 |
| Francisco Foz, Medium | Como Tratar Outliers sem Excluí-los | 14/03/2022 | [https://franciscofoz.medium.com/como-tratar-outliers-sem-exclu%C3%AD-los-19dd5c1ba3e6] | 10/06/2024 |
| Ehab Abouelnaga, Kaggle | Multiple Disease Prediction | 03/03/2024 | [https://www.kaggle.com/datasets/ehababoelnaga/multiple-disease-prediction] | 10/06/2024 |
| Daniele Santiago, Medium | Aprenda a Balancear seus Dados com Undersampling e Oversampling em Python | 05/06/2023 | [https://medium.com/@daniele.santiago/aprenda-a-balancear-seus-dados-com-undersampling-e-oversampling-em-python-6fd87095d717] | 11/05/2024 |
