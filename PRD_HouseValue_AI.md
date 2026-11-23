# PRD — HouseValue AI
## Sistema Inteligente de Estimativa de Preços de Imóveis

**Versão:** 2.0 — revisão corrigida  
**Status:** Especificação para implementação  
**Natureza:** Projeto acadêmico de Machine Learning e desenvolvimento de software  
**Disciplina:** TK0231 — Sistemas de Informação Gerencial  
**Instituição:** Universidade Federal do Ceará (UFC)  
**Metodologia:** CRISP-DM  
**Idioma da aplicação e documentação:** Português (Brasil), preservando nomes originais das colunas e termos técnicos quando necessário.

---

# 1. Resumo executivo

O **HouseValue AI** será uma aplicação web acadêmica que estima o preço de venda de imóveis residenciais com base em características físicas, estruturais e de localização. O projeto utilizará a base **House Prices: Advanced Regression Techniques**, indicada na categoria de regressão do documento de opções de datasets disponibilizado na disciplina.

A variável-alvo é `SalePrice`, que representa o preço pelo qual cada imóvel foi vendido. O conjunto descreve imóveis de Ames, Iowa, nos Estados Unidos. Portanto, os valores históricos estão em dólares americanos (USD), e o modelo não deverá ser apresentado como ferramenta de avaliação de imóveis brasileiros ou de preços atuais.

A solução será desenvolvida com:

- **Python, Pandas e NumPy** para leitura, inspeção e preparação dos dados;
- **Scikit-learn** para pipelines, treinamento, validação e avaliação dos modelos;
- **FastAPI** para disponibilizar previsões e indicadores por API;
- **HTML, CSS e JavaScript** para a interface web;
- **Plotly.js** para gráficos interativos;
- **Joblib** para persistência do pipeline treinado.

O sistema deverá demonstrar um fluxo completo de Machine Learning, da auditoria da base à disponibilização de previsões em uma interface. A implementação deverá ser reproduzível, documentada e metodologicamente correta.

**Correção de escopo:** este projeto é de avaliação de imóveis. Não utilizar dados de automóveis, Tabela FIPE, API FIPE ou qualquer módulo relacionado a veículos.

# 2. Origem e justificativa da escolha do dataset

## 2.1. Fonte indicada pela disciplina

O documento `Bases de dados - Projetos de ML.docx`, disponibilizado na disciplina, apresenta na seção **Regressão** a opção:

- **Dataset:** House Prices: Advanced Regression Techniques.
- **Descrição no documento:** contém características de casas em Iowa, EUA, e o preço pelo qual foram vendidas.
- **Link indicado:** https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques

A escolha é adequada ao objetivo de regressão porque a base contém uma variável-alvo numérica de preço de venda, chamada `SalePrice`.

## 2.2. Fonte de dados de referência

Fonte de distribuição do dataset: Kaggle — House Prices: Advanced Regression Techniques.

- Página da competição: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview
- Página dos dados: https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data

A descrição oficial do Kaggle identifica `SalePrice` como o preço de venda do imóvel e descreve 79 variáveis explicativas sobre imóveis em Ames, Iowa. A página de dados descreve os arquivos padrão `train.csv`, `test.csv`, `data_description.txt` e `sample_submission.csv`.

**Atenção:** os arquivos efetivamente encontrados no ZIP do projeto são a fonte operacional da implementação. O agente deverá inspecioná-los e registrar o conteúdo real. Quantidades de linhas, colunas, nomes de arquivos e tipos de dados deverão ser confirmados localmente; não assumir que o ZIP contém exatamente a estrutura padrão se o conteúdo for diferente.

## 2.3. Limites de interpretação

- A base representa imóveis vendidos em Ames, Iowa, e não o mercado imobiliário brasileiro.
- O preço-alvo está em USD, não em reais.
- Os registros são históricos; não representam automaticamente preços atuais.
- A estimativa produzida é um exercício acadêmico baseado na distribuição da base, não um laudo de avaliação imobiliária.
- Não converter preços para BRL como se isso transformasse o modelo em avaliador do mercado brasileiro.
- A interface deverá informar de forma visível a localização, a moeda e a natureza histórica dos dados.

# 3. Visão do produto

## 3.1. Problema

O preço de venda de um imóvel depende de múltiplos fatores, como área do terreno, área construída, qualidade geral, idade, estado de conservação, garagem, localização e características internas. O objetivo é investigar como essas características se relacionam com o valor de venda e se modelos de regressão conseguem estimar esse valor com erro mensurável.

## 3.2. Proposta de valor

Transformar uma base educacional real em uma aplicação funcional que permita:

1. compreender os dados e suas limitações;
2. comparar algoritmos de regressão sob o mesmo protocolo;
3. receber uma estimativa de preço a partir de características de um imóvel;
4. visualizar fatores associados ao preço e os resultados dos experimentos;
5. demonstrar, de ponta a ponta, um projeto de Machine Learning com CRISP-DM.

## 3.3. Objetivo geral

Desenvolver e avaliar um sistema de regressão capaz de estimar o preço histórico de venda de imóveis da base House Prices, utilizando suas características como variáveis preditoras e disponibilizando o modelo final em uma aplicação web.

## 3.4. Objetivos específicos

- Localizar e auditar o ZIP já disponibilizado na pasta de trabalho do projeto.
- Confirmar a presença e a estrutura dos arquivos da base House Prices.
- Entender o significado das colunas usando `data_description.txt`, quando presente.
- Definir `SalePrice` como variável-alvo, caso confirmada no conjunto rotulado.
- Realizar análise exploratória e documentar problemas de qualidade.
- Preparar dados numéricos e categóricos sem vazamento de informação.
- Treinar modelos de referência e modelos de regressão.
- Comparar os modelos por validação cruzada e conjunto de teste interno.
- Selecionar e persistir um pipeline completo de pré-processamento e previsão.
- Disponibilizar uma API FastAPI e uma interface web em JavaScript.
- Produzir documentação metodológica e roteiro de apresentação.

## 3.5. Pergunta de pesquisa

Em que medida características físicas, estruturais e de localização de imóveis de Ames, Iowa, permitem estimar seus preços históricos de venda por meio de algoritmos de regressão?

## 3.6. Hipóteses a investigar

- H1: as características dos imóveis apresentam associação com o preço de venda.
- H2: modelos não lineares podem capturar relações que um modelo linear simples não representa adequadamente.
- H3: diferentes algoritmos apresentarão desempenhos distintos quando avaliados nos mesmos dados.
- H4: é possível disponibilizar o modelo selecionado em uma aplicação interativa sem alterar a lógica de pré-processamento utilizada no treinamento.

As hipóteses são perguntas de investigação. Não apresentar nenhuma delas como resultado antes da execução dos experimentos.

# 4. Usuários e cenários de uso

## 4.1. Usuário principal

Professor e estudantes que precisam compreender o problema, visualizar as análises e acompanhar uma demonstração funcional do projeto.

## 4.2. Usuário demonstrador

A pessoa que apresenta o projeto deverá conseguir preencher um formulário com características do imóvel e demonstrar a resposta do modelo, explicar os indicadores e mostrar como os modelos foram comparados.

## 4.3. Cenários principais

1. Abrir o dashboard e identificar o contexto e a origem dos dados.
2. Consultar os indicadores gerais da base e os resultados do treinamento.
3. Preencher as características de um imóvel e solicitar uma previsão.
4. Visualizar o valor estimado em USD e uma explicação simples do resultado.
5. Consultar a comparação dos modelos e os gráficos de avaliação.
6. Explicar as seis etapas do CRISP-DM com base em artefatos reais do projeto.

# 5. Escopo

## 5.1. Escopo obrigatório — MVP

- Descoberta do ZIP existente na pasta de trabalho.
- Auditoria dos arquivos e confirmação da base House Prices.
- Análise exploratória dos dados.
- Preparação de dados com pipelines do Scikit-learn.
- Treinamento de um baseline e de pelo menos dois modelos de regressão.
- Validação cruzada e avaliação em holdout.
- Exportação do pipeline final.
- API FastAPI com endpoint de previsão.
- Interface web JavaScript para realizar previsões.
- Dashboard com indicadores e visualizações derivadas dos dados reais.
- Documentação CRISP-DM, auditoria e avaliação.
- Instruções de execução local e testes básicos.

## 5.2. Escopo desejável

- Explicabilidade global por importância de variáveis ou coeficientes.
- Explicação local simplificada das características informadas, desde que tecnicamente sustentada.
- Comparação visual entre valores reais e previstos.
- Exportação de um CSV de previsões para o `test.csv` oficial, somente se os arquivos e o formato forem confirmados.
- Histórico local das últimas avaliações, se for simples e não comprometer o prazo.

## 5.3. Fora do escopo

- Tabela FIPE, veículos ou estimativa de automóveis.
- Avaliação de imóveis em Fortaleza, no Brasil ou em qualquer mercado diferente da base.
- Integração com portais imobiliários ou anúncios externos.
- Consulta de endereço, matrícula, IPTU ou cadastro imobiliário real.
- Autenticação, gestão de usuários ou permissões.
- Pagamentos, assinaturas ou funcionalidades comerciais.
- Aplicativo mobile nativo.
- Microserviços, Kubernetes ou infraestrutura distribuída.
- Implantação obrigatória em nuvem.
- Promessa de precisão profissional ou substituição de avaliação imobiliária regulamentada.

# 6. Descoberta obrigatória do ZIP e auditoria dos dados

Esta é a primeira atividade de implementação e deverá ocorrer antes da criação do modelo e da interface.

## 6.1. Instrução sobre o ZIP existente

O usuário já baixou o ZIP e o colocou na pasta de trabalho/repositório que será aberto pelo Codex ou Claude Code.

**Não pedir ao usuário que envie o ZIP novamente.** O agente deverá procurar o arquivo dentro do workspace atual, incluindo a raiz e subpastas razoáveis. Não presumir um nome ou caminho específico.

Procedimento:

1. Inspecionar a estrutura do repositório e identificar arquivos `.zip` existentes.
2. Excluir da busca diretórios gerados e volumosos, como `.git`, `.venv`, `venv`, `node_modules`, `__pycache__` e `dist`.
3. Listar o conteúdo dos ZIPs encontrados sem executar arquivos contidos neles.
4. Identificar arquivos candidatos, priorizando `train.csv`, `test.csv`, `data_description.txt` e `sample_submission.csv`.
5. Se o nome dos arquivos for diferente, procurar um CSV que contenha a coluna `SalePrice` e validar o restante do esquema com o dicionário de dados.
6. Se houver mais de um ZIP, identificar qual contém a base House Prices e registrar a decisão.
7. Se o ZIP já estiver extraído, procurar também os arquivos tabulares existentes no workspace.
8. Registrar caminho, tamanho e hash SHA-256 do ZIP original, quando possível.
9. Extrair uma cópia de trabalho sem sobrescrever o arquivo original nem arquivos preexistentes.
10. Preservar a origem dos dados e documentar os caminhos utilizados.

Se o ZIP não for encontrado, não inventar uma base nem fazer download silencioso de outra fonte para substituir a que o usuário informou que já colocou na pasta. Registrar o caminho pesquisado e a limitação, concluir a documentação/estrutura que não dependa dos dados e solicitar intervenção somente com evidência objetiva do problema.

## 6.2. Arquivos esperados na distribuição padrão

A distribuição padrão do Kaggle contém:

- `train.csv`: registros com `SalePrice` conhecido; é a fonte rotulada para treinamento e avaliação.
- `test.csv`: registros sem `SalePrice`; serve para gerar previsões, não para calcular métricas supervisionadas.
- `data_description.txt`: descrição das colunas e categorias.
- `sample_submission.csv`: exemplo do formato de submissão da competição.

O agente deverá confirmar quais desses arquivos estão realmente presentes no ZIP.

## 6.3. Auditoria técnica

Criar `docs/data_audit.md` e `artifacts/data_profile.json`, contendo:

- nome e caminho dos arquivos utilizados;
- tamanho e hash do arquivo-fonte, quando possível;
- formato e codificação;
- quantidade de linhas e colunas por arquivo;
- nomes das colunas;
- tipos inferidos e tipos esperados;
- presença de `SalePrice` no arquivo rotulado;
- quantidade de valores ausentes por coluna e percentual;
- quantidade de duplicidades;
- estatísticas descritivas das variáveis numéricas;
- frequência das categorias principais;
- valores extremos e inconsistências detectadas;
- distribuição da variável-alvo;
- variáveis com alto percentual de ausência;
- limitações identificadas;
- decisão de uso ou exclusão de variáveis, com justificativa.

Não registrar apenas conclusões. Incluir tabelas e evidências suficientes para que outra pessoa consiga verificar a auditoria.

## 6.4. Interpretação de valores ausentes

Não assumir que todo valor ausente significa dado desconhecido. No House Prices, algumas ausências categóricas podem representar a inexistência de um elemento do imóvel (por exemplo, ausência de garagem, porão, piscina ou acesso por viela), conforme o significado indicado no dicionário de dados.

O agente deverá consultar `data_description.txt` e distinguir, quando possível:

- ausência estrutural do atributo;
- informação desconhecida ou não registrada;
- ausência causada por falha ou inconsistência.

Não substituir automaticamente todas as ausências categóricas por uma única categoria genérica sem avaliar o significado. A estratégia adotada deverá ser documentada e aplicada dentro do pipeline, aprendendo parâmetros apenas no conjunto de treinamento.

## 6.5. Decisão de viabilidade

A base será considerada apta para regressão quando:

- houver arquivo rotulado com `SalePrice` numérico;
- houver número suficiente de observações válidas para treino e avaliação;
- as variáveis preditoras puderem ser interpretadas;
- inconsistências críticas puderem ser tratadas ou excluídas com justificativa.

Se o arquivo local não corresponder à base indicada, o agente deverá documentar o desvio e não assumir que a descrição deste PRD corresponde automaticamente ao conteúdo encontrado.

# 7. Definição do problema de Machine Learning

## 7.1. Tipo de problema

Regressão supervisionada.

## 7.2. Variável-alvo

`SalePrice` — preço pelo qual o imóvel foi vendido, expresso em dólares americanos (USD), conforme a descrição do dataset.

## 7.3. Variáveis preditoras

A competição apresenta 79 variáveis explicativas sobre propriedades, abrangendo características do lote, localização, estrutura, acabamento, cômodos, porão, garagem, áreas externas e condições de venda.

Exemplos de colunas que podem ser encontradas na distribuição padrão:

- `OverallQual`: avaliação geral de material e acabamento;
- `GrLivArea`: área habitável acima do solo;
- `YearBuilt`: ano de construção;
- `YearRemodAdd`: ano da reforma ou remodelação;
- `TotalBsmtSF`: área total do porão;
- `GarageCars`: capacidade da garagem em número de carros;
- `GarageArea`: área da garagem;
- `FullBath`: quantidade de banheiros completos acima do solo;
- `BedroomAbvGr`: quantidade de quartos acima do nível do porão;
- `LotArea`: área do terreno;
- `Neighborhood`: localização/bairro dentro de Ames;
- `KitchenQual`: qualidade da cozinha.

A lista é ilustrativa. O agente deverá validar cada coluna no arquivo local e no dicionário. Não criar colunas que não existam.

## 7.4. Exclusões obrigatórias

- `SalePrice` nunca poderá ser utilizado como variável de entrada.
- `Id` é um identificador e deverá ser excluído dos preditores, salvo experimento separado e justificativa metodológica explícita. Não deverá fazer parte do modelo de produção do MVP.
- Qualquer campo que revele direta ou indiretamente o valor-alvo deverá ser investigado e excluído para evitar vazamento.

## 7.5. Esquema de entrada da aplicação

O formulário da aplicação não precisa solicitar as 79 variáveis. O agente deverá selecionar um subconjunto coerente de características que:

- tenham significado compreensível para o usuário;
- sejam utilizáveis em uma avaliação conceitual do imóvel;
- estejam presentes e tenham qualidade suficiente na base;
- apresentem utilidade preditiva demonstrada;
- possam ser preenchidas de maneira realista na interface.

A seleção deverá ser documentada. Exemplos candidatos incluem área habitável, área do terreno, qualidade geral, ano de construção, quantidade de quartos, banheiros, garagem, porão e bairro. A lista definitiva depende da auditoria.

**Regra de consistência:** o pipeline usado pela API deverá ter sido treinado exatamente com as variáveis previstas no contrato de entrada do formulário. Não treinar com um conjunto de colunas e enviar outro na previsão.

O agente poderá realizar experimentos com o conjunto completo de atributos e com um conjunto reduzido para a interface. Se fizer isso, deverá separar claramente os resultados e explicar qual artefato é utilizado pela aplicação.

# 8. Metodologia CRISP-DM

O desenvolvimento deverá ser documentado nas seis etapas abaixo. Criar `docs/crisp_dm.md` com decisões, evidências, resultados e limitações de cada etapa.

## 8.1. Business Understanding — Entendimento do negócio

- contextualizar a estimativa de preços de imóveis;
- definir problema e objetivos;
- formular pergunta e hipóteses;
- identificar usuários e cenário de demonstração;
- definir critérios de sucesso analítico e funcional;
- esclarecer o limite geográfico e temporal do dataset.

**Entregáveis:** definição do problema, objetivos, hipóteses, escopo e critérios de sucesso.

## 8.2. Data Understanding — Entendimento dos dados

- localizar e carregar os arquivos;
- estudar o dicionário de dados;
- identificar a variável-alvo;
- descrever estrutura, tipos e cardinalidade;
- analisar ausências, duplicidades e valores atípicos;
- estudar distribuição de `SalePrice`;
- analisar relações entre atributos e preço.

**Entregáveis:** auditoria, perfil JSON, gráficos exploratórios e registro de limitações.

## 8.3. Data Preparation — Preparação dos dados

- excluir identificadores e variáveis com vazamento;
- corrigir tipos;
- tratar valores ausentes de acordo com o significado;
- separar variáveis numéricas e categóricas;
- codificar categorias;
- escalar variáveis quando necessário;
- definir conjunto de atributos do modelo;
- separar treino e holdout antes de ajustar transformadores.

Toda transformação aprendida (imputação, escalonamento, codificação ou seleção baseada em dados) deverá ser ajustada somente nos folds ou dados de treinamento correspondentes.

**Entregáveis:** pipeline reproduzível e documentação das transformações.

## 8.4. Modeling — Modelagem

Treinar baseline e modelos de regressão descritos na seção 9. Todos os modelos comparados deverão utilizar a mesma partição de dados e o mesmo conjunto de atributos para que a comparação seja justa.

**Entregáveis:** pipelines treinados, configuração, resultados de validação cruzada e artefato candidato.

## 8.5. Evaluation — Avaliação

- calcular métricas em validação cruzada;
- avaliar o candidato final no holdout reservado;
- comparar previsões e valores reais;
- estudar distribuição dos resíduos;
- examinar erros mais altos;
- verificar overfitting e limitações;
- decidir se o modelo supera o baseline.

**Entregáveis:** tabela comparativa, gráficos e relatório de avaliação.

## 8.6. Deployment — Implantação

- persistir o pipeline completo;
- carregar o mesmo artefato na API;
- validar entrada e saída;
- desenvolver interface de demonstração;
- disponibilizar indicadores e gráficos;
- documentar execução local e demonstração.

**Entregáveis:** aplicação funcional, API, instruções e roteiro de apresentação.

# 9. Estratégia de modelagem e avaliação

## 9.1. Baseline obrigatório

Treinar `DummyRegressor` (estratégia `median` ou outra estratégia justificada) para estabelecer uma referência mínima. Um modelo de Machine Learning não deverá ser considerado bem-sucedido apenas por gerar previsões: ele deverá superar o baseline em métricas de validação relevantes.

## 9.2. Modelos candidatos

Treinar pelo menos dois modelos de regressão, além do baseline:

1. **Ridge Regression** ou `LinearRegression`: referência linear e relativamente interpretável. Preferir Ridge se a análise indicar multicolinearidade relevante.
2. **RandomForestRegressor**: modelo baseado em árvores, capaz de representar relações não lineares e interações.
3. **HistGradientBoostingRegressor** ou `GradientBoostingRegressor`: candidato adicional caso seja compatível com a versão instalada e com o pré-processamento definido.

Não é necessário usar bibliotecas externas como XGBoost, LightGBM ou CatBoost no MVP. Só adicioná-las se houver justificativa, disponibilidade no ambiente e ganho demonstrado.

## 9.3. Pré-processamento

Utilizar `Pipeline` e `ColumnTransformer` do Scikit-learn.

- Numéricas: imputação adequada; escalonamento para modelos sensíveis à escala, como Ridge e KNN caso KNN seja incluído.
- Categóricas: imputação sem perder o significado de ausências estruturais; `OneHotEncoder(handle_unknown="ignore")` ou alternativa justificada.
- O pré-processamento deve ser parte do pipeline salvo junto com o estimador.
- Não calcular médias, medianas, escalas ou vocabulários de categorias usando o holdout.

## 9.4. Transformação da variável-alvo

A distribuição de `SalePrice` deverá ser inspecionada. A documentação da competição utiliza erro quadrático sobre o logaritmo dos preços como métrica de avaliação.

O agente deverá testar ou justificar o uso de transformação `log1p(SalePrice)` no treinamento. Se usar, implementar transformação e inversão de forma reproduzível, por exemplo com `TransformedTargetRegressor` ou wrapper equivalente. A previsão apresentada ao usuário deverá voltar à escala original em USD.

Não aplicar transformação automaticamente sem documentar o efeito e o protocolo de avaliação.

## 9.5. Divisão e validação

Estratégia padrão do MVP:

- reservar 20% do arquivo rotulado como holdout final;
- utilizar 80% para desenvolvimento e validação;
- aplicar `KFold(n_splits=5, shuffle=True, random_state=42)` na parte de desenvolvimento, se a auditoria confirmar que a divisão aleatória é apropriada;
- utilizar a mesma divisão e os mesmos folds para comparar todos os algoritmos;
- selecionar hiperparâmetros exclusivamente dentro dos dados de desenvolvimento;
- avaliar o modelo selecionado uma única vez no holdout final após as decisões de modelagem.

A escolha aleatória deverá ser justificada como estimativa de generalização para dados da mesma população amostrada. Se houver razão concreta para uma análise temporal, o agente poderá fazer uma avaliação temporal complementar usando `YrSold` e documentar as limitações de tamanho. Não substituir a avaliação principal silenciosamente.

## 9.6. Métricas

Apresentar no mínimo:

- **MAE (USD):** erro absoluto médio, fácil de interpretar na escala monetária;
- **RMSE (USD):** raiz do erro quadrático médio, com penalização maior para erros grandes;
- **R²:** proporção da variabilidade explicada pelo modelo, interpretada com cautela;
- **RMSLE:** raiz do erro quadrático médio entre `log1p(y_real)` e `log1p(y_previsto)`, desde que os preços previstos e reais sejam não negativos.

A métrica de referência da competição Kaggle é RMSE entre os logaritmos dos preços. Portanto, RMSLE deverá ser reportada para permitir comparação conceitual com a competição. Para a apresentação de negócio, MAE e RMSE em USD também são obrigatórias.

Definir previamente qual métrica será primária para a seleção (recomendação: RMSLE para alinhamento com a competição, acompanhada de MAE em USD). Registrar a decisão em `docs/model_evaluation.md`.

Não comparar métricas calculadas em escalas ou conjuntos diferentes.

## 9.7. Diagnóstico

Gerar, no mínimo:

- gráfico de preço real versus previsto;
- gráfico de resíduos;
- distribuição dos erros absolutos;
- tabela com os maiores erros absolutos;
- comparação de métricas dos algoritmos;
- análise de desempenho do treino versus validação, quando aplicável.

Evitar conclusões causais a partir de correlações ou importância de variáveis.

## 9.8. Arquivo `test.csv` e submissão Kaggle

Se `test.csv` estiver presente, ele não contém `SalePrice` e não serve para calcular métricas locais. Poderá ser utilizado apenas para gerar previsões após a seleção do modelo.

A geração de `submission.csv` é opcional e deverá seguir o formato exigido pela competição (`Id`, `SalePrice`). Não misturar esse resultado com a avaliação holdout e não afirmar que foi validado sem rótulos.

# 10. Explicabilidade e interpretação

## 10.1. Explicação global

Fornecer uma visão das características mais relevantes para as previsões, conforme o modelo selecionado:

- importância por permutação, preferencialmente calculada em dados de validação/holdout;
- coeficientes padronizados ou não padronizados para modelos lineares, com interpretação adequada;
- importância nativa do Random Forest apenas como análise complementar;
- gráficos de dependência simples para algumas variáveis, se apropriado.

## 10.2. Explicação individual

O MVP poderá apresentar as principais características do imóvel informado e uma observação sobre o contexto de uso do modelo. Não inventar uma decomposição exata da previsão sem usar método compatível e validado.

SHAP é opcional. Não adicioná-lo se elevar muito a complexidade ou se a explicação não puder ser apresentada corretamente.

# 11. Experiência do usuário e interface

## 11.1. Direção visual

Criar uma aplicação profissional de análise imobiliária, simples e legível. Priorizar hierarquia visual e compreensão imediata, em vez de excesso de cards, indicadores ou efeitos.

A interface deverá estar em português, mas poderá manter os nomes de variáveis originais em tooltips ou na documentação técnica.

## 11.2. Navegação principal

1. **Visão geral** — contexto, indicadores da base e resumo do modelo.
2. **Avaliar imóvel** — formulário e estimativa.
3. **Análise dos dados** — distribuições e relações entre características e preços.
4. **Modelos e resultados** — comparação de algoritmos e métricas.
5. **Sobre o projeto** — origem, CRISP-DM, limitações e versão do modelo.

## 11.3. Dashboard

Apresentar dados reais, quando disponíveis:

- quantidade de imóveis rotulados;
- faixa de preços e mediana;
- preço médio, quando útil e acompanhado de contexto;
- algoritmo selecionado;
- MAE, RMSE, RMSLE e R² do holdout;
- data de treinamento do artefato;
- indicação explícita de que os dados são de Ames, Iowa, e históricos.

Não criar indicadores ilustrativos que pareçam resultados reais. Se o modelo ainda não tiver sido treinado, exibir estado vazio ou aviso de configuração pendente.

## 11.4. Formulário de avaliação

O formulário deverá utilizar exatamente o subconjunto de variáveis definido para o modelo de aplicação.

Requisitos:

- apresentar rótulos em linguagem simples;
- apresentar ajuda contextual para variáveis menos conhecidas;
- utilizar listas para variáveis categóricas, com opções provenientes dos dados de treinamento ou de metadados versionados;
- utilizar campos numéricos com validação de faixa plausível baseada nos dados e no dicionário;
- não impor limites arbitrários sem justificativa;
- informar quais campos são obrigatórios;
- mostrar mensagens claras para campos inválidos;
- mostrar loading durante a previsão;
- impedir múltiplos envios acidentais;
- indicar moeda USD no resultado.

Se o modelo utilizar mais variáveis do que o usuário consegue informar razoavelmente, reduzir o conjunto de atributos ou criar uma experiência guiada. Não preencher silenciosamente campos indispensáveis com valores inventados.

## 11.5. Tela de resultado

Exibir:

- preço estimado em USD;
- nome/versão do modelo;
- indicação de que se trata de estimativa experimental;
- resumo das características enviadas;
- mensagem sobre o escopo geográfico e histórico;
- explicação simples de que o resultado depende da distribuição da base e não é avaliação profissional.

Não apresentar intervalo de confiança ou faixa de preço como se fosse estatisticamente calibrado sem que esse intervalo tenha sido calculado e validado.

## 11.6. Responsividade e acessibilidade

- priorizar desktop para apresentação;
- suportar tablet e celular;
- manter contraste e tamanho de texto adequados;
- garantir foco visível e navegação por teclado nos controles principais;
- usar rótulos associados aos campos;
- não depender apenas de cores para transmitir significado.

# 12. Arquitetura técnica

## 12.1. Visão geral

```text
Usuário
  |
  v
Frontend (HTML + CSS + JavaScript)
  |
  | HTTP / JSON
  v
FastAPI
  |
  +--> Prediction Service
  |       |
  |       v
  |   Pipeline Scikit-learn (.joblib)
  |
  +--> Analytics Service
          |
          v
      Artefatos de dados e métricas

Processo offline de ML:
Arquivos locais -> auditoria -> preparação -> treino/validação
               -> avaliação -> pipeline final -> artefatos
```

O treinamento deverá ser executado por scripts offline. A API apenas carrega o pipeline pronto e realiza previsões. Não treinar modelos durante uma requisição.

## 12.2. Tecnologias

| Camada | Tecnologia | Responsabilidade |
|---|---|---|
| Linguagem de dados | Python | Inspeção, preparação e modelagem |
| Manipulação | Pandas e NumPy | Dados tabulares e operações numéricas |
| ML | Scikit-learn | Pipelines, modelos e métricas |
| Backend | FastAPI + Pydantic | API e validação |
| Servidor | Uvicorn | Execução local da API |
| Frontend | HTML, CSS e JavaScript | Interface web |
| Gráficos | Plotly.js | Visualizações interativas |
| Persistência | Joblib + JSON | Modelo e metadados |
| Testes | Pytest | Testes de dados, ML e API |

Não utilizar framework frontend pesado se HTML/CSS/JS resolverem o escopo. React ou Vue só deverão ser adotados se já existirem no repositório ou houver justificativa concreta.

## 12.3. Organização sugerida do repositório

Adaptar à estrutura existente, preservando arquivos e convenções já adotados. Estrutura sugerida:

```text
housevalue-ai/
├── PRD_HouseValue_AI.md
├── README.md
├── requirements.txt
├── .env.example
├── .gitignore
│
├── data/
│   ├── raw/                 # cópia de trabalho dos arquivos originais
│   ├── processed/           # dados derivados, reproduzíveis
│   └── README.md            # origem, caminho e política de dados
│
├── ml/
│   ├── __init__.py
│   ├── inspect_data.py
│   ├── exploratory_analysis.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   ├── explainability.py
│   └── config.py
│
├── backend/
│   ├── __init__.py
│   ├── main.py
│   ├── core/
│   │   └── config.py
│   ├── schemas/
│   │   ├── prediction.py
│   │   └── metadata.py
│   ├── api/
│   │   ├── health.py
│   │   ├── predictions.py
│   │   └── analytics.py
│   └── services/
│       ├── prediction_service.py
│       └── analytics_service.py
│
├── frontend/
│   ├── index.html
│   └── assets/
│       ├── css/style.css
│       └── js/
│           ├── api.js
│           ├── app.js
│           └── charts.js
│
├── artifacts/
│   ├── models/
│   ├── metrics/
│   ├── figures/
│   └── reports/
│
├── docs/
│   ├── data_audit.md
│   ├── crisp_dm.md
│   ├── methodology.md
│   ├── model_evaluation.md
│   ├── model_card.md
│   └── presentation.md
│
└── tests/
    ├── test_data.py
    ├── test_preprocessing.py
    ├── test_model.py
    └── test_api.py
```

O agente poderá adaptar nomes e localização de módulos, desde que mantenha separação clara entre dados, ML, API, frontend, artefatos, documentação e testes.

# 13. Contrato da API

Base local sugerida: `http://localhost:8000`.

Documentação automática: `/docs`.

## 13.1. `GET /api/health`

Verificar disponibilidade da aplicação e do artefato de modelo.

Resposta conceitual:

```json
{
  "status": "ok",
  "model_loaded": true,
  "app_version": "1.0.0"
}
```

## 13.2. `GET /api/model/info`

Retornar algoritmo, versão, data de treinamento, variável-alvo, moeda, lista de variáveis esperadas e principais métricas de avaliação.

## 13.3. `GET /api/form-options`

Retornar opções de campos categóricos, rótulos, descrições e limites de entrada usados no formulário. As opções deverão corresponder às categorias aceitas pelo pipeline e não ser inventadas no frontend.

## 13.4. `POST /api/predict`

Receber os atributos selecionados para o formulário e retornar a estimativa.

O schema final deverá ser definido somente depois da auditoria e da escolha do conjunto de atributos.

Exemplo ilustrativo, não definitivo:

```json
{
  "overall_qual": 7,
  "gr_liv_area": 1800,
  "year_built": 2005,
  "total_bsmt_sf": 900,
  "garage_cars": 2,
  "bedroom_abv_gr": 3,
  "full_bath": 2,
  "neighborhood": "CollgCr"
}
```

Resposta conceitual:

```json
{
  "predicted_sale_price": 215000.0,
  "currency": "USD",
  "target": "SalePrice",
  "model_name": "RandomForestRegressor",
  "model_version": "1.0.0",
  "is_estimate": true,
  "dataset_context": "Ames, Iowa — vendas históricas"
}
```

Os valores do exemplo são fictícios e servem apenas para ilustrar o contrato. Nunca utilizá-los como resultado ou demonstração de desempenho.

## 13.5. `GET /api/analytics/overview`

Retornar agregações reais, como quantidade de registros, estatísticas de `SalePrice` e métricas do modelo. Não calcular agregações a partir de dados fictícios.

## 13.6. `GET /api/analytics/model-comparison`

Retornar comparação de modelos com métricas, protocolo de validação e identificação do modelo selecionado.

## 13.7. `GET /api/analytics/feature-importance`

Retornar importâncias ou coeficientes conforme método disponível e documentado. A resposta deverá incluir o tipo de medida para evitar interpretação incorreta.

## 13.8. Erros

Tratar, no mínimo:

- modelo não encontrado ou incompatível;
- campo obrigatório ausente;
- tipo de dado inválido;
- categoria não reconhecida;
- valor numérico fora dos limites definidos;
- falha ao gerar previsão;
- artefatos de métricas indisponíveis.

Retornar códigos HTTP apropriados e mensagens compreensíveis. Não expor stack traces ao frontend.

# 14. Persistência e artefatos

## 14.1. Modelo

Persistir o pipeline completo (pré-processamento + estimador + transformação de alvo, se utilizada) em `artifacts/models/`.

## 14.2. Metadados

Criar `artifacts/models/model_metadata.json` com:

- nome do algoritmo;
- versão do artefato;
- data/hora de treinamento;
- nome do arquivo de origem;
- hash da fonte, quando possível;
- variável-alvo;
- variáveis de entrada e ordem esperada;
- estratégia de pré-processamento;
- método de validação;
- semente aleatória;
- métricas de validação e holdout;
- versões das bibliotecas relevantes.

## 14.3. Métricas

Salvar comparação em `artifacts/metrics/model_comparison.json` e uma tabela CSV ou Markdown de leitura humana.

## 14.4. Gráficos

Salvar figuras em `artifacts/figures/`, preferencialmente em PNG e/ou HTML interativo, com nomes descritivos.

## 14.5. Dados

Não versionar dados brutos grandes ou arquivos de origem protegidos sem necessidade. Preservar o ZIP original e documentar como reproduzir os dados processados.

# 15. Requisitos funcionais

| ID | Requisito | Prioridade |
|---|---|---|
| RF-01 | Localizar o ZIP existente no workspace | Obrigatória |
| RF-02 | Inspecionar e identificar os arquivos House Prices | Obrigatória |
| RF-03 | Gerar auditoria e perfil dos dados | Obrigatória |
| RF-04 | Definir `SalePrice` como alvo após validação | Obrigatória |
| RF-05 | Realizar análise exploratória | Obrigatória |
| RF-06 | Preparar os dados por pipeline reproduzível | Obrigatória |
| RF-07 | Treinar baseline e pelo menos dois modelos | Obrigatória |
| RF-08 | Comparar modelos por validação cruzada | Obrigatória |
| RF-09 | Avaliar o modelo escolhido em holdout | Obrigatória |
| RF-10 | Salvar o pipeline e seus metadados | Obrigatória |
| RF-11 | Disponibilizar API de previsão | Obrigatória |
| RF-12 | Criar formulário web para previsão | Obrigatória |
| RF-13 | Apresentar indicadores e gráficos reais | Obrigatória |
| RF-14 | Exibir escopo, moeda e limitações | Obrigatória |
| RF-15 | Apresentar explicabilidade global | Desejável |
| RF-16 | Gerar `submission.csv` para Kaggle | Opcional |
| RF-17 | Guardar histórico local de previsões | Opcional |

# 16. Requisitos não funcionais

| ID | Requisito |
|---|---|
| RNF-01 | Código modular, tipado quando apropriado e documentado |
| RNF-02 | Treinamento separado da execução da API |
| RNF-03 | Resultados reproduzíveis com sementes e versões registradas |
| RNF-04 | Ausência de vazamento entre treino, validação e holdout |
| RNF-05 | Interface responsiva e legível |
| RNF-06 | Tratamento de erros previsível |
| RNF-07 | Nenhum segredo ou chave no frontend ou no repositório |
| RNF-08 | Execução local documentada |
| RNF-09 | Métricas e exemplos sempre provenientes de dados/execução reais, salvo quando explicitamente marcados como exemplos fictícios |
| RNF-10 | Complexidade compatível com um projeto acadêmico individual ou pequeno grupo |

# 17. Segurança e confiabilidade

- Não executar scripts ou binários encontrados dentro do ZIP durante a inspeção.
- Não sobrescrever ZIP, CSVs ou documentos originais.
- Não armazenar chaves ou credenciais no código.
- Validar entrada com Pydantic e validações de domínio.
- Limitar tamanho de requisições e tratar erros de conversão.
- Manter dependências documentadas.
- Não carregar artefatos Joblib de origem desconhecida; neste projeto, carregar apenas artefatos gerados localmente e controlados pelo próprio pipeline.
- Não registrar dados além do necessário.
- Não expor caminhos locais absolutos ou informações internas desnecessárias na resposta da API.

# 18. Testes e critérios de aceite

## 18.1. Dados

- [ ] O agente encontrou o ZIP no workspace sem pedir novo upload.
- [ ] O arquivo original foi preservado.
- [ ] O conteúdo foi inspecionado e documentado.
- [ ] A presença de `SalePrice` foi confirmada no arquivo rotulado.
- [ ] As quantidades de linhas/colunas foram calculadas, não presumidas.
- [ ] A descrição das colunas foi consultada, quando disponível.
- [ ] A política de valores ausentes foi documentada.

## 18.2. Metodologia e ML

- [ ] O projeto documenta as seis fases do CRISP-DM.
- [ ] `SalePrice` é o alvo e não aparece entre as features.
- [ ] `Id` não entra no modelo final sem justificativa.
- [ ] Existe um baseline.
- [ ] Pelo menos dois modelos foram treinados.
- [ ] Todos os modelos comparados usaram os mesmos folds e atributos.
- [ ] O pré-processamento está dentro do pipeline.
- [ ] Não há ajuste de transformadores usando holdout.
- [ ] As métricas MAE, RMSE, R² e RMSLE (quando aplicável) foram calculadas corretamente.
- [ ] O resultado de holdout foi mantido separado do processo de seleção.
- [ ] O modelo selecionado foi comparado com o baseline.
- [ ] Os resultados e limitações foram documentados.

Não estabelecer um valor mínimo arbitrário de R² ou um limite de erro sem fundamento. Se nenhum modelo superar o baseline de maneira consistente, registrar o resultado honestamente, investigar causas e não declarar sucesso preditivo.

## 18.3. Backend

- [ ] FastAPI inicia sem erro.
- [ ] `/api/health` informa o estado da API e do modelo.
- [ ] `/api/model/info` retorna metadados reais.
- [ ] `/api/form-options` retorna opções consistentes com o pipeline.
- [ ] `/api/predict` recebe uma entrada válida e retorna previsão numérica em USD.
- [ ] Entradas inválidas geram respostas de erro adequadas.
- [ ] A API não treina o modelo durante uma requisição.
- [ ] `/docs` está acessível localmente.

## 18.4. Frontend

- [ ] A aplicação abre localmente.
- [ ] A navegação é compreensível.
- [ ] O formulário utiliza os atributos corretos.
- [ ] Os campos categóricos usam opções aceitas pelo modelo.
- [ ] A previsão é obtida pela API real.
- [ ] Os resultados identificam USD e o contexto de Ames, Iowa.
- [ ] Os gráficos apresentam dados e métricas reais.
- [ ] Estados de carregamento, erro e ausência de artefatos são tratados.
- [ ] Layout funciona em desktop e em tela menor.

## 18.5. Entrega acadêmica

- [ ] Auditoria da base concluída.
- [ ] Relatório CRISP-DM concluído.
- [ ] Relatório de avaliação concluído.
- [ ] Model card/limitações concluídos.
- [ ] README de instalação e execução atualizado.
- [ ] Roteiro de apresentação concluído.
- [ ] Demonstração ponta a ponta testada.

# 19. Estrutura dos documentos finais

## 19.1. `docs/data_audit.md`

Registrar origem, conteúdo real do ZIP, estrutura, perfil, ausências, duplicidades, variáveis, limitações e decisão sobre a base.

## 19.2. `docs/crisp_dm.md`

Descrever as seis etapas CRISP-DM, com decisões, evidências e entregáveis.

## 19.3. `docs/methodology.md`

Descrever a definição do problema, variável-alvo, atributos, pré-processamento, divisão de dados, algoritmos e métricas.

## 19.4. `docs/model_evaluation.md`

Apresentar tabela de métricas, validação cruzada, holdout, gráficos, comparação, modelo escolhido e limitações.

## 19.5. `docs/model_card.md`

Registrar finalidade, dados utilizados, população representada, métricas, usos apropriados, usos não recomendados, limitações e versão.

## 19.6. `docs/presentation.md`

Preparar apresentação de 7 a 10 minutos, com a seguinte organização sugerida:

1. Problema e objetivo.
2. Origem e descrição da base.
3. CRISP-DM.
4. Análise exploratória e preparação.
5. Algoritmos testados.
6. Métricas e comparação.
7. Demonstração da aplicação.
8. Limitações e conclusão.

O roteiro deverá usar resultados reais produzidos pelo código, não números de exemplo deste PRD.

# 20. Plano de execução do agente

Executar em fases sequenciais. Não começar pelo design da interface.

## Fase 0 — Reconhecimento do repositório

1. Ler este PRD.
2. Inspecionar a estrutura atual do workspace.
3. Identificar ambiente, dependências e convenções existentes.
4. Localizar o ZIP já baixado pelo usuário.
5. Registrar os arquivos encontrados e sua origem.

**Saída:** nota inicial de descoberta e plano de implementação ajustado ao repositório.

## Fase 1 — Auditoria e decisão dos dados

1. Inspecionar o ZIP com segurança.
2. Localizar os arquivos House Prices.
3. Ler o dicionário de dados.
4. Carregar e perfilar os dados.
5. Confirmar `SalePrice` e as variáveis candidatas.
6. Registrar problemas e decisões.

**Gate:** não avançar para modelagem sem confirmar a variável-alvo e a adequação da base.

## Fase 2 — Análise e preparação

1. Construir script de análise exploratória.
2. Gerar gráficos.
3. Definir atributos para os modelos.
4. Definir atributos para o formulário.
5. Implementar pipelines sem vazamento.
6. Documentar as escolhas.

**Saída:** dados e pipeline de preparação reproduzíveis.

## Fase 3 — Treinamento e avaliação

1. Implementar baseline.
2. Implementar Ridge/Linear Regression.
3. Implementar Random Forest.
4. Implementar um terceiro modelo se houver justificativa.
5. Executar validação cruzada.
6. Selecionar modelo.
7. Avaliar no holdout.
8. Gerar métricas e gráficos.
9. Salvar pipeline e metadados.

**Gate:** a previsão deve ser reproduzível a partir do artefato salvo antes de iniciar a integração com a API.

## Fase 4 — Backend

1. Criar aplicação FastAPI.
2. Criar schemas com as features definitivas.
3. Implementar carregamento do pipeline.
4. Implementar endpoints.
5. Implementar tratamento de erros.
6. Criar testes.

**Saída:** API funcional e documentada.

## Fase 5 — Frontend

1. Criar estrutura de navegação.
2. Implementar dashboard.
3. Implementar formulário.
4. Integrar com API.
5. Apresentar resultado.
6. Implementar gráficos.
7. Validar responsividade e acessibilidade básica.

**Saída:** demonstração web funcional.

## Fase 6 — Finalização acadêmica

1. Executar todos os testes.
2. Verificar coerência dos números exibidos.
3. Atualizar documentação.
4. Criar roteiro de apresentação.
5. Testar a demonstração do início ao fim.
6. Registrar limitações conhecidas.

**Saída:** projeto reproduzível e pronto para apresentação.

# 21. Comandos de execução esperados

O agente deverá adaptar os comandos ao ambiente e ao gerenciador de dependências encontrado. Se utilizar `requirements.txt`, fornecer instruções equivalentes às seguintes:

Criar ambiente virtual:

```bash
python -m venv .venv
```

Ativar no Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Instalar dependências:

```bash
pip install -r requirements.txt
```

Auditar os dados:

```bash
python -m ml.inspect_data
```

Executar análise exploratória:

```bash
python -m ml.exploratory_analysis
```

Treinar:

```bash
python -m ml.train
```

Avaliar:

```bash
python -m ml.evaluate
```

Iniciar API:

```bash
uvicorn backend.main:app --reload
```

A documentação da API deverá estar disponível em `http://localhost:8000/docs`.

O README deverá explicar como servir o frontend e como iniciar backend e frontend no ambiente local.

# 22. Instruções finais ao Codex / Claude Code

Atue como engenheiro de software e cientista de dados responsável por implementar este produto. Siga o PRD como especificação, respeitando o conteúdo real do repositório e dos arquivos de dados.

## Regras obrigatórias

1. Leia o PRD integralmente antes de implementar.
2. O ZIP já foi baixado pelo usuário e está na pasta de trabalho. Procure-o no workspace; não solicite novo upload.
3. Não presuma o caminho nem o nome do ZIP: descubra-os.
4. Não sobrescreva nem modifique o ZIP original.
5. Inspecione os arquivos reais antes de definir colunas, tipos, filtros ou modelos.
6. Não invente quantidade de linhas, colunas, estatísticas, métricas ou resultados.
7. Não use FIPE ou dados de veículos; o projeto é HouseValue AI.
8. Utilize `SalePrice` como alvo somente após confirmar sua presença e significado.
9. Exclua `Id` dos preditores do modelo final.
10. Previna vazamento de dados em todas as etapas.
11. Compare baseline e modelos usando os mesmos dados e métricas.
12. Separe seleção de modelo e avaliação final em holdout.
13. Salve o pipeline completo, não apenas o estimador.
14. Faça a API consumir o artefato salvo; não treine na requisição.
15. Garanta que o formulário envie exatamente as features esperadas pelo modelo.
16. Mantenha todos os resultados da interface vinculados a artefatos reais.
17. Documente decisões e limitações.
18. Execute testes e corrija falhas antes de concluir.

## Autonomia e comunicação

- Tome decisões técnicas razoáveis dentro do escopo e registre-as.
- Não pare para pedir confirmação sobre escolhas de implementação comuns.
- Não pergunte novamente onde está o ZIP antes de pesquisar no workspace.
- Se houver múltiplos arquivos candidatos, use evidências de nomes, schema e dicionário para identificar a base correta e documente a seleção.
- Se a base estiver incompleta ou diferente do esperado, explique precisamente o que foi encontrado e proponha a alternativa mais fiel ao objetivo acadêmico.
- Não declare uma etapa concluída sem executar e verificar seus resultados.
- Ao final, apresente resumo do que foi implementado, comandos para executar, testes realizados, localização dos artefatos e limitações restantes.

## Definição de concluído

O projeto estará concluído quando a base real tiver sido auditada, o fluxo CRISP-DM estiver documentado, o modelo tiver sido treinado e avaliado de forma reproduzível, o pipeline salvo estiver integrado à API, a interface conseguir solicitar e exibir uma previsão real, e a demonstração puder ser executada localmente seguindo o README.

---

# 23. Referências

1. **Documento da disciplina:** `Bases de dados - Projetos de ML.docx`, seção Regressão, item 2 — House Prices: Advanced Regression Techniques. Documento disponibilizado pelo usuário.
2. **Kaggle — House Prices: Advanced Regression Techniques, Overview:** https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview
3. **Kaggle — House Prices: Advanced Regression Techniques, Data:** https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data
4. **Scikit-learn — User Guide:** https://scikit-learn.org/stable/user_guide.html
5. **FastAPI — Documentation:** https://fastapi.tiangolo.com/
6. **CRISP-DM — referência metodológica:** https://www.datascience-pm.com/crisp-dm-2/

**Nota de proveniência:** a seleção do dataset House Prices deriva do documento da disciplina. As características oficiais da competição, a definição de `SalePrice`, os arquivos padrão e a métrica Kaggle foram conferidos na página da competição. A estrutura e os resultados efetivos da cópia local deverão ser confirmados pelo agente na auditoria do ZIP.
