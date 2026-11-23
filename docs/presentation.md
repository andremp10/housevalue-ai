# Roteiro de fala — HouseValue AI (≈8 min de fala + demo ao vivo)

> Gerado por `python -m ml.build_docs` a partir dos artefatos: todos os números abaixo são os reais do projeto.
> Acompanha o deck de slides (mesma numeração, 16 slides). A fala completa de cada slide também está nas notas do deck.
> **Fio da história:** problema e hipóteses (1–3) → o caminho em 5 etapas, CRISP-DM (4) → dados, preparação, treino, prova (5–11) → o sistema (12–14) → resposta às hipóteses (15) → perguntas (16). A casa da capa volta na conclusão para fechar a história.
> **Antes de começar:** `.venv\Scripts\python -m uvicorn backend.main:app --port 8000` e abrir `http://localhost:8000/` (abre na aba **Simular um imóvel**).
> **Plano B:** se o servidor não subir, o slide 13 e a tabela do slide 14 são capturas reais do sistema.
> Cores no sistema e nos gráficos: **laranja = preço real da venda**, **azul = estimativa do modelo**; verde / âmbar / vermelho = erro até 10% / 10–20% / mais de 20%.

## Parte 1 · O problema (CRISP-DM: entendimento do negócio)

### 1. Capa: "Quanto vale esta casa?" (≈25 s)
- Casa real da prova (Id 434, bairro Gilbert, 149 m², 3 quartos, construída em 1997, vendida em 2008): vendida por US$ 181.000; o modelo estimou US$ 184.160 (1,7%).
- "Hoje vamos mostrar como chegamos nisso e como provamos o quanto ele acerta, inclusive quando erra." Apresentar a convenção laranja/azul.

### 2. Imagine que você vai vender sua casa (≈30 s)
- Você sabe a área, os quartos, o ano, o acabamento, o bairro. Não sabe **quanto pedir**.
- Hoje: palpite, vizinhos ou avaliação paga. Um corretor compara com **vendas parecidas**: é isso que vamos ensinar o computador a fazer.

### 3. A nossa pergunta e as hipóteses (≈40 s)
- **Dá para estimar o preço de uma casa só pelas características dela?**
- Machine Learning: exemplos (1.460 vendas com preço real) → o computador aprende os padrões → casa nova recebe um preço. A resposta é um número: **regressão**.
- Hipóteses: **H1** as características têm relação com o preço; **H2** modelos que aprendem curvas (árvores) erram menos que uma "linha reta" (regressão linear); **H3** algoritmos diferentes têm desempenhos diferentes nos mesmos dados; **H4** o modelo pode virar um sistema de uso real, com o mesmo tratamento do treino.

## Parte 2 · Como construímos

### 4. O caminho, em 5 etapas (≈30 s)
CRISP-DM tem 6 fases: a 1ª (entendimento do negócio) é o problema; as outras viram as etapas 1 Conhecer os dados (entendimento dos dados) · 2 Preparar (preparação) · 3 Treinar e comparar (modelagem) · 4 Provar (avaliação) · 5 Virar sistema (implantação). Cada etapa deixa um resultado: 1.460 vendas · 14 informações · 9 candidatos · 292 casas nunca vistas · sistema com 3 abas. A barrinha no alto de cada slide mostra a etapa.

### 5. Etapa 1 · Os dados (≈35 s)
- Base **House Prices** (Kaggle), indicada pela disciplina: **1.460 casas vendidas** com o preço real e **79 características** cada; mais 1.459 casas sem preço (não entram na avaliação).
- Metade das casas custa menos de US$ 163.000; os preços vão de US$ 34.900 a US$ 755.000, com poucas casas muito caras (cauda longa). Por isso o modelo aprende o preço em escala log e converte de volta para dólares.
- Limite: **Ames, Iowa (EUA), vendas de 2006 a 2010**. O método vale; os valores não servem para o Brasil de hoje.

### 6. Etapa 2 · Preparar: 14 informações (≈40 s)
- Ninguém digita 79 informações: o formulário usa **14** que qualquer pessoa sabe responder: áreas (construída, porão, terreno, em m²), qualidade, conservação, cozinha, ar central, anos de construção e reforma, quartos, banheiros, garagem, lareiras e bairro. Repetidas saíram (ex.: vagas × área da garagem).
- **Tratamento dos dados:** "NA" em várias colunas quer dizer "não tem" (ex.: "não tem piscina"), não "faltando". Corrigido pelo dicionário da base, sem inventar valores (detalhes em `docs/data_audit.md`).
- **Sem "colar":** tudo decidido sem olhar as casas reservadas para o teste final.

### 7. Etapa 3 · Como um aluno antes da prova (≈55 s)
1. **A prova fica lacrada:** 292 casas (20%) separadas antes de tudo; o modelo só as vê no final, uma única vez.
2. **Cinco simulados** (validação cruzada, KFold 5) nas 1.168 casas de estudo: cada parte vira teste uma vez; a nota é a média dos 5.
3. **9 candidatos, mesmas regras:** chute (mediana), regressão linear (Ridge), Random Forest e Gradient Boosting, com variações; os ajustes de cada um (hiperparâmetros) testados por GridSearchCV nos mesmos simulados. Nota: **RMSLE** (erro relativo ao preço).
4. **Ferramentas:** Python + scikit-learn; tratamento e modelo num único **pipeline**, o mesmo arquivo usado pela API.

### 8. Etapa 3 · Por que o Gradient Boosting venceu (≈60 s)
Quanto cada tipo de modelo errou, **em média, nos simulados** (melhor versão de cada tipo):

| Modelo | Em palavras simples | Erro médio | Nota RMSLE |
|---|---|---|---|
| Chute | sempre o preço do meio | US$ 54.573 | 0,3907 |
| Regressão linear | uma "linha reta" | US$ 18.575 | 0,1482 |
| Random Forest | árvores lado a lado | US$ 17.404 | 0,1378 |
| **Gradient Boosting** | árvores em fila | **US$ 16.747** | **0,1325** |

- O grande salto é sair do chute; entre os modelos de verdade a diferença é pequena, e isso é dito com honestidade.
- **Como ele aprende:** avaliadores em fila; o 1º dá um palpite, o 2º corrige o erro do 1º, e assim por diante (500 pequenas correções; learning_rate = 0.03, max_leaf_nodes = 15).
- **Por que venceu:** (1) menor erro entre os 9 candidatos; (2) **decora menos**: a Random Forest erra US$ 6.524 nas casas que estudou e US$ 17.404 nas novas; o vencedor, US$ 10.008 → US$ 16.747; (3) **bom senso**: mais área ou qualidade nunca baixam o preço (restrição monotônica).
- **Para o professor:** Em 20 partições extras, o vencedor ganhou 16: **empate técnico**. (contra a versão sem as regras de bom senso).

### 9. Etapa 4 · Provar: as 292 casas nunca vistas (≈45 s)
- **70%** das casas com erro de até 10% ("7 em cada 10"); 89% com até 20%.
- **Erro típico: 6,1%** (metade das casas erra menos que isso).
- **Erro médio: US$ 16.298**, contra US$ 59.568 do chute (≈3,7× menor).
- **R² = 0,91**: explica cerca de 91% das diferenças de preço. RMSLE 0,1367 (v1: 0,1390), perto da validação cruzada: os simulados não foram otimistas.
- Ler o gráfico: cada ponto é uma casa; na linha tracejada o modelo acertaria em cheio.

### 10. Etapa 4 · Onde ele erra, e por quê (≈45 s)
1. **Nos extremos, puxa para o meio:** superestima a faixa mais barata em média US$ 6.879 e subestima a mais cara em US$ 10.758 (regressão à média). Por quê: há poucas casas muito baratas ou muito caras para aprender; na dúvida, fica perto do "normal".
2. **Vendas fora do normal:** casa **Id 582**, vendida ainda em construção (`Partial`) por US$ 253.293; o modelo estimou US$ 425.313 (+68%). Por quê: o modelo sabe como a casa é, não como a venda aconteceu.
3. **Por isso, uma faixa e não só um número:** −13% a +15%; nas casas da prova o preço real caiu dentro dela em **79,1%** (meta: 80%).

### 11. Etapa 4 · Sem a área, o erro mais que dobra (≈40 s)
- O teste: **esconder uma informação de cada vez** (embaralhar a coluna entre as casas da prova) e medir o erro típico de novo. Quanto mais o erro sobe, mais o modelo depende dela.
- Com todas as informações: erro típico **6,1%**.
- Sem: Área construída (sem porão): **13,7%** (casas com erro até 10%: 38%)
- Sem: Qualidade geral do acabamento: **10,8%** (casas com erro até 10%: 47%)
- Sem: Área do porão: **9,0%** (casas com erro até 10%: 55%)
- Sem: Bairro (Ames, Iowa): **8,4%** (casas com erro até 10%: 57%)
- Sem: Área do terreno: **7,8%** (casas com erro até 10%: 60%)
- Por que faz sentido: área = tamanho da casa; qualidade = custo de construção; bairro = localização (a mesma casa típica: US$ 155.524 em Edwards, US$ 202.659 em Stone Brook). Quartos quase não mudam nada: a área já diz o tamanho.
- Isso mostra **do que o modelo depende**, não causa. Números de `python -m ml.importance_plain` (`artifacts/metrics/importance_plain.json`).

## Parte 3 · O sistema (CRISP-DM: implantação)

### 12. Etapa 5 · Como as peças se conectam (≈45 s)
- **Feito uma vez, no desenvolvimento:** CSV do Kaggle → **pipeline de ML** (`ml/`, Python + scikit-learn) → **modelo treinado** (`.joblib` + metadados com hash SHA-256, métricas e gráficos em `artifacts/`).
- **A cada clique, no uso:** a **interface web** (`frontend/`, HTML/CSS/JS + Plotly) envia as 14 características → a **API FastAPI** (`backend/`) confere os valores, converte m² → pés², roda o modelo e devolve **preço, faixa de erro e casas parecidas vendidas**.
- A API carrega o modelo uma vez quando liga, confere o hash e **nunca treina** durante o uso (H4).

### 13. Etapa 5 · O que a pessoa vê: 3 abas (≈20 s)
**Simular um imóvel** (abre aqui) · **Imóveis reais** (o modelo calcula agora as 292 casas da prova e mostra Real × Modelo) · **Como o modelo funciona** (a história do treino).
- **Como sabemos que funciona:** 46 testes automáticos (pytest), um teste de ponta a ponta no navegador que clica nas 3 abas e a API que recusa o arquivo do modelo se o hash SHA-256 não conferir.

### 14. Demo ao vivo (≈2,5 min)
1. **Simular um imóvel:** mudar área construída, qualidade e bairro; a estimativa e a faixa típica mudam na hora.
   Descer até **Sua casa × casas reais vendidas**: características lado a lado e, para cada casa parecida, **Real | Modelo | Diferença**.
2. **Imóveis reais:** "o modelo acabou de calcular as 292 casas de teste". Clicar num ponto verde (acerto) e mostrar o painel **Real × Modelo**.
   Filtrar **Erro grande** e abrir o Id 582: mostrar o aviso de venda fora do normal.
3. Opcional: **Abrir esta casa no simulador** e depois **Voltar para a casa que você montou**.

## Parte 4 · A resposta

### 15. Voltando à pergunta (≈55 s)
- **Fechar a história com a casa da capa:** Casa real da prova (Id 434, bairro Gilbert, 149 m², 3 quartos, construída em 1997, vendida em 2008): vendida por US$ 181.000; o modelo estimou US$ 184.160 (1,7%). Com a faixa típica, o sistema diria "provavelmente entre US$ 160.781 e US$ 211.785": o preço real caiu dentro.
- **Dá para estimar? Sim, e sabendo o quanto confiar:** com 14 informações simples, erro de até 10% em 70% das casas nunca vistas, e uma faixa de erro que se confirmou (79,1% para 80% prometidos).
- **H1** ✔ as características explicam ~91% das diferenças de preço (R² 0,91).
- **H2** ✔ as árvores (erro médio US$ 17.404 e US$ 16.747) erraram menos que a "linha reta" (US$ 18.575).
- **H3** ✔ os tipos de algoritmo diferem; entre as duas melhores versões, empate técnico.
- **H4** ✔ o sistema usa o mesmo arquivo do treino (pipeline), sem mudar nada.
- Limites: uma cidade dos EUA, 2006–2010; não é laudo de avaliação. Próximos passos: dados locais e atuais, tipo e data da venda como entradas, faixas de erro por tipo de casa.

### 16. Obrigado! Perguntas?
Deixar na tela durante as perguntas: 1.460 vendas reais · 7 em cada 10 casas com erro até 10% · erro médio US$ 16.298 (chute: US$ 59.568) · 79,1% das casas dentro da faixa. Respostas prontas para as perguntas mais prováveis: `docs/HouseValue_Guia_da_Apresentacao.pdf`, parte 4.
