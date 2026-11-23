# Product

## Register

product

## Users
Estudantes e professor da disciplina TK0231 (UFC) testando o modelo de Machine Learning, cada um no próprio notebook, com 10 a 20 minutos para explorar. Também a pessoa que apresenta o projeto e precisa mostrar o modelo funcionando. Conhecem pouco do mercado de Ames, Iowa, e das 79 variáveis originais.

## Product Purpose
Aplicação para testar um modelo de regressão que estima o preço histórico de venda (USD, Ames, Iowa, 2006–2010) a partir de 14 características do imóvel. O sucesso é a pessoa entender, mexendo nos controles, como o modelo reage, onde acerta e onde erra, e quais são os limites dos dados. Indicadores, comparação de modelos e análise dos dados apoiam esse entendimento.

## Brand Personality
Um software de computador comum e bem feito, como um aplicativo de desktop que alguém usa no PC: janela com barra lateral, painéis, barra de status, controles nativos e previsíveis. Três palavras: prático, claro, confiável. Sem tom de marketing.

## Anti-references
- Template SaaS genérico: grades de cards idênticos com ícone, gradientes roxos, números gigantes de "hero metric".
- Também evitar cara de portal imobiliário (fotos, banners de anúncio).

## Design Principles
1. Testar é a tarefa principal: abrir o app e já poder mexer no imóvel e ver o preço reagir.
2. Mostrar, com dados reais, o comportamento do modelo (curvas de sensibilidade, imóveis reais do holdout com preço real), nunca números ilustrativos.
3. Honestidade sobre limites: USD histórico, Ames, faixas do treino e erro médio sempre visíveis, sem intervalo de confiança inventado.
4. Comportamento de aplicativo: estados claros (carregando, erro, sem conexão), controles que respondem na hora, nada escondido atrás de modais.

## Accessibility & Inclusion
WCAG 2.1 AA: contraste adequado, foco visível, navegação completa por teclado nos controles próprios (sliders, steppers, combobox, segmentados) com papéis ARIA corretos, rótulos associados, significado nunca só por cor, respeito a `prefers-reduced-motion`. Funcionar em notebook primeiro, mas sem quebrar em tablet e celular.
