# Documentação — HouseValue AI

Leia nesta ordem. Todos os arquivos abaixo são **gerados a partir dos artefatos** (os números não são digitados à mão); este índice é o único escrito à mão.

| # | Documento | O que responde | Gerado por |
|---|---|---|---|
| 1 | [data_audit.md](data_audit.md) | Como a base é de verdade: arquivos, hashes, colunas, ausências estruturais × desconhecidas, a armadilha do `"None"` em `MasVnrType`, inconsistências e casas atípicas. | `python -m ml.inspect_data` |
| 2 | [crisp_dm.md](crisp_dm.md) | O projeto inteiro nas 6 etapas do CRISP-DM, cada uma apontando para a evidência (código, artefato, teste). | `python -m ml.build_docs` |
| 3 | [methodology.md](methodology.md) | O protocolo: holdout de 20% separado antes de tudo, validação cruzada em 5 partes, pré-processamento dentro do pipeline, métricas e regra de seleção. | `python -m ml.build_docs` |
| 4 | [model_evaluation.md](model_evaluation.md) | Os resultados: as 9 configurações, o holdout, CV × holdout, maiores erros, importâncias, v1 × v2, teste de empate, curva de aprendizado, erro por segmento e faixa típica de erro. | `python -m ml.build_docs` |
| 5 | [model_card.md](model_card.md) | Ficha do modelo em produção: entradas, métricas, usos apropriados e não recomendados, limitações. | `python -m ml.build_docs` |
| 6 | [presentation.md](presentation.md) | Roteiro de fala slide a slide (7 a 10 min + demo ao vivo), com os números reais e os passos da demo. | `python -m ml.build_docs` |
| 7 | [HouseValue_Guia_da_Apresentacao.pdf](HouseValue_Guia_da_Apresentacao.pdf) | **Para ler antes de apresentar:** o projeto explicado do começo ao fim, imagem de cada slide com o ponto-chave e a fala sugerida (≈9 min + demo), perguntas prováveis do professor com respostas, glossário e checklist do dia. | gerado a partir dos artefatos e do deck |

Figuras usadas nos documentos e nos slides: `../artifacts/figures/` (`eda_*` da análise exploratória, `eval_*` da avaliação). Convenção de cores igual à da interface: laranja = preço real, azul = estimativa do modelo, verde / âmbar / vermelho = situação do erro.
