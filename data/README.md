# Dados — origem e política

## Origem
- Base: **House Prices: Advanced Regression Techniques** (Kaggle) — https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data
- Indicada no documento da disciplina `Bases de dados - Projetos de ML.docx`, seção *Regressão*.
- Arquivo original encontrado nesta máquina: `house-prices-advanced-regression-techniques.zip`, na pasta Downloads do usuário (nenhum ZIP dentro do workspace). Os quatro CSV/TXT na raiz do workspace têm SHA-256 idêntico aos membros do ZIP. Hashes completos em `data/source_manifest.json`.

## Pastas
- `raw/` — cópia de trabalho extraída do ZIP por `python -m ml.inspect_data`. Os arquivos existentes nunca são sobrescritos (se já existirem, o hash é comparado e registrado).
- `processed/` — reservado. O pipeline não persiste dados transformados: toda transformação é feita dentro do `Pipeline` do scikit-learn, a partir de `raw/`.
- `source_manifest.json` — busca do ZIP, tamanhos, hashes e ação tomada para cada arquivo.

## Política
- O ZIP original e os arquivos da raiz do workspace não são modificados.
- Nada de dentro do ZIP é executado; apenas bytes são lidos.
- Dados brutos não devem ser versionados (ver `.gitignore`); para reproduzir, basta ter o ZIP e rodar `python -m ml.inspect_data`.
- Leitura dos CSVs: só o token `NA` é ausente (`keep_default_na=False, na_values=["NA"]`), porque `"None"` é categoria válida de `MasVnrType`.

## Observação de ambiente
Na primeira tentativa de leitura, `pd.read_csv("train.csv")` falhou com `ParserError: ... out of memory` enquanto o disco C: estava com ~0,1 GB livre. Depois de liberar espaço (limpeza do cache do pip, autorizada pelo usuário), o mesmo arquivo foi lido normalmente (1460 × 81). O carregador do projeto lê os CSVs como bytes a partir de `data/raw/`.
