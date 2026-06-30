"""Conteúdo educacional do software (fonte única de verdade dos textos).

`VARIABLES`: o que é cada variável do freMTPL2, o que significa e como entra no
modelo. `CONCEITOS`: os blocos de metodologia (negócio, prêmio, GLM, GBM, …).
Separar o conteúdo do código é boa prática: a UI consome estes dicionários.
"""
from __future__ import annotations

# Para cada variável: (o que é) + (como entra/por que importa no pricing).
VARIABLES: dict[str, dict[str, str]] = {
    "VehPower": {
        "o_que": "Potência do veículo (categoria ordinal).",
        "no_modelo": "Carros mais potentes tendem a maior frequência; entra como fator categórico, cada nível com sua relatividade.",
    },
    "VehAge": {
        "o_que": "Idade do veículo, em anos.",
        "no_modelo": "Veículos muito novos (valor alto) e muito velhos têm padrões de risco distintos — efeito não-linear que o GBM capta melhor.",
    },
    "DrivAge": {
        "o_que": "Idade do condutor principal, em anos.",
        "no_modelo": "Jovens têm frequência bem maior (curva em U com a idade). É um dos drivers mais fortes.",
    },
    "BonusMalus": {
        "o_que": "Índice bônus-malus (50 a 230). Abaixo de 100 = bônus (bom histórico, desconto); acima de 100 = malus (sinistrou, agravo).",
        "no_modelo": "É o sinal de risco individual mais forte em auto — concentra o histórico de sinistralidade da pessoa.",
    },
    "Density": {
        "o_que": "Densidade populacional da região do condutor (hab/km²).",
        "no_modelo": "Áreas densas (urbanas) → mais trânsito e exposição → maior frequência de sinistros.",
    },
    "VehBrand": {
        "o_que": "Marca/categoria do veículo (B1–B14, anonimizadas).",
        "no_modelo": "Proxy do tipo de carro e do perfil de quem o dirige; cada marca tem sua relatividade.",
    },
    "VehGas": {
        "o_que": "Combustível: Diesel ou Regular (gasolina).",
        "no_modelo": "Diesel costuma indicar maior quilometragem anual → mais exposição ao risco.",
    },
    "Area": {
        "o_que": "Código de área (A–F) ordenado por densidade urbana.",
        "no_modelo": "Resumo geográfico de urbanização, correlacionado com a frequência.",
    },
    "Region": {
        "o_que": "Região administrativa da França (R11–R94).",
        "no_modelo": "Captura diferenças geográficas de risco (tráfego, clima, roubo) que as demais variáveis não pegam.",
    },
}

# Blocos de metodologia exibidos na seção "Entenda o modelo".
CONCEITOS: list[dict[str, str]] = [
    {
        "titulo": "1. O que é seguro e prêmio",
        "texto": "Seguro é transferência de risco: o cliente paga um valor pequeno e certo (o prêmio) "
                 "e a seguradora assume um prejuízo grande e incerto (o sinistro). O negócio se sustenta "
                 "pela mutualização — muitos pagam, poucos sinistram. Como o custo do sinistro só se revela "
                 "depois, precificar é estimar um custo que ainda não existe.",
    },
    {
        "titulo": "2. Prêmio puro = Frequência × Severidade",
        "texto": "O custo esperado de sinistros decompõe-se em quantas vezes a pessoa sinistra por ano "
                 "(frequência) e quanto custa cada sinistro (severidade). Separar é mais preciso: os fatores "
                 "que mudam a frequência (idade, bônus-malus) são diferentes dos que mudam a severidade "
                 "(valor do veículo). Prêmio puro = frequência × severidade.",
    },
    {
        "titulo": "3. O que é um GLM (e por que Poisson/Gamma)",
        "texto": "O GLM (Modelo Linear Generalizado) é o padrão atuarial: prevê com a distribuição certa "
                 "(Poisson para contagem de sinistros, Gamma para custo) e é interpretável por construção — "
                 "cada variável vira um multiplicador (relatividade). Regulador exige preço justificável, "
                 "por isso o GLM é a base.",
    },
    {
        "titulo": "4. O que é um GBM (e por que SHAP)",
        "texto": "O GBM (gradient boosting, árvores) é mais preciso: capta não-linearidades e interações "
                 "que o GLM linear achata. Mas é opaco. O SHAP abre a caixa-preta e mostra, para cada "
                 "previsão, quanto cada variável empurrou o risco — permitindo auditar se o modelo se apoia "
                 "em fatores legítimos.",
    },
    {
        "titulo": "5. Como ler o resultado",
        "texto": "À esquerda, o GLM mostra a decomposição: base da carteira × fator de cada variável = "
                 "frequência prevista (acima de 1 encarece, abaixo barateia). À direita, o SHAP faz o mesmo "
                 "para o GBM. Frequência × severidade = prêmio puro. Comparar GLM e GBM é o trade-off "
                 "precisão × interpretabilidade no centro do pricing.",
    },
]
