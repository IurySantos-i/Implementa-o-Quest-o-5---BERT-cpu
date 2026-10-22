# Q5 — Fundamentação e previsões para o experimento

Texto de apoio à seção 2 do relatório. As previsões abaixo são inferências para Adult; não são resultados experimentais. As quatro fontes primárias foram consultadas em 13/10/2026.

## O que os quatro trabalhos sustentam

**Ma et al. (2024), *Rewrite the Stars*.** O produto de duas projeções afins contém interações quadráticas sem materializar todos os produtos entre atributos. Nos experimentos de visão, substituir a soma entre dois ramos pela multiplicação melhora a acurácia, com ganhos maiores em redes estreitas. A extrapolação testável é que o ganho Star–Plain em Adult seja positivo nas menores larguras e diminua nas maiores. Isso não é garantido: o artigo compara blocos de visão, incluindo convoluções, profundidade e normalização, e sua ablação soma/produto preserva dois ramos. A comparação de Q5 usa uma camada simples mais larga; portanto muda também a parametrização. A análise motiva medir eficiência, mas não estabelece a superioridade de Star em dados censitários nem em NumPy float64. [Fonte primária, §§3.1, 3.4 e 4.3](https://arxiv.org/html/2403.19967v1).

**Shazeer (2020), *GLU Variants Improve Transformer*.** ReGLU usa exatamente o formato ReLU(projeção) multiplicada por outra projeção; GEGLU e SwiGLU substituem a ativação. Nos experimentos em T5, as duas últimas obtêm melhor perplexidade, mantendo aproximadamente os parâmetros e a computação. Para isso, o trabalho reduz a largura interna para dois terços: são três matrizes no FFN com porta, contra duas no original. A previsão para a comparação de GLU, GEGLU e SwiGLU deste estudo é que a escolha da ativação altere a relação acurácia/custo. O artigo não sustenta prever uma ordem universal entre essas variantes em Adult: seu treinamento usa textos, arquitetura encoder–decoder e um orçamento muito maior. A identidade algébrica entre Star-ReLU e ReGLU também impede tratá-las como duas arquiteturas independentes. [Fonte primária, §§2–3](https://arxiv.org/html/2002.05202v1).

**Dauphin et al. (2017), *Language Modeling with Gated Convolutional Networks*.** A GLU combina uma projeção linear com uma porta sigmoide. O ramo linear oferece um caminho de gradiente sem passar pela derivada de uma ativação adicional nesse ramo, e os experimentos de linguagem favorecem GLU em relação a alternativas avaliadas. Isso motiva observar convergência e estabilidade, além da acurácia final, na comparação com GLU incluída neste estudo. Não prevê que GLU necessariamente supere ReLU em uma MLP rasa: ReLU também permite passagem do gradiente nas unidades ativas. A vantagem de paralelizar posições de uma sequência em relação a redes recorrentes não se transfere diretamente para a comparação entre duas MLPs tabulares. [Fonte primária, §§2–3 e 5.2–5.3](https://arxiv.org/html/1612.08083v3).

**Grinsztajn, Oyallon e Varoquaux (2022), *Why do tree-based models still outperform deep learning on tabular data?*.** O estudo compara modelos em 45 bases, considerando busca de hiperparâmetros, e associa vantagens de árvores a funções irregulares, atributos pouco informativos e orientação das colunas. A previsão relevante é cautelosa: introduzir produtos pode ajudar certas interações, mas não resolve automaticamente esses três problemas. As ablações deste estudo comparam a vantagem Star–Plain usando apenas atributos contínuos e apenas indicadores categóricos. Uma vantagem maior no segundo caso seria compatível com a explicação por interações, sem demonstrá-la causalmente. Sem treinar uma árvore no mesmo protocolo, Q5 não permite concluir que Star reduz a distância até árvores. O trabalho também não estabelece uma lei sobre toda base tabular ou todo tamanho amostral. [Fonte primária, §§4–6](https://arxiv.org/html/2207.08815v1).

## Derivação própria: o alcance das interações

Considere a entrada aumentada com viés na primeira posição, \(\tilde x=(1,x_1,\ldots,x_d)^\top\), \(u=a^\top\tilde x\) e \(v=b^\top\tilde x\), com índices \(0,\ldots,d\) e \(a_0,b_0\) representando os vieses. Sem ativação,

\[
s=uv=\sum_i a_i b_i\tilde x_i^2+
\sum_{i<j}(a_i b_j+a_j b_i)\tilde x_i\tilde x_j.
\]

O produto é um polinômio de grau no máximo dois, que inclui termos lineares e constantes pelo viés. Entretanto, seus coeficientes são compartilhados: a matriz quadrática simétrica \((ab^\top+ba^\top)/2\) tem posto no máximo dois. Ter monômios implícitos não equivale a aprender livremente todos os coeficientes de um polinômio quadrático.

Com a ativação de Q5, \(s=\operatorname{ReLU}(u)v\), o resultado é **quadrático por partes**, e não um único polinômio quadrático global. Na região \(u>0\), vale a expansão anterior; na região \(u<0\), a saída é zero. Fora da fronteira \(u=0\),

\[
\frac{\partial s}{\partial x_j}
=\mathbf{1}_{u>0}a_jv+\operatorname{ReLU}(u)b_j,
\qquad
\frac{\partial^2s}{\partial x_i\partial x_j}
=\mathbf{1}_{u>0}(a_i b_j+a_j b_i).
\]

Essa derivada mista pode ser diferente de zero, mas não precisa ser para toda entrada ou todo par de pesos. Na fronteira, usa-se a convenção de subgradiente da implementação; verificações por diferenças finitas devem evitar esse ponto. Uma MLP ReLU simples também representa interações funcionais por meio de regiões e fronteiras de ativação; a diferença é que seus logits são afins dentro de cada região, enquanto os logits de uma camada Star podem ser quadráticos ali.

Para indicadores one-hot, \(x_i^2=x_i\). Duas categorias mutuamente exclusivas da mesma variável satisfazem \(x_ix_j=0\). Logo, nem todos os monômios acrescentam informação: os candidatos úteis incluem categorias de variáveis diferentes e produtos entre atributos contínuos e categóricos. Isso é uma justificativa algébrica para a ablação proposta, não evidência de que o modelo a explorou.

## Derivação própria: o que significa igualar o custo

A igualdade \(h\leftrightarrow2h\) aproxima o custo das projeções de entrada. Ela não iguala exatamente uma rede completa. Para uma única camada escondida, entrada de dimensão \(d\), saída com \(c\) logits e viés incluído nas matrizes, contando duas operações por multiplicação-acumulação, o custo matricial por exemplo é

\[
C_{\rm star}=4(d+1)h+2(h+1)c,
\quad
C_{\rm plain}=4(d+1)h+2(2h+1)c.
\]

A saída Plain custa \(2hc\) operações a mais. Para igualar somente essas multiplicações, sua largura seria

\[
m=h\frac{2(d+1)+c}{d+1+c},
\]

antes do arredondamento. Ativações, produto dos ramos, retropropagação e atualização dos parâmetros ainda precisam ser contabilizados. Redes com camadas adicionais exigem uma nova soma de custos. Portanto, o protocolo deve preservar os pares pedidos, apresentar os FLOPs efetivamente contados para cada modelo e informar a diferença relativa de custo. Um ajuste de largura pode ser uma análise adicional; não deve substituir silenciosamente o pareamento solicitado.

FLOPs não são segundos: BLAS, tamanho das matrizes, acesso à memória e custos do interpretador interferem no tempo em CPU. Resultados float64 de uma MLP pequena não reproduzem a eficiência de implementações dos artigos em outros regimes numéricos e de hardware.

## Como transformar as leituras em afirmações refutáveis

O [protocolo anterior aos experimentos](protocolo.md) registra a hipótese principal: **“Para h=32 e h=64, estrela-ReLU superará o denso-2h em pelo menos 0,3 ponto percentual de acurácia média de teste, com separação dos intervalos média ± desvio-padrão amostral.”** O limiar e a regra de separação são escolhas operacionais do estudo, não valores estabelecidos pelos artigos. A observação do comportamento em h=16 e h=128 também responde à pergunta sobre largura, sem substituir esse critério principal.

O mesmo protocolo registra a hipótese mecanística secundária: **“A vantagem estrela−denso será maior com somente indicadores categóricos que com somente atributos contínuos.”** As variantes GLU, GEGLU e SwiGLU e as duas ablações de atributos integram o desenho experimental. A previsão de diminuição do ganho com a largura, motivada por Ma et al., é uma expectativa da literatura; não acrescenta um critério primário concorrente. Um resultado inconclusivo entre três sementes não demonstra equivalência; um resultado negativo no orçamento escolhido não demonstra incapacidade representacional da arquitetura.

## Referências

1. MA, Xu; DAI, Xiyang; BAI, Yue; WANG, Yizhou; FU, Yun. *Rewrite the Stars*. CVPR, 2024. [arXiv:2403.19967](https://arxiv.org/abs/2403.19967).
2. SHAZEER, Noam. *GLU Variants Improve Transformer*. 2020. [arXiv:2002.05202](https://arxiv.org/abs/2002.05202).
3. DAUPHIN, Yann N.; FAN, Angela; AULI, Michael; GRANGIER, David. *Language Modeling with Gated Convolutional Networks*. ICML, 2017. [arXiv:1612.08083](https://arxiv.org/abs/1612.08083).
4. GRINSZTAJN, Léo; OYALLON, Edouard; VAROQUAUX, Gaël. *Why do tree-based models still outperform deep learning on tabular data?* NeurIPS, 2022. [arXiv:2207.08815](https://arxiv.org/abs/2207.08815).
