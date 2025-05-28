# Controle de Pêndulo Invertido

Este projeto implementa a simulação e controle de um pêndulo invertido utilizando diferentes técnicas de controle fuzzy (FIS, Neuro-Fuzzy, Genetic-Fuzzy) com interface gráfica interativa.

## Funcionalidades

- **Simulação física realista** do pêndulo invertido, baseada nas equações diferenciais completas do sistema (incluindo momento de inércia, massa do carrinho, etc).
- **Interface gráfica intuitiva** para ajuste de todos os parâmetros físicos e de controle.
- **Três tipos de controladores fuzzy**: FIS, Neuro-Fuzzy e Genetic-Fuzzy.
- **Visualização em tempo real** do comportamento do pêndulo e do carrinho.

## Parâmetros Ajustáveis

Na interface, você pode ajustar todos os parâmetros relevantes do sistema físico:

- **Massa do pêndulo (`mass`)**: massa do bastão do pêndulo.
- **Comprimento do pêndulo (`length`)**: comprimento do bastão do pêndulo.
- **Massa do carrinho (`cart_mass`)**: massa do carrinho que se move na horizontal.
- **Gravidade (`gravity`)**: aceleração gravitacional do ambiente.
- **Momento de inércia (`inertia`)**: pode ser definido manualmente ou calculado automaticamente como \( m_p \cdot l^2 \) se deixado em zero.
- **Passo de tempo (`dt`)**: intervalo de tempo de cada iteração da simulação.

Além disso, cada controlador possui seus próprios parâmetros ajustáveis (ganho, número de regras, taxa de aprendizado, etc).

## Tipos de Controladores

- **FIS (Fuzzy Inference System):** Utiliza regras fuzzy clássicas para determinar a força de controle com base no ângulo e velocidade angular do pêndulo.
- **Neuro-Fuzzy:** Combina redes neurais e lógica fuzzy, permitindo ajuste automático dos parâmetros fuzzy via aprendizado.
- **Genetic-Fuzzy:** Utiliza algoritmos genéticos para otimizar as regras e parâmetros do sistema fuzzy, buscando melhor desempenho de controle.

## Como Usar

1. **Instale as dependências** (exemplo para ambiente Python):
    ```bash
    pip install -r requirements.txt
    ```

2. **Execute a aplicação**:
    ```bash
    python src/gui/main_window.py
    ```

3. **Ajuste os parâmetros** no painel lateral:
    - Modifique massa, comprimento, massa do carrinho, gravidade, inércia e passo de tempo conforme desejado.
    - Escolha o tipo de controlador e ajuste seus parâmetros específicos.

4. **Inicie a simulação** clicando em "Iniciar".
    - Você pode pausar, resetar ou ajustar parâmetros a qualquer momento.

5. **Visualize o comportamento** do pêndulo e do carrinho em tempo real no gráfico.

## Exemplo de Execução

```bash
pip install -r requirements.txt
python src/gui/main_window.py
```

Ajuste os parâmetros desejados e clique em "Iniciar" para visualizar a simulação.

## Resultados Esperados

Abaixo, um exemplo de gráfico gerado pela simulação (adicione uma imagem real do seu projeto se possível):

![Exemplo de Simulação](exemplo_simulacao.png)

## Equações Utilizadas

A simulação utiliza as equações diferenciais completas do pêndulo invertido:

\[
\ddot{x} = \frac{m_p l [\dot{\theta}^2 \sin(\theta) - \ddot{\theta} \cos(\theta)] + F}{m_c + m_p}
\]
\[
\ddot{\theta} = \frac{m_p l [g \sin(\theta) - \ddot{x} \cos(\theta)]}{I + m_p l^2}
\]

Onde:
- \( m_p \): massa do pêndulo
- \( m_c \): massa do carrinho
- \( l \): comprimento do pêndulo
- \( I \): momento de inércia
- \( F \): força aplicada
- \( g \): gravidade
- \( \theta \): ângulo do pêndulo
- \( x \): posição do carrinho

## Estrutura do Projeto

- `src/simulation/pendulum_sim.py`: Simulação física do pêndulo invertido.
- `src/gui/main_window.py`: Interface gráfica e integração dos controladores.
- `src/controllers/`: Implementação dos controladores FIS, Neuro-Fuzzy e Genetic-Fuzzy.

## Requisitos

- Python 3.7+
- PyQt5
- numpy
- matplotlib
- scikit-fuzzy
- torch (para Neuro-Fuzzy)

## Autores

- Alberto Zilio
- Lucas Steffens
- Roni Pereira

## Licença

Este projeto é distribuído sob a licença MIT.

## Referências

- Documento base: PDF do trabalho fornecido pelo professor.
- Literatura sobre controle fuzzy e pêndulo invertido:
    - KATAYAMA, T. Subspace Methods for System Identification. Springer, 2005.
    - ZADEH, L. A. Fuzzy sets. Information and Control, 1965.
    - Outros materiais utilizados no desenvolvimento do projeto.

