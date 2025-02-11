import numpy as np
import random
import math

# Parâmetros do Q-learning
alpha = 0.1  # Taxa de aprendizado
gamma = 0.9  # Fator de desconto
tau = 1.0    # Temperatura para Softmax

# Definição do espaço de estados (discretização pode ser ajustada conforme necessário)
STATE_SPACE = {} #n é usado
Q_table = {}  # Dicionário para armazenar os valores Q

# Número de qualidades possíveis
num_qualities = 20

# Função Softmax para escolher a ação
def softmax_selection(state):
    if state not in Q_table:
        Q_table[state] = np.zeros(num_qualities)  # Inicializa Q(s, a) se não existir, cria um array de 20 zeros no dicionário Q_table com chave = state.
    
    q_values = Q_table[state] #passa o array de 20 valores Q (um p/ cada ação) pra variavel q_values
    exp_q = np.exp(q_values / tau)
    probabilities = exp_q / np.sum(exp_q)
    return np.random.choice(range(num_qualities), p=probabilities) #seleciona através da política softmax

# Atualização da Q-table (fazer na ss_response)
def update_q_table(state, action, reward, next_state):
    if next_state not in Q_table:
        Q_table[next_state] = np.zeros(num_qualities) # Inicializa Q(s, a) se não existir, cria um array de 20 zeros no dicionário Q_table com chave = state.
    
    best_next_action = np.argmax(Q_table[next_state])
    Q_table[state][action] += alpha * (reward + gamma * Q_table[next_state][best_next_action] - Q_table[state][action])

# Função para discretizar o estado
# Aqui podemos definir níveis para buffer, largura de banda e oscilações
# A discretização pode ser refinada para melhor desempenho

def discretize_state(buffer, buffer_change, quality, bandwidth, osc_length, osc_depth):
    return (round(buffer, 1), round(buffer_change, 1), quality, round(bandwidth, 1), osc_length, osc_depth)

# Loop de treinamento (simulação)
for episode in range(1000):  # Número de iterações de aprendizado
    # Obter estado inicial do ambiente
    buffer = random.uniform(0, 10)  # Simulação de preenchimento do buffer
    buffer_change = random.uniform(-5, 5)  # Simulação de variação do buffer
    quality = random.randint(0, num_qualities - 1)  # Qualidade atual
    bandwidth = random.uniform(0, 5000)  # Simulação de largura de banda
    osc_length = random.randint(0, 30)  # Comprimento da oscilação
    osc_depth = random.randint(0, num_qualities - 1)  # Profundidade da oscilação
    
    state = discretize_state(buffer, buffer_change, quality, bandwidth, osc_length, osc_depth)
    
    # Selecionar ação usando Softmax
    action = softmax_selection(state)
    
    # Simular execução da ação (download do segmento)
    # Aqui você pode adicionar sua lógica para medir recompensa real
    reward = random.uniform(-1, 1)  # Exemplo de recompensa aleatória
    
    # Obter novo estado após baixar o segmento!!!
    buffer += random.uniform(-1, 1)
    buffer_change = random.uniform(-5, 5)
    bandwidth = random.uniform(0, 5000)
    osc_length = random.randint(0, 30)
    osc_depth = random.randint(0, num_qualities - 1)
    
    next_state = discretize_state(buffer, buffer_change, action, bandwidth, osc_length, osc_depth)
    
    # Atualizar a Q-table
    update_q_table(state, action, reward, next_state)

# Uso do agente para selecionar qualidade no DASH Client
# Supondo que temos os valores reais do ambiente:
def select_quality(buffer, buffer_change, quality, bandwidth, osc_length, osc_depth):
    state = discretize_state(buffer, buffer_change, quality, bandwidth, osc_length, osc_depth)
    action = softmax_selection(state)
    select.qi(action)  # Chama a função para baixar o segmento na qualidade escolhida
