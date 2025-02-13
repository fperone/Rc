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
      pass
  # Função para discretizar o estado
  # Aqui podemos definir níveis para buffer, largura de banda e oscilações
  # A discretização pode ser refinada para melhor desempenho

  def discretize_state(buffer, buffer_change, quality, bandwidth, osc_length, osc_depth):
    return (round(buffer, 1), round(buffer_change, 1), quality, round(bandwidth, 1), osc_length, osc_depth)


  # Uso do agente para selecionar qualidade no DASH Client
  # Supondo que temos os valores reais do ambiente:
  def select_quality(buffer, buffer_change, quality, bandwidth, osc_length, osc_depth):
    state = discretize_state(buffer, buffer_change, quality, bandwidth, osc_length, osc_depth)
    action = softmax_selection(state)
    return action  # Chama a função para baixar o segmento na qualidade escolhida
