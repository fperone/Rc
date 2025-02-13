from r2a.ir2a import IR2A
from player.parser import *
import time
import numpy as np
import random
import math

class R2AQlearning(IR2A):
  def __init__(self, id):
    self.throughputs = []
    self.qi = []
    self.request_time = []
    # Parâmetros do Q-learning
    #alpha = 0.1  # Taxa de aprendizado
    #gamma = 0.9  # Fator de desconto
    #tau = 1.0    # Temperatura para Softmax

    # Definição do espaço de estados (discretização pode ser ajustada conforme necessário)
    #MUDAR, BOTAR SELF. NESSES AQ
    self.seg_num = 0
    #STATE_SPACE = {} #n é usado
    self.Q_table = {}  # Dicionário para armazenar os valores Q
    #num_qualities = len(self.qi)
    pass

  def handle_xml_request(self, msg):
     self.request_time = time.perf_counter()
     self.send_down(msg)

  def handle_xml_response(self, msg):
    parsed_mpd = parse_mpd(msg.get_payload())
    self.qi = parsed_mpd.get_qi()
    t = (time.perf_counter() - self.request_time)/2
    self.throughputs.append(msg.get_bit_length()/t)
    Bfmax = self.whiteboard.get_max_buffer_size() 
    self.send_up(msg)


  def handle_segment_size_request(self,msg):
    num_qualities = len(self.qi)
    # Parâmetros do Q-learning
    alpha = 0.1  # Taxa de aprendizado
    gamma = 0.9  # Fator de desconto
    tau = 1.0    # Temperatura para Softmax
    if self.seg_num == 0:
      # Loop de treinamento (simulação)
      for episode in range(1000):  # Número de iterações de aprendizado
          # Obter estado inicial do ambiente
          bufferfilling = random.uniform(0, Bfmax)  # Simulação de preenchimento do buffer
          buffer_change = random.uniform(-Bfmax + bufferfilling, Bfmax - bufferfilling)  # Simulação de variação do buffer
          quality = random.choice(self.qi)  # Qualidade atual MUDAR TALVEZ random.randint(self.qi)
          bandwidth = random.randint(20000, 5000000)  # Simulação de largura de banda
          osc_length = random.randint(0, 60)  # Comprimento da oscilação
          osc_depth = random.randint(0, 19)  # Profundidade da oscilação
          state = [(round(bufferfilling, 1), round(buffer_change, 1), quality, round(bandwidth, 1), osc_length, osc_depth)]
          # Selecionar ação usando Softmax
          #aqui
          if state not in self.Q_table:
            self.Q_table[state] = np.zeros(num_qualities)  # Inicializa Q(s, a) se não existir, cria um array de 20 zeros no dicionário Q_table com chave = state.
    
          q_values = self.Q_table[state] #passa o array de 20 valores Q (um p/ cada ação) pra variavel q_values
          exp_q = np.exp(q_values / tau)
          probabilities = exp_q / np.sum(exp_q)
          action = np.random.choice(range(num_qualities), p=probabilities) #seleciona através da política softmax
          # Simular execução da ação (download do segmento)
          # lógica para medir recompensa
          N = self.qi.index(quality)
          RQ = ((((quality - 1)/ N - 1) * 2) - 1)
          if osc_length and osc_depth != 0:
            RO = (-1/osc_length**(2/osc_depth)) + ((osc_length - 1)/((60 - 1)* (60 **(2/osc_depth))))
          else:
            RO = 0
          if bufferfilling <= (0.1 * Bfmax):
            RB = -1
          else:
            RB = (2 * bufferfilling)/((1-0.1) * Bfmax) - ((1 + 0.1)/(1 - 0.1))
          bufferfilling_anterior = bufferfilling - buffer_change
          if buffer_change <= 0:
            RBC = buffer_change/bufferfilling_anterior
          else:
            RBC = buffer_change/(bufferfilling - (bufferfilling_anterior/2))
          reward = 2*RQ + 1*RO + 4*RB + 3*RBC # Exemplo de recompensa aleatória
          # Novo estado após baixar o segmento!!!
          bufferfilling = random.uniform(0, Bfmax)  # Simulação de preenchimento do buffer MUDAR p/ get buffer max
          buffer_change = random.uniform(-Bfmax + bufferfilling, Bfmax - bufferfilling)  # Simulação de variação do buffer MUDAR p/ get buffer max
          quality = random.choice(self.qi)  # Qualidade atual
          bandwidth = random.randint(20000, 5000000)  # Simulação de largura de banda
          osc_length = random.randint(0, 60)  # Comprimento da oscilação
          osc_depth = random.randint(0, 19)  # Profundidade da oscilação
          next_state = [round(bufferfilling, 1), round(buffer_change, 1), quality, round(bandwidth, 1), osc_length, osc_depth]
          # Atualizar a Q-table
          #update_q_table(state, action, reward, next_state)
          if next_state not in self.Q_table:
            self.Q_table[next_state] = np.zeros(num_qualities) # Inicializa Q(s, a) se não existir, cria um array de 20 zeros no dicionário Q_table com chave = state.
      
          best_next_action = np.argmax(self.Q_table[next_state])
          self.Q_table[state][action] += alpha * (reward + gamma * self.Q_table[next_state][best_next_action] - self.Q_table[state][action])
    #IMPLEMENTAÇÃO REAL Q-LEARNING
    #PROTOCOLO ABR
    #suposição que len(Buffer_filling_lista) e len(quality_lista) == número de segmentos que foram reproduzidos até agr
    if self.seg_num == 0:
      bufferfilling = 5  # Simulação de preenchimento do buffer
      buffer_change = 0  # Simulação de variação do buffer
      quality = 0  # Qualidade atual
      bandwidth = self.throughputs[0]  # Simulação de largura de banda
      osc_length = 0  # Comprimento da oscilação
      osc_depth = 0  # Profundidade da oscilação
      #action = select_quality(bufferfilling, buffer_change, quality, bandwidth, osc_length, osc_depth)
      #def select_quality(buffer, buffer_change, quality, bandwidth, osc_length, osc_depth):
      state = [bufferfilling, buffer_change, quality, bandwidth, osc_length, osc_depth)
      #action = softmax_selection(state)
      #def softmax_selection(state):
      if state not in self.Q_table:
        self.Q_table[state] = np.zeros(num_qualities)  # Inicializa Q(s, a) se não existir, cria um array de 20 zeros no dicionário Q_table com chave = state.
    
      q_values = self.Q_table[state] #passa o array de 20 valores Q (um p/ cada ação) pra variavel q_values
      exp_q = np.exp(q_values / tau)
      probabilities = exp_q / np.sum(exp_q)
      action = np.random.choice(range(num_qualities), p=probabilities) #seleciona através da política softmax
      msg.add_quality_id(action)
    else:
      buffer_filling_lista = self.whiteboard.get_playback_buffer_size() 
      bufferfilling = buffer_filling_lista[-1][1]
      if len(buffer_filling_lista) >= 2:
        buffer_change = buffer_filling - buffer_filling_lista[-2][1]
      else:
        buffer_change = 0
      quality_lista = self.whiteboard.get_playback_qi()
      quality = quality_lista[-1][1]
      bandwidth = self.throughputs[self.seg_num]
      if len(quality_lista) < 2:
        osc_length = 0  # Comprimento da oscilação
        osc_depth = 0  # Profundidade da oscilação
      else:
        tempo_referencia = quality_lista[-1][0]  # Tempo da amostra mais recente
        qualidade_referencia = quality_lista[-1][1]  # Qualidade da amostra mais recente
        for i in range(len(quality_lista) - 2, -1, -1):  # Itera sobre as amostras anteriores
            tempo_amostra_i = quality_lista[i][0]
            qualidade_amostra_i = quality_lista[i][1]
            if tempo_referencia - tempo_amostra_i > 60.0:
              osc_length = 0
              osc_depth = 0
              break
            else:
              if qualidade_amostra_i != qualidade_referencia:
                osc_length = i + 1
                osc_depth = self.qi.index(qualidade_amostra_i) - self.qi.index(qualidade_referencia)
                break
              else:
                osc_length = 0
                osc_depth = 0
      #action = select_quality(bufferfilling, buffer_change, quality, bandwidth, osc_length, osc_depth)
      #def select_quality(buffer, buffer_change, quality, bandwidth, osc_length, osc_depth):
      #aqui
      state = [round(bufferfilling, 1), round(buffer_change, 1), quality, round(bandwidth, 1), osc_length, osc_depth]
      if state not in self.Q_table:
        self.Q_table[state] = np.zeros(num_qualities)  # Inicializa Q(s, a) se não existir, cria um array de 20 zeros no dicionário Q_table com chave = state.
    
      q_values = self.Q_table[state] #passa o array de 20 valores Q (um p/ cada ação) pra variavel q_values
      exp_q = np.exp(q_values / tau)
      probabilities = exp_q / np.sum(exp_q)
      action = np.random.choice(range(num_qualities), p=probabilities) #seleciona através da política softmax
      msg.add_quality_id(action)      
    #msg.add_quality_id()
    #FIM DO PROTOCOLO ABR NO REQUEST
    self.seg_num += 1
    self.request_time = time.perf_counter()
    self.send_down(msg)
  def handle_segment_size_response(self,msg):
    num_qualities = len(self.qi)
    t= (time.perf_counter() - self.request_time)/2
    self.throughputs.append(msg.get_bit_length()/t)
    #FEEDBACK PROTOCOLO ABR
    # lógica para medir recompensa
    N = self.qi.index(quality)
    RQ = ((((quality - 1)/ N - 1) * 2) - 1)
    if osc_length and osc_depth != 0:
      RO = (-1/osc_length**(2/osc_depth)) + ((osc_length - 1)/((60 - 1)* (60 **(2/osc_depth))))
    else:
      RO = 0
    if bufferfilling <= (0.1 * Bfmax):
      RB = -1
    else:
      RB = (2 * bufferfilling)/((1-0.1) * Bfmax) - ((1 + 0.1)/(1 - 0.1))
    bufferfilling_anterior = bufferfilling - buffer_change
    if buffer_change <= 0:
      RBC = buffer_change/bufferfilling_anterior
    else:
      RBC = buffer_change/(bufferfilling - (bufferfilling_anterior/2))
    reward = 2*RQ + 1*RO + 4*RB + 3*RBC # Exemplo de recompensa aleatória

    buffer_filling_lista = self.whiteboard.get_playback_buffer_size() 
    bufferfilling = buffer_filling_lista[-1][1]
    if len(buffer_filling_lista) >= 2:
      buffer_change = buffer_filling - buffer_filling_lista[-2][1]
    else:
      buffer_change = 0
      quality_lista = self.whiteboard.get_playback_qi()
      quality = quality_lista[-1][1]
      bandwidth = self.throughputs[self.seg_num]
    if len(quality_lista) < 2:
      osc_length = 0  # Comprimento da oscilação
      osc_depth = 0  # Profundidade da oscilação
    else:
      tempo_referencia = quality_lista[-1][0]  # Tempo da amostra mais recente
      qualidade_referencia = quality_lista[-1][1]  # Qualidade da amostra mais recente
      for i in range(len(quality_lista) - 2, -1, -1):  # Itera sobre as amostras anteriores
          tempo_amostra_i = quality_lista[i][0]
          qualidade_amostra_i = quality_lista[i][1]
          if tempo_referencia - tempo_amostra_i > 60.0:
            osc_length = 0
            osc_depth = 0
            break
          else:
            if qualidade_amostra_i != qualidade_referencia:
              osc_length = i + 1
              osc_depth = self.qi.index(qualidade_amostra_i) - self.qi.index(qualidade_referencia)
              break
            else:
              osc_length = 0
              osc_depth = 0
    #FIM DO FEEDBACK
    next_state = [round(bufferfilling, 1), round(buffer_change, 1), quality, round(bandwidth, 1), osc_length, osc_depth]
    #update_q_table(state, action, reward, next_state)
    if next_state not in self.Q_table:
      self.Q_table[next_state] = np.zeros(num_qualities) # Inicializa Q(s, a) se não existir, cria um array de 20 zeros no dicionário Q_table com chave = state.
      
    best_next_action = np.argmax(self.Q_table[next_state])
    self.Q_table[state][action] += alpha * (reward + gamma * self.Q_table[next_state][best_next_action] - self.Q_table[state][action])
    self.send_up(msg)

  def initialize(self):
    pass

  def finalization(self):
    pass
