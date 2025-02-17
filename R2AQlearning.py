# ALUNO: FELIPE PERONE - MATRÍCULA: 200017705 GRUPO: 13
from r2a.ir2a import IR2A
from player.parser import *
import time
import numpy as np
import random
import math


class R2AQlearning(IR2A):
  def __init__(self, id):
    IR2A.__init__(self, id)
    self.throughputs = [] #lista com as vazões de rede
    self.qi = [] #lista com as qualidades
    self.request_time = [] #lista p/ armazenar o tempo em que acontece um request
    self.quality_lista_1 = [] #lista pra salvar as qualidades requisitadas
    self.seg_num = 0 #contador p/ puxar o número de segmento atual
    self.state_space = [] #passa valores de state de ss_request p/ ss_response
    self.action_space = [] #passa valores de action de ss_request p/ ss_response
    self.Q_table = {}  # Dicionário para armazenar os valores Q
    self.osc_list = [] #lista que salva o núm de segmento em que ocorreu uma oscilação
    pass

  def handle_xml_request(self, msg):
     self.request_time = time.perf_counter() #inicia contagem de tempo da requisição xml
     self.send_down(msg)

  def handle_xml_response(self, msg):
    parsed_mpd = parse_mpd(msg.get_payload()) 
    self.qi = parsed_mpd.get_qi() #obtém lista de qualidade
    t = (time.perf_counter() - self.request_time)/2
    self.throughputs.append(msg.get_bit_length()/t) #salvo a vazão na lista
    self.send_up(msg)


  def handle_segment_size_request(self,msg):
    num_qualities = len(self.qi) #num_qualities = num de qualidades
    Bfmax = self.whiteboard.get_max_buffer_size() #Bfmax = capacidade maxima do buffer
    # Parâmetros do Q-learning
    alpha = 0.3  # Taxa de aprendizado
    gamma = 0.95  # Fator de desconto
    tau = 1.0    # Temperatura para Softmax
    if self.seg_num == 0: #se for o primeiro segmento
      # Loop de treinamento (simulação)
      for episode in range(100000):  # Número de iterações de aprendizado
          # Obter estado inicial do ambiente
          bufferfilling = random.uniform(0, Bfmax)  # Simulação de preenchimento do buffer
          buffer_change = random.uniform(-Bfmax + bufferfilling, Bfmax - bufferfilling)  # Simulação de variação do buffer
          quality = random.randint(0,num_qualities-1)  # Qualidade atual ESTRATEGIA RELACIONAR À BANDWIDTH
          bandwidth = random.choice(self.qi)  # Simulação de largura de banda
          osc_length = random.randint(0, 60)  # Comprimento da oscilação
          osc_depth = random.randint(0, 19)  # Profundidade da oscilação
          state = (round(bufferfilling, 1), round(buffer_change, 1), quality, round(bandwidth, 1), osc_length, osc_depth)
          # Selecionar ação usando Softmax
          if state not in self.Q_table:
            self.Q_table[state] = np.zeros(num_qualities)  # Inicializa Q(s, a) se não existir, cria um array de 20 zeros no dicionário Q_table com chave = state.
    
          q_values = self.Q_table[state] #passa o array de 20 valores Q (um p/ cada ação) pra variavel q_values
          exp_q = np.exp(q_values / tau)
          probabilities = exp_q / np.sum(exp_q)
          action = np.random.choice(range(num_qualities), p=probabilities) #seleciona através da política softmax
        
          # lógica para medir recompensa
          RQ = ((((quality - 1)/ num_qualities - 1) * 2) - 1)
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
          reward = 2*RQ + 1*RO + 4*RB + 3*RBC # Função Recompensa
          # Novo estado, simulando após baixar o segmento!!!
          bufferfilling = random.uniform(0, Bfmax)  # Simulação de preenchimento do buffer MUDAR p/ get buffer max
          buffer_change = random.uniform(-Bfmax + bufferfilling, Bfmax - bufferfilling)  # Simulação de variação do buffer MUDAR p/ get buffer max
          quality = random.randint(0,num_qualities-1) #quality = random.choice(self.qi)   Qualidade atual MUDAR AQ DPS
          bandwidth = random.choice(self.qi)  # Simulação de largura de banda
          osc_length = random.randint(0, 60)  # Comprimento da oscilação
          osc_depth = random.randint(0, 19)  # Profundidade da oscilação
          next_state = (round(bufferfilling, 1), round(buffer_change, 1), quality, round(bandwidth, 1), osc_length, osc_depth)
          # Atualizar a Q-table
          #update_q_table(state, action, reward, next_state)
          if next_state not in self.Q_table:
            self.Q_table[next_state] = np.zeros(num_qualities) # Inicializa Q(s, a) se não existir, cria um array de 20 zeros no dicionário Q_table com chave = state.
      
          best_next_action = np.argmax(self.Q_table[next_state])
          self.Q_table[state][action] += alpha * (reward + gamma * self.Q_table[next_state][best_next_action] - self.Q_table[state][action])
    #IMPLEMENTAÇÃO REAL Q-LEARNING
    #PROTOCOLO ABR
    #sabendo q len(Buffer_filling_lista) e len(quality_lista) == número de segmentos que foram reproduzidos até agr
    if self.seg_num == 0: #mesma hipótese do if anterior, coloquei aq só pra ficar mais claro que oq vem aqui pra baixo está fora do looping for
      tau = 0.50    # Temperatura para Softmax
      bufferfilling = 5  # Simulação de preenchimento do buffer
      buffer_change = 0  # Simulação de variação do buffer
      quality = 0  # Qualidade atual
      bandwidth_referencial = self.throughputs[0]  # Simulação de largura de banda
      osc_length = 0  # Comprimento da oscilação
      osc_depth = 0  # Profundidade da oscilação
      if bandwidth_referencial <= self.qi[0]:
        bandwidth = self.qi[0]
      elif bandwidth_referencial >= self.qi[len(self.qi)-1]:
        bandwidth = self.qi[len(self.qi)-1]
      else:
        for i in range(len(self.qi)):
          if bandwidth_referencial >= self.qi[i]:
            pass
          else:
            bandwidth = self.qi[i-1]
            break
      #estado = (bufferfilling, buffer_change, quality, bandwidth, osc_length, osc_depth)
      quality = self.qi.index(bandwidth) # Qualidade atual
      #def select_quality(buffer, buffer_change, quality, bandwidth, osc_length, osc_depth):
      state = (bufferfilling, buffer_change, quality, bandwidth, osc_length, osc_depth)
      #action = softmax_selection(state)
      #def softmax_selection(state):
      if state not in self.Q_table:
        self.Q_table[state] = np.zeros(num_qualities)  # Inicializa Q(s, a) se não existir, cria um array de 20 zeros no dicionário Q_table com chave = state.
    
      q_values = self.Q_table[state] #passa o array de 20 valores Q (um p/ cada ação) pra variavel q_values
      exp_q = np.exp(q_values / tau)
      probabilities = exp_q / np.sum(exp_q)
      action = np.random.choice(range(num_qualities), p=probabilities) #seleciona através da política softmax
      msg.add_quality_id(self.qi[action]) #adiciona a qualidade na msg
      self.quality_lista_1.append(action) #adiciona a ação na lista de histórico de qualidade
      self.state_space.append(state) #adc o estado atual na lista p/ passar pro handle response
      self.action_space.append(action) #adc o ação atual na lista p/ passar pro handle response
    else: #self.seg_num > 0
      buffer_filling_lista = self.whiteboard.get_playback_buffer_size()  
      bufferfilling = buffer_filling_lista[-1][1] #pega o preenchimento mais atual pra ser o do state
      if len(buffer_filling_lista) >= 2:
        buffer_change = bufferfilling - buffer_filling_lista[-2][1]
      else:
        buffer_change = 0
      #quality_lista = self.whiteboard.get_playback_qi() #n usa
      quality = self.quality_lista_1[-1] #Determinação da qualidade
      bandwidth_referencial = self.throughputs[self.seg_num]
      if bandwidth_referencial <= self.qi[0]: #bandwidth = discretização da vazão atual (maior valor em self.qi que é menor que bandwidth_referencial)
        bandwidth = self.qi[0]
      elif bandwidth_referencial >= self.qi[len(self.qi)-1]:
        bandwidth = self.qi[len(self.qi)-1]
      else:
        for i in range(len(self.qi)):
          if bandwidth_referencial >= self.qi[i]:
            pass
          else:
            bandwidth = self.qi[i-1]
            break     
      if len(self.quality_lista_1) < 2:
        #caso a lista tenha menos de 2 elementos não tem oscilação!
        osc_length = 0  # Comprimento da oscilação
        osc_depth = 0  # Profundidade da oscilação
      else:
        if self.quality_lista_1[-1] < self.quality_lista_1[-2]:
          self.osc_list.append(self.seg_num)
          if len(self.osc_list) < 2:
            #caso a lista tenha menos de 2 elementos não tem oscilação!
            osc_length = 0  # Comprimento da oscilação
            osc_depth = 0  # Profundidade da oscilação
          else:
            ol = self.osc_list[-1] - self.osc_list[-2]
            if ol >= 60:
              osc_length = 0
              osc_depth = 0
            else:
              osc_length = ol
              osc_depth = self.quality_lista_1[-2] - self.quality_lista_1[-1]
        else: 
          osc_length = 0
          osc_depth = 0
      #action = select_quality(bufferfilling, buffer_change, quality, bandwidth, osc_length, osc_depth)
      #def select_quality(buffer, buffer_change, quality, bandwidth, osc_length, osc_depth):
      #aqui determina o estado
      state = (round(bufferfilling, 1), round(buffer_change, 1), quality, round(bandwidth, 1), osc_length, osc_depth)
      if state not in self.Q_table:
        self.Q_table[state] = np.zeros(num_qualities)  # Inicializa Q(s, a) se não existir, cria um array de 20 zeros no dicionário Q_table com chave = state.
    
      q_values = self.Q_table[state] #passa o array de 20 valores Q (um p/ cada ação) pra variavel q_values
      exp_q = np.exp(q_values / tau)
      probabilities = exp_q / np.sum(exp_q)
      action = np.random.choice(range(num_qualities), p=probabilities) #seleciona através da política softmax
      msg.add_quality_id(self.qi[action])
      self.quality_lista_1.append(action)
      self.state_space.append(state)
      self.action_space.append(action)
    #msg.add_quality_id()
    #FIM DO PROTOCOLO ABR NO REQUEST
    self.seg_num += 1 #soma 1 no contador de segmento
    self.request_time = time.perf_counter()
    self.send_down(msg)
  def handle_segment_size_response(self,msg):
    Bfmax = self.whiteboard.get_max_buffer_size() 
    state = self.state_space[0]
    action = self.action_space[0]
    num_qualities = len(self.qi)
    t= (time.perf_counter() - self.request_time)/2
    self.throughputs.append(msg.get_bit_length()/t)
    alpha = 0.3  # Taxa de aprendizado
    gamma = 0.95  # Fator de desconto
    tau = 0.50    # Temperatura para Softmax
    #FEEDBACK PROTOCOLO ABR
    #state = (round(bufferfilling, 1), round(buffer_change, 1), quality, round(bandwidth, 1), osc_length, osc_depth)
    bufferfilling = self.state_space[0][0]
    buffer_change = self.state_space[0][1]
    quality = self.state_space[0][2]
    bandwidth = self.state_space[0][3]
    osc_length = self.state_space[0][4]
    osc_depth = self.state_space[0][5]
    
    # lógica para medir recompensa
    RQ = ((((quality - 1)/ num_qualities - 1) * 2) - 1)
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
    if self.seg_num == 1: #verifica se response do segnum é igual a zero (como a soma é no final do ss_request estamos defasados de 1 aqui no response!)
      #caso seja é necessário pular, uma vez que as listas estão vazias
      #valores são iguais ao seg_num do request, que foram importados através do self.state_space()
      pass
    else:
      bufferfilling = buffer_filling_lista[-1][1]
      if len(buffer_filling_lista) >= 2:
        buffer_change = bufferfilling - buffer_filling_lista[-2][1]
      else:
        buffer_change = 0
      quality_lista = self.whiteboard.get_playback_qi()
      quality = (self.quality_lista_1[-1]) #verificar
      bandwidth_referencial = self.throughputs[self.seg_num]
      if bandwidth_referencial <= self.qi[0]:
        bandwidth = self.qi[0]
      elif bandwidth_referencial >= self.qi[len(self.qi)-1]:
        bandwidth = self.qi[len(self.qi)-1]
      else:
        for i in range(len(self.qi)):
          if bandwidth_referencial >= self.qi[i]:
            pass
          else:
            bandwidth = self.qi[i-1]
            break
      if len(self.quality_lista_1) < 2:
        #caso a lista tenha menos de 2 elementos não tem oscilação!
        osc_length = 0  # Comprimento da oscilação
        osc_depth = 0  # Profundidade da oscilação
      else:
        if self.quality_lista_1[-1] < self.quality_lista_1[-2]:
          self.osc_list.append(self.seg_num)
          if len(self.osc_list) < 2:
            #caso a lista tenha menos de 2 elementos não tem oscilação!
            osc_length = 0  # Comprimento da oscilação
            osc_depth = 0  # Profundidade da oscilação
          else:
            ol = self.osc_list[-1] - self.osc_list[-2]
            if ol >= 60:
              osc_length = 0
              osc_depth = 0
            else:
              osc_length = ol
              osc_depth = self.quality_lista_1[-2] - self.quality_lista_1[-1]
        else: 
          osc_length = 0
          osc_depth = 0

    #FIM DO FEEDBACK
    next_state = (round(bufferfilling, 1), round(buffer_change, 1), quality, round(bandwidth, 1), osc_length, osc_depth)
    #update_q_table(state, action, reward, next_state)
    if next_state not in self.Q_table: 
      self.Q_table[next_state] = np.zeros(num_qualities) # Inicializa Q(s, a) se não existir, cria um array de 20 zeros no dicionário Q_table com chave = state.
      
    best_next_action = np.argmax(self.Q_table[next_state]) #melhor ação no prox estado (next_state)
    self.Q_table[state][action] += alpha * (reward + gamma * self.Q_table[next_state][best_next_action] - self.Q_table[state][action]) #formula Q learning
    self.state_space.clear() #esvazia a lista state space p/ prox execuções
    self.action_space.clear() #esvaliza a lista action space p/ prox execuções
    #final do handle_segment_size_response
    self.send_up(msg)

  def initialize(self):
    pass

  def finalization(self):
    pass
