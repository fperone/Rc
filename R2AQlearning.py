from r2a.ir2a import IR2A
import time

class R2AQlearning(IR2A):
  def __init__(self, id):
    self.throughputs = []
    self.qi = []
    self.request_time = []
    pass

  def handle_xml_request(self, msg):
     self.request_time = time.perf_counter()
     self.send_down(msg)

  def handle_xml_response(self, msg)
    parsed_mpd = parse_mpd(msg.get_payload())
    self.qi = parsed_mpd.get_qi()
    t = (time.perf_counter() - self.request_tine)/2
    self.throughputs.append(msg.get_bit_length()/t)
    self.send_up(msg)


  def handle_segment_size_request(self,msg):
    self.request_time = time.perf_counter()
    #PROTOCOLO ABR
    #msg.add.quality_id()
    #FIM DO PROTOCOLO ABR NO REQUEST

  def handle_segment.size_response(self,msg):
    t= (time.perf_counter() - self.request_time)/2
    self.throughputs.append(msg.get_bit_length()/t)
    #FEEDBACK PROTOCOLO ABR
    #FIM DO FEEDBACK
    self.send_up(msg)

  def initialize(self):
    pass

  def finalization(self):
    pass
