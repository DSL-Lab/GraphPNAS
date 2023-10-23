import torch.multiprocessing as mp

class Queues:
  """
  Wraps a bunch of queues together for easier use during NAS.
  """

  def __init__(self):
    self.gen_queue = mp.SimpleQueue()
    self.eva_queue = mp.SimpleQueue()
    self.log_queue = mp.SimpleQueue()

  def exit(self):
    self.push("__exit__")

  def push(self, message):
    self.push_gen(message)
    self.push_eva(message)
    self.push_log(message)

  def push_gen(self, message):
    self.gen_queue.put(message)

  def push_eva(self, message):
    self.eva_queue.put(message)

  def push_log(self, message):
    self.log_queue.put(message)


  def log(self, message):
    self.log_queue.put(("log", message))

  def pop_gen(self):
    return self.gen_queue.get()

  def pop_eva(self):
    return self.eva_queue.get()

  def pop_log(self):
    return self.log_queue.get()

