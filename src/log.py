class Logger():
  def __init__(self, file):
    with open(file, 'w') as f:
      pass
    self.file = file
  def __call__(self, dict):
    with open(self.file, 'a') as f:
      for k, v in dict.items():
        f.write(str(k)+":"+str(v)+"   ")
      f.write("\n")
