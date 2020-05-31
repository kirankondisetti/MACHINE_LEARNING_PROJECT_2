import re

def export_graph(file_in, file_out):
  f_in = open(file_in, "r")
  f_out = open(file_out, "w")

  nodes = []
  labels = []

  lines = f_in.readlines()

  for line in lines:
    node = re.search(r'^"(.*?)" .* label="(.*?)"', line)
    if (node != None):
      nodes.append(node.group(1))
      labels.append(node.group(2))

  for node in nodes:
    maps = []
    children = []
    for x in lines:
      map = re.search(r'^(.*?) -> (.*?) .* label="(.*?)"', x)
      if(map != None) and (map.group(1) == node):
        children.append(map.group(2))
        maps.append(map.group(3))

    for i, child in enumerate(children):
      f_out.write(labels[int(node)]+" "+ maps[i] + "\n")


  print("Ha")