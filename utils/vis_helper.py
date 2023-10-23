from graphviz import Digraph

def graphviz_draw_smalldarts(ops, concat, name=""):
  prefix = name + "_"
  nls = [
    "c_{k-2}", "c_{k-1}", "0", "1", "2", "3", "c_{k}",
  ]
  id_to_nl = [prefix + nl for nl in nls]

  dot = Digraph(name="cluster" + name)

  dot.attr(label=name, labelloc="t", fontsize="16")

  dot.attr('node', style='filled')
  dot.attr('edge', arrowhead='empty')
  dot.attr(rankdir='LR')

  dot.attr('node', shape='box', fillcolor='#C6F0B7')
  dot.node(id_to_nl[0], label=nls[0])
  dot.node(id_to_nl[1], label=nls[1])
  dot.attr('node', shape='square', fillcolor='#BCD6E5',)
  dot.node(id_to_nl[2], label=nls[2])
  dot.node(id_to_nl[3], label=nls[3])
  dot.node(id_to_nl[4], label=nls[4])
  dot.node(id_to_nl[5], label=nls[5])
  dot.attr('node', shape='box', fillcolor='#EBEBAF')
  dot.node(id_to_nl[6], label=nls[6])

  twice_dst = 4
  for op, src in ops:
    dot.edge(id_to_nl[src], id_to_nl[twice_dst // 2], label=op)
    twice_dst += 1

  for src in concat:
    dot.edge(id_to_nl[src], id_to_nl[-1], label="")

  return dot


def graphviz_draw_darts(genotype, label="", view=True, print_source=True):
  normal = graphviz_draw_smalldarts(genotype.normal, genotype.normal_concat, name="Normal")
  reduce = graphviz_draw_smalldarts(genotype.reduce, genotype.reduce_concat, name="Reduce")

  graph = Digraph()
  graph.attr(rankdir='LR')
  graph.subgraph(normal)
  graph.subgraph(reduce)

  graph.attr(label=label, labelloc="t", fontsize="24")

  if print_source:
    print(graph.source)
  file_name = label
  if not file_name:
    file_name = "tmp"
  graph.render(f'render/{file_name}.gv', view=view)



