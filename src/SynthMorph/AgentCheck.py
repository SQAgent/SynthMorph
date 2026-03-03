from SynthMorph.Tools.debugtool import save_graph_image
from SynthMorph.graph import build_elastic_matrix_graph


agent = build_elastic_matrix_graph()
save_graph_image(graph=agent, filename="graph.png")
