from scipy.io import mmread
import networkx as nx
from networkx.utils import reverse_cuthill_mckee_ordering
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import time

# calcula a largura de banda da matriz


def calculate_bandwidth(matrix):
    length = matrix.shape[0]
    array = []

    for i in range(length):
        row = matrix.getrow(i)
        if row.nonzero():
            minim = (i - min(row.nonzero()[1]))
            array.append(minim)

    return(max(array))

# salva as imagens das matrizes


def save_image(matrix, filename, flag, state):
    plt.spy(matrix)
    plt.savefig("./images/" + flag + "/" +
                filename + "_" + state + ".png")
    plt.cla()

# escreve os tempos de processamento de cada matriz em um arquivo


def write_file(path, array):
    myfile = open(path, 'w')

    for line in array:
        myfile.write(str(line))
        myfile.write('\n')

    myfile.close()

# função que realiza a redução da largura de banda da matriz


def reduce_bandwidth(path, file, flag):

    file_name = file + ".mtx"

    init_time = time.time()
    matrix = mmread(path + file_name)

    def check_simmetric(M):
        a = M.transpose()
        if (np.allclose(M.A, a.A)):
            return True
        else:
            return False

    save_image(matrix, file.split(".")[0], flag, "inicial")

    if (check_simmetric(matrix) == False):
        matrix += matrix.transpose()

    # Transforma a matriz lida no file em um grafo compatível com a função rcm
    graph = nx.from_scipy_sparse_matrix(matrix)

    def smallest_degree(G):
        return min(G, key=G.degree)

    rcm = list(reverse_cuthill_mckee_ordering(
        graph, heuristic=smallest_degree))

    # converte o grafo de volta para matriz de adjacencias
    solution = nx.to_scipy_sparse_matrix(graph, nodelist=rcm)

    end_time = time.time()

    save_image(solution, file.split(".")[0], flag, "reduzido")

    return {file: (end_time - init_time, calculate_bandwidth(matrix), calculate_bandwidth(solution))}


def main():
    simetric_files = ['1138_bus', 'ash85', 'G21',
                      'G45', 'G59', 'GD06_theory']

    assimetric_files = ['GD95_a', 'gre_1107',
                        'gre_115', 'gre_185', 'CSphd', 'cage8']

    simetric_time_results = []
    assimetric_time_results = []

    for file in simetric_files:
        simetric_time_results.append(reduce_bandwidth(
            './inputs/simetric/', file, "simetric"))

    for file in assimetric_files:
        assimetric_time_results.append(reduce_bandwidth(
            './inputs/assimetric/', file, "assimetric"))

    write_file("./output/simetric.txt", simetric_time_results)
    write_file("./output/assimetric.txt", assimetric_time_results)


if __name__ == "__main__":
    main()
