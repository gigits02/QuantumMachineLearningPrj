import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from qutip import *
from scipy.optimize import minimize
from tqdm import tqdm

# GRAFO
# Parametri
n_nodes = 3
edges = [(0, 1), (1, 2), (2, 0)]
graph = nx.Graph()
graph.add_nodes_from(range(n_nodes))
graph.add_edges_from(edges)
positions = nx.spring_layout(graph, seed=1)
#positions = nx.shell_layout(graph)
#positions = nx.kamada_kawai_layout(graph)
nx.draw(graph, positions, with_labels=True, node_color="lightgreen", edge_color="black", node_size=600)
plt.show()

# Trova il nodo con grado minimo
fixed_node = min(graph.degree, key=lambda x: x[1])[0]
fixed_color = 0
# Lavora con un nodo (qdit) in meno (fissato uno)
active_nodes = [i for i in range(n_nodes) if i != fixed_node]

#FOR K_COLORS IN RANGE(30) GIRERÀ CERCANDO IL NUM CROMATICO DEL GRAFO
for k_colors in range(2,30):

    # Stato iniziale: prodotto di stati uniformi
    def initial_state(n_nodes, d):
        psi0 = tensor([basis(d, 0) for _ in range(n_nodes)])
        hadamard_d = (1 / np.sqrt(d)) * sum([basis(d, i) for i in range(d)])
        return tensor([hadamard_d for _ in range(n_nodes)])

    # Hamiltoniano di costo: penalizza se due nodi hanno stesso colore
    def cost_hamiltonian(n_nodes, active_nodes, d, edges, fixed_node, fixed_color):
    
        H = 0
        for (i, j) in edges:
            
            # Caso 1: entrambi i nodi sono attivi (non fissati)
            if i != fixed_node and j != fixed_node:
                i_idx = active_nodes.index(i)
                j_idx = active_nodes.index(j)
                for c in range(d):
                    proj_i = basis(d, c) * basis(d, c).dag()
                    proj_j = basis(d, c) * basis(d, c).dag()
                    term = tensor([
                        proj_i if k == i_idx else
                        proj_j if k == j_idx else
                        qeye(d)
                        for k in range(n_nodes - 1)
                    ])
                    H += term

            # Caso 2: solo j è fisso
            elif i != fixed_node and j == fixed_node:
                i_idx = active_nodes.index(i)
                for c in range(d):
                    if c == fixed_color:
                        proj_i = basis(d, c) * basis(d, c).dag()
                        term = tensor([
                            proj_i if k == i_idx else qeye(d)
                            for k in range(n_nodes - 1)
                        ])
                        H += term

            # Caso 3: solo i è fisso
            elif i == fixed_node and j != fixed_node:
                j_idx = active_nodes.index(j)
                for c in range(d):
                    if c == fixed_color:
                        proj_j = basis(d, c) * basis(d, c).dag()
                        term = tensor([
                            proj_j if k == j_idx else qeye(d)
                            for k in range(n_nodes - 1)
                        ])
                        H += term

        return H

    # Mixer Hamiltonian: somma di permutazioni fra i livelli
    def mixer_hamiltonian(n_nodes, d):
        H = 0
        for i in range(n_nodes):
            for m in range(d):
                for n in range(m+1, d):
                    flip = basis(d, m) * basis(d, n).dag() + basis(d, n) * basis(d, m).dag()
                    op = tensor([flip if j == i else qeye(d) for j in range(n_nodes)])
                    H += op
        return H

    # Stato iniziale
    psi0 = initial_state(len(active_nodes), k_colors)

    # Hamiltoniani
    Hc = cost_hamiltonian(n_nodes, active_nodes, k_colors, edges, fixed_node, fixed_color)
    Hm = mixer_hamiltonian(len(active_nodes), k_colors)

    # Funzione costo: aspettazione dell'H_cost sullo stato finale
    loss_history = []

    def qaoa_expectation(params):
        depth = len(params) // 2
        gammas = params[:depth]
        alphas = params[depth:]
        psi = psi0
        for gamma, alpha in zip(gammas, alphas):
            psi = (-1j * gamma * Hc).expm() * psi
            psi = (-1j * alpha * Hm).expm() * psi
        expectation = expect(Hc, psi)
        loss_history.append(expectation)  # salva loss
        return expectation

    # Preparazione della barra di caricamento per l'ottimizzazione (con callback da "minimize")
    n_steps = 100  # massimo numero di iterazioni dell’ottimizzatore
    pbar = tqdm(total=n_steps, desc=f"Training Progress k={k_colors}...")

    def optimization_callback(xk):
        pbar.update(1)

    # Inizializzazione random
    depth = 1
    init_params = np.random.uniform(0, np.pi, 2 * depth)

    # Ottimizzazione
    result = minimize(
        qaoa_expectation,
        init_params,
        method='COBYLA',
        callback=optimization_callback,
        options={'maxiter': n_steps}
    )
    pbar.close()
    opt_params = result.x
    opt_expectation = result.fun

    # Stato finale con parametri ottimizzati
    final_state = psi0
    for gamma, alpha in zip(opt_params[:depth], opt_params[depth:]):
        final_state = (-1j * gamma * Hc).expm() * final_state
        final_state = (-1j * alpha * Hm).expm() * final_state

    # Probabilità finali
    probs = np.abs(final_state.full().flatten())**2

    # digitstring completa (include nodo fissato)
    def insert_fixed_digitstring(raw_digitstring, fixed_node, fixed_color, n_nodes):
        chunks = []
        active_index = 0
        for node in range(n_nodes):
            if node == fixed_node:
                chunks.append(str(fixed_color))
            else:
                chunks.append(str(raw_digitstring[active_index]))
                active_index += 1
        return ''.join(chunks)

    digitstrings = []
    for i in range(k_colors**len(active_nodes)):
        raw = np.base_repr(i, base=k_colors).zfill(len(active_nodes))
        full = insert_fixed_digitstring(raw, fixed_node, fixed_color, n_nodes)
        digitstrings.append(full)

    # Decodifica e analisi delle stringhe di output
    def decode_qudit_string(s):
        return {node: int(color) for node, color in enumerate(s)}

    def is_valid_coloring_qudit(assignment, edges):
        for u, v in edges:
            if assignment[u] == assignment[v]:
                return False
        return True

    threshold = max(probs) - 0.0001
    print("Bitstring | Assegnamento | Valido | Probabilità")
    print("-" * 50)

    valid_count = 0
    best_assignment = None

    for i, p in enumerate(probs):
        if p > threshold:
            s = digitstrings[i]
            assignment = decode_qudit_string(s)
            valid = is_valid_coloring_qudit(assignment, edges)
            print(f"{s} | {assignment} | {valid} | {p:.4f}")
            if valid:
                valid_count += 1
                best_assignment = assignment

    #Output: numero cromatico
    if valid_count > 0:
        print(f"Il numero minimo di colori per colorare il grafo è {k_colors}")
        break
    else:
        print(f"\nNessuna colorazione valida trovata con {k_colors} colori. Provo con {k_colors+1}...")        

# STAMPE E PLOTS

print("Valore minimo atteso H_cost:", opt_expectation)
print("Migliori parametri trovati:", opt_params)

# Plot convergenza funzione costo
plt.plot(loss_history)
plt.xlabel("Step")
plt.ylabel("Costo")
plt.title("Convergenza funzione costo")
plt.grid(True)
plt.show()

# Istogramma
plt.figure(figsize=(10, 4))
plt.bar(digitstrings, probs)
plt.xticks(rotation=90)
plt.xlabel("Stati")
plt.ylabel("Probabilità")
plt.title("Distribuzione delle probabilità - QAOA con qudit")
plt.grid(True)
plt.tight_layout()
plt.show()

# Visualizzazione grafo colorato
def plot_colored_graph(graph, assignment, positions, cmap=plt.cm.Set3):
    node_colors = [assignment[n] for n in graph.nodes]
    unique_colors = sorted(set(node_colors))
    
    n_colors = len(unique_colors)
    color_list = [cmap(i / max(1, n_colors - 1)) for i in range(n_colors)]
    color_map = {c: color_list[i] for i, c in enumerate(unique_colors)}
    final_colors = [color_map[c] for c in node_colors]

    plt.figure(figsize=(6, 4))
    nx.draw(
        graph,
        pos=positions,
        with_labels=True,
        node_color=final_colors,
        edge_color="black",
        node_size=800,
        font_color="black",
        font_weight="bold"
    )
    plt.title("Grafo colorato secondo l'assegnazione QAOA", fontsize=14)
    plt.axis('off')
    plt.show()

if best_assignment:
    plot_colored_graph(graph, best_assignment, positions)
