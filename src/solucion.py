#!/usr/bin/env python3
"""
Solución al problema "El Convoy" con varios enfoques:
- Fuerza bruta (exacto para k=2 y grafos pequeños)
- Greedy secuencial (heurístico)
- Greedy aleatorizado (heurístico)

Modelo: grafo dirigido con pesos en aristas.
"""
from typing import Dict, List, Tuple, Optional
import heapq
import random

Graph = Dict[int, List[Tuple[int, float]]]


def dijkstra(graph: Graph, s: int, t: int, banned_edges: Optional[set] = None) -> Optional[Tuple[float, List[Tuple[int, int, int]]]]:
    """Camino más corto de s a t evitando aristas en banned_edges.
    Cada arista se identifica por triple (u, v, idx) donde idx es el índice en la lista de adyacencia de u.
    Retorna (distancia, lista de aristas con ids) o None si no hay camino.
    """
    if banned_edges is None:
        banned_edges = set()
    dist = {node: float('inf') for node in graph}
    prev_node = {node: None for node in graph}
    prev_edge_idx = {node: None for node in graph}
    dist[s] = 0.0
    pq = [(0.0, s)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        if u == t:
            break
        for idx, (v, w) in enumerate(graph.get(u, [])):
            edge_id = (u, v, idx)
            if edge_id in banned_edges:
                continue
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                prev_node[v] = u
                prev_edge_idx[v] = idx
                heapq.heappush(pq, (nd, v))
    if dist[t] == float('inf'):
        return None
    # reconstruir camino con ids de aristas
    path_nodes = []
    cur = t
    while cur is not None:
        path_nodes.append(cur)
        cur = prev_node[cur]
    path_nodes.reverse()
    edges = []
    for i in range(len(path_nodes) - 1):
        u = path_nodes[i]
        v = path_nodes[i + 1]
        idx = prev_edge_idx[v]
        edges.append((u, v, idx))
    return dist[t], edges


def path_cost(graph: Graph, edges: List[Tuple[int, int, int]]) -> float:
    cost = 0.0
    for u, v, idx in edges:
        x, w = graph[u][idx]
        # assert x == v
        cost += w
    return cost


def greedy_k_edge_disjoint(graph: Graph, s: int, t: int, k: int) -> Optional[List[List[Tuple[int, int, int]]]]:
    """Encuentra k caminos disjuntos en aristas seleccionando repetidamente el camino más corto disponible."""
    banned = set()
    paths = []
    for _ in range(k):
        res = dijkstra(graph, s, t, banned)
        if res is None:
            return None
        dist, edges = res
        paths.append(edges)
        # prohibir aristas usadas
        for e in edges:
            banned.add(e)
    return paths


def randomized_greedy(graph: Graph, s: int, t: int, k: int, trials: int = 50, noise: float = 0.05) -> Optional[List[List[Tuple[int, int, int]]]]:
    """Aplica múltiples ejecuciones greedy con perturbaciones aleatorias en pesos.
    Devuelve el conjunto con menor máximo de costo."""
    best_paths = None
    best_max = float('inf')
    # crear copia mutable de pesos
    base_graph = {u: list(adj) for u, adj in graph.items()}
    for _ in range(trials):
        # aplicar ruido multiplicativo pequeño
        perturbed = {}
        for u, adj in base_graph.items():
            pert = []
            for v, w in adj:
                factor = 1.0 + random.uniform(-noise, noise)
                pert.append((v, max(0.0, w * factor)))
            perturbed[u] = pert
        paths = greedy_k_edge_disjoint(perturbed, s, t, k)
        if paths is None:
            continue
        costs = [path_cost(graph, p) for p in paths]
        m = max(costs)
        if m < best_max:
            best_max = m
            best_paths = paths
    return best_paths


def brute_force_k2(graph: Graph, s: int, t: int) -> Optional[Tuple[List[Tuple[int, int, int]], List[Tuple[int, int, int]]]]:
    """Fuerza bruta para k=2:
    - enumera todos los caminos simples de s a t con backtracking
    - selecciona dos disjuntos en aristas minimizando el máximo costo
    Advertencia: explosivo en grafos medianos.
    """
    all_paths = []
    def dfs(u: int, visited_edges: set, path_nodes: List[int]):
        if u == t:
            # construir aristas
            edges = []
            for i in range(len(path_nodes)-1):
                uu = path_nodes[i]
                vv = path_nodes[i+1]
                # buscar índice de la arista usada entre uu->vv que no esté marcada
                for idx, (x, _) in enumerate(graph[uu]):
                    if x == vv:
                        e_id = (uu, vv, idx)
                        if e_id in visited_edges:
                            continue
                        edges.append(e_id)
                        break
            all_paths.append(edges)
            return
        for idx, (v, _) in enumerate(graph.get(u, [])):
            e = (u, v, idx)
            if e in visited_edges:
                continue
            visited_edges.add(e)
            path_nodes.append(v)
            dfs(v, visited_edges, path_nodes)
            path_nodes.pop()
            visited_edges.remove(e)
    dfs(s, set(), [s])
    best = None
    best_max = float('inf')
    n = len(all_paths)
    for i in range(n):
        for j in range(i+1, n):
            p1 = all_paths[i]
            p2 = all_paths[j]
            # chequear disjunción en aristas
            set1 = set(p1)
            set2 = set(p2)
            if set1 & set2:
                continue
            c1 = path_cost(graph, p1)
            c2 = path_cost(graph, p2)
            m = max(c1, c2)
            if m < best_max:
                best_max = m
                best = (p1, p2)
    return best


if __name__ == "__main__":
    # Ejemplo de uso
    # Grafo pequeño en forma de etapas (para reproducir la reducción)
    # nodos: 0 -> 1 -> 2 -> 3
    # en cada etapa, dos aristas paralelas: peso a_i y 0
    graph: Graph = {
        0: [(1, 3.0), (1, 0.0)],
        1: [(2, 4.0), (2, 0.0)],
        2: [(3, 5.0), (3, 0.0)],
        3: []
    }
    s, t, k = 0, 3, 2
    print("Greedy secuencial:")
    paths = greedy_k_edge_disjoint(graph, s, t, k)
    if paths is None:
        print("No existen k caminos disjuntos")
    else:
        costs = [path_cost(graph, p) for p in paths]
        print("Caminos:", paths)
        print("Costos:", costs, "max:", max(costs))

    print("\nGreedy aleatorizado:")
    paths = randomized_greedy(graph, s, t, k, trials=50)
    if paths is None:
        print("No existen k caminos disjuntos")
    else:
        costs = [path_cost(graph, p) for p in paths]
        print("Caminos:", paths)
        print("Costos:", costs, "max:", max(costs))

    print("\nFuerza bruta k=2:")
    res = brute_force_k2(graph, s, t)
    if res is None:
        print("No existen dos caminos disjuntos")
    else:
        p1, p2 = res
        c1 = path_cost(graph, p1)
        c2 = path_cost(graph, p2)
        print("Camino 1:", p1, "costo:", c1)
        print("Camino 2:", p2, "costo:", c2)
        print("Max:", max(c1, c2))
