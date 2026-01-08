#!/usr/bin/env python3
"""
Módulo de experimentación para el problema "El Convoy".
Compara los diferentes enfoques algorítmicos en términos de:
- Calidad de la solución (valor máximo del camino más largo)
- Tiempo de ejecución
- Escalabilidad con el tamaño del grafo
"""

import time
import random
import statistics
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from solucion import (
    Graph,
    greedy_k_edge_disjoint,
    randomized_greedy,
    brute_force_k2,
    path_cost,
    dijkstra
)


@dataclass
class ResultadoExperimento:
    """Almacena los resultados de un experimento."""
    nombre_algoritmo: str
    max_costo: Optional[float]
    tiempo_segundos: float
    encontro_solucion: bool
    costos_caminos: Optional[List[float]] = None


def generar_grafo_etapas(n_etapas: int, k: int, rango_pesos: Tuple[int, int] = (1, 100)) -> Graph:
    """
    Genera un grafo en forma de etapas (similar al de la reducción NP).
    
    Args:
        n_etapas: Número de etapas (equivalente a n elementos en Partition)
        k: Número de aristas paralelas por etapa (= número de vehículos)
        rango_pesos: Rango (min, max) para los pesos aleatorios
    
    Returns:
        Grafo donde cada etapa tiene k aristas paralelas
    """
    graph: Graph = {}
    for i in range(n_etapas + 1):
        graph[i] = []
    
    for i in range(n_etapas):
        # Una arista con peso aleatorio
        peso = random.randint(*rango_pesos)
        graph[i].append((i + 1, float(peso)))
        # k-1 aristas con peso 0 (o pequeño)
        for _ in range(k - 1):
            graph[i].append((i + 1, 0.0))
    
    return graph


def generar_grafo_aleatorio(n_nodos: int, densidad: float = 0.3, 
                            rango_pesos: Tuple[int, int] = (1, 100)) -> Graph:
    """
    Genera un grafo dirigido aleatorio.
    
    Args:
        n_nodos: Número de nodos
        densidad: Probabilidad de que exista una arista entre dos nodos
        rango_pesos: Rango para pesos aleatorios
    
    Returns:
        Grafo aleatorio
    """
    graph: Graph = {i: [] for i in range(n_nodos)}
    
    for i in range(n_nodos):
        for j in range(i + 1, n_nodos):
            if random.random() < densidad:
                peso = random.randint(*rango_pesos)
                graph[i].append((j, float(peso)))
            # Posibilidad de arista inversa (menor probabilidad)
            if random.random() < densidad * 0.3:
                peso = random.randint(*rango_pesos)
                graph[j].append((i, float(peso)))
    
    return graph


def generar_grafo_grid(filas: int, columnas: int, 
                       rango_pesos: Tuple[int, int] = (1, 50)) -> Tuple[Graph, int, int]:
    """
    Genera un grafo en forma de grid (malla), simulando calles de una ciudad.
    
    Args:
        filas: Número de filas
        columnas: Número de columnas
        rango_pesos: Rango para pesos
    
    Returns:
        (grafo, nodo_origen, nodo_destino)
    """
    def nodo_id(f: int, c: int) -> int:
        return f * columnas + c
    
    graph: Graph = {nodo_id(f, c): [] for f in range(filas) for c in range(columnas)}
    
    for f in range(filas):
        for c in range(columnas):
            actual = nodo_id(f, c)
            # Arista hacia la derecha
            if c + 1 < columnas:
                peso = random.randint(*rango_pesos)
                graph[actual].append((nodo_id(f, c + 1), float(peso)))
            # Arista hacia abajo
            if f + 1 < filas:
                peso = random.randint(*rango_pesos)
                graph[actual].append((nodo_id(f + 1, c), float(peso)))
            # Aristas diagonales (opcional, menor probabilidad)
            if f + 1 < filas and c + 1 < columnas and random.random() < 0.3:
                peso = random.randint(*rango_pesos)
                graph[actual].append((nodo_id(f + 1, c + 1), float(peso)))
    
    origen = nodo_id(0, 0)
    destino = nodo_id(filas - 1, columnas - 1)
    return graph, origen, destino


def ejecutar_experimento(graph: Graph, s: int, t: int, k: int,
                        usar_fuerza_bruta: bool = True,
                        trials_random: int = 50) -> List[ResultadoExperimento]:
    """
    Ejecuta todos los algoritmos sobre un grafo dado y mide resultados.
    
    Args:
        graph: El grafo de entrada
        s: Nodo origen
        t: Nodo destino
        k: Número de caminos (vehículos)
        usar_fuerza_bruta: Si True, ejecuta fuerza bruta (solo para k=2 y grafos pequeños)
        trials_random: Número de iteraciones para el greedy aleatorizado
    
    Returns:
        Lista de resultados por algoritmo
    """
    resultados = []
    
    # 1. Greedy secuencial
    inicio = time.perf_counter()
    paths_greedy = greedy_k_edge_disjoint(graph, s, t, k)
    tiempo_greedy = time.perf_counter() - inicio
    
    if paths_greedy is None:
        resultados.append(ResultadoExperimento(
            nombre_algoritmo="Greedy Secuencial",
            max_costo=None,
            tiempo_segundos=tiempo_greedy,
            encontro_solucion=False
        ))
    else:
        costos = [path_cost(graph, p) for p in paths_greedy]
        resultados.append(ResultadoExperimento(
            nombre_algoritmo="Greedy Secuencial",
            max_costo=max(costos),
            tiempo_segundos=tiempo_greedy,
            encontro_solucion=True,
            costos_caminos=costos
        ))
    
    # 2. Greedy aleatorizado
    inicio = time.perf_counter()
    paths_random = randomized_greedy(graph, s, t, k, trials=trials_random)
    tiempo_random = time.perf_counter() - inicio
    
    if paths_random is None:
        resultados.append(ResultadoExperimento(
            nombre_algoritmo=f"Greedy Aleatorizado ({trials_random} trials)",
            max_costo=None,
            tiempo_segundos=tiempo_random,
            encontro_solucion=False
        ))
    else:
        costos = [path_cost(graph, p) for p in paths_random]
        resultados.append(ResultadoExperimento(
            nombre_algoritmo=f"Greedy Aleatorizado ({trials_random} trials)",
            max_costo=max(costos),
            tiempo_segundos=tiempo_random,
            encontro_solucion=True,
            costos_caminos=costos
        ))
    
    # 3. Fuerza bruta (solo para k=2 y grafos pequeños)
    if usar_fuerza_bruta and k == 2:
        inicio = time.perf_counter()
        result_bf = brute_force_k2(graph, s, t)
        tiempo_bf = time.perf_counter() - inicio
        
        if result_bf is None:
            resultados.append(ResultadoExperimento(
                nombre_algoritmo="Fuerza Bruta (k=2)",
                max_costo=None,
                tiempo_segundos=tiempo_bf,
                encontro_solucion=False
            ))
        else:
            p1, p2 = result_bf
            c1 = path_cost(graph, p1)
            c2 = path_cost(graph, p2)
            resultados.append(ResultadoExperimento(
                nombre_algoritmo="Fuerza Bruta (k=2)",
                max_costo=max(c1, c2),
                tiempo_segundos=tiempo_bf,
                encontro_solucion=True,
                costos_caminos=[c1, c2]
            ))
    
    return resultados


def imprimir_resultados(resultados: List[ResultadoExperimento], titulo: str = ""):
    """Imprime los resultados de forma tabular."""
    print("\n" + "=" * 70)
    if titulo:
        print(f"  {titulo}")
        print("=" * 70)
    
    print(f"{'Algoritmo':<40} {'Max Costo':>12} {'Tiempo (s)':>12}")
    print("-" * 70)
    
    for r in resultados:
        if r.encontro_solucion:
            print(f"{r.nombre_algoritmo:<40} {r.max_costo:>12.2f} {r.tiempo_segundos:>12.6f}")
        else:
            print(f"{r.nombre_algoritmo:<40} {'N/A':>12} {r.tiempo_segundos:>12.6f}")
    
    # Mostrar mejora relativa si hay fuerza bruta
    bf_result = next((r for r in resultados if "Fuerza Bruta" in r.nombre_algoritmo and r.encontro_solucion), None)
    if bf_result:
        print("-" * 70)
        print("Comparación con óptimo (Fuerza Bruta):")
        for r in resultados:
            if r.encontro_solucion and "Fuerza Bruta" not in r.nombre_algoritmo:
                gap = ((r.max_costo - bf_result.max_costo) / bf_result.max_costo) * 100 if bf_result.max_costo > 0 else 0
                speedup = bf_result.tiempo_segundos / r.tiempo_segundos if r.tiempo_segundos > 0 else float('inf')
                print(f"  {r.nombre_algoritmo}: Gap = {gap:+.1f}%, Speedup = {speedup:.1f}x")


def experimento_escalabilidad(sizes: List[int], k: int = 2, repeticiones: int = 5):
    """
    Experimenta cómo escalan los algoritmos con el tamaño del grafo.
    
    Args:
        sizes: Lista de tamaños (número de etapas) a probar
        k: Número de caminos
        repeticiones: Número de repeticiones por tamaño
    """
    print("\n" + "=" * 70)
    print("  EXPERIMENTO DE ESCALABILIDAD")
    print("=" * 70)
    print(f"Configuración: k={k}, repeticiones={repeticiones}")
    print(f"{'Tamaño':<10} {'Greedy (ms)':<15} {'Random (ms)':<15} {'BF (ms)':<15}")
    print("-" * 70)
    
    for n in sizes:
        tiempos_greedy = []
        tiempos_random = []
        tiempos_bf = []
        
        for _ in range(repeticiones):
            graph = generar_grafo_etapas(n, k)
            s, t = 0, n
            
            # Solo usar fuerza bruta para grafos pequeños
            usar_bf = n <= 12 and k == 2
            
            resultados = ejecutar_experimento(graph, s, t, k, usar_fuerza_bruta=usar_bf)
            
            for r in resultados:
                if "Greedy Secuencial" in r.nombre_algoritmo:
                    tiempos_greedy.append(r.tiempo_segundos * 1000)
                elif "Aleatorizado" in r.nombre_algoritmo:
                    tiempos_random.append(r.tiempo_segundos * 1000)
                elif "Fuerza Bruta" in r.nombre_algoritmo:
                    tiempos_bf.append(r.tiempo_segundos * 1000)
        
        avg_greedy = statistics.mean(tiempos_greedy) if tiempos_greedy else 0
        avg_random = statistics.mean(tiempos_random) if tiempos_random else 0
        avg_bf = statistics.mean(tiempos_bf) if tiempos_bf else float('nan')
        
        bf_str = f"{avg_bf:.3f}" if tiempos_bf else "N/A"
        print(f"{n:<10} {avg_greedy:<15.3f} {avg_random:<15.3f} {bf_str:<15}")


def experimento_calidad(n_grafos: int = 10, n_etapas: int = 8, k: int = 2):
    """
    Compara la calidad de las soluciones entre algoritmos.
    
    Args:
        n_grafos: Número de grafos aleatorios a generar
        n_etapas: Número de etapas por grafo
        k: Número de caminos
    """
    print("\n" + "=" * 70)
    print("  EXPERIMENTO DE CALIDAD DE SOLUCIÓN")
    print("=" * 70)
    print(f"Configuración: {n_grafos} grafos, {n_etapas} etapas, k={k}")
    
    gaps_greedy = []
    gaps_random = []
    
    for i in range(n_grafos):
        graph = generar_grafo_etapas(n_etapas, k)
        s, t = 0, n_etapas
        
        resultados = ejecutar_experimento(graph, s, t, k, usar_fuerza_bruta=True)
        
        bf_result = next((r for r in resultados if "Fuerza Bruta" in r.nombre_algoritmo and r.encontro_solucion), None)
        
        if bf_result and bf_result.max_costo > 0:
            for r in resultados:
                if r.encontro_solucion:
                    gap = ((r.max_costo - bf_result.max_costo) / bf_result.max_costo) * 100
                    if "Greedy Secuencial" in r.nombre_algoritmo:
                        gaps_greedy.append(gap)
                    elif "Aleatorizado" in r.nombre_algoritmo:
                        gaps_random.append(gap)
    
    print("\nResultados (Gap respecto al óptimo):")
    print("-" * 50)
    
    if gaps_greedy:
        print(f"Greedy Secuencial:")
        print(f"  Promedio: {statistics.mean(gaps_greedy):+.2f}%")
        print(f"  Máximo:   {max(gaps_greedy):+.2f}%")
        print(f"  Mínimo:   {min(gaps_greedy):+.2f}%")
    
    if gaps_random:
        print(f"\nGreedy Aleatorizado:")
        print(f"  Promedio: {statistics.mean(gaps_random):+.2f}%")
        print(f"  Máximo:   {max(gaps_random):+.2f}%")
        print(f"  Mínimo:   {min(gaps_random):+.2f}%")


def experimento_grid(filas: int = 5, columnas: int = 5, k: int = 2):
    """
    Experimenta con un grafo tipo grid (ciudad).
    
    Args:
        filas: Número de filas del grid
        columnas: Número de columnas del grid
        k: Número de caminos
    """
    print("\n" + "=" * 70)
    print(f"  EXPERIMENTO GRID ({filas}x{columnas})")
    print("=" * 70)
    
    graph, s, t = generar_grafo_grid(filas, columnas)
    n_nodos = filas * columnas
    n_aristas = sum(len(adj) for adj in graph.values())
    
    print(f"Nodos: {n_nodos}, Aristas: {n_aristas}")
    print(f"Origen: {s}, Destino: {t}, Vehículos: {k}")
    
    # Solo fuerza bruta para grids pequeños
    usar_bf = n_nodos <= 16 and k == 2
    
    resultados = ejecutar_experimento(graph, s, t, k, usar_fuerza_bruta=usar_bf, trials_random=100)
    imprimir_resultados(resultados)


def main():
    """Ejecuta todos los experimentos."""
    random.seed(42)  # Para reproducibilidad
    
    print("\n" + "#" * 70)
    print("#" + " " * 20 + "EXPERIMENTACIÓN - EL CONVOY" + " " * 21 + "#")
    print("#" * 70)
    
    # Experimento 1: Ejemplo básico (grafo de la reducción)
    print("\n>>> EXPERIMENTO 1: Grafo básico tipo reducción")
    graph = generar_grafo_etapas(5, 2, rango_pesos=(10, 50))
    resultados = ejecutar_experimento(graph, 0, 5, k=2)
    imprimir_resultados(resultados, "Grafo de 5 etapas, k=2")
    
    # Experimento 2: Escalabilidad
    print("\n>>> EXPERIMENTO 2: Escalabilidad")
    experimento_escalabilidad(sizes=[5, 8, 10, 12, 15, 20, 30], k=2, repeticiones=3)
    
    # Experimento 3: Calidad de soluciones
    print("\n>>> EXPERIMENTO 3: Calidad de soluciones")
    experimento_calidad(n_grafos=20, n_etapas=8, k=2)
    
    # Experimento 4: Grid (ciudad)
    print("\n>>> EXPERIMENTO 4: Grafo tipo ciudad (grid)")
    experimento_grid(filas=4, columnas=4, k=2)
    
    # Experimento 5: Más vehículos
    print("\n>>> EXPERIMENTO 5: Múltiples vehículos (k=3)")
    graph = generar_grafo_etapas(6, 3, rango_pesos=(10, 50))
    resultados = ejecutar_experimento(graph, 0, 6, k=3, usar_fuerza_bruta=False)
    imprimir_resultados(resultados, "Grafo de 6 etapas, k=3")
    
    print("\n" + "#" * 70)
    print("#" + " " * 20 + "FIN DE EXPERIMENTACIÓN" + " " * 26 + "#")
    print("#" * 70 + "\n")


if __name__ == "__main__":
    main()
