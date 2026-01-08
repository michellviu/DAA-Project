#!/usr/bin/env python3
"""
Generador de gráficas para el informe del problema "El Convoy".
Genera visualizaciones de los resultados experimentales.
"""

import os
import random
import statistics
import time
from typing import List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Importar desde el módulo de solución
from solucion import (
    Graph,
    greedy_k_edge_disjoint,
    randomized_greedy,
    brute_force_k2,
    path_cost,
)

# Configuración de estilo
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.figsize'] = (10, 6)

# Directorio de salida
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'docs', 'images')


def asegurar_directorio():
    """Crea el directorio de salida si no existe."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Directorio de salida: {os.path.abspath(OUTPUT_DIR)}")


def generar_grafo_etapas(n_etapas: int, k: int, rango_pesos: Tuple[int, int] = (1, 100)) -> Graph:
    """Genera un grafo en forma de etapas."""
    graph: Graph = {i: [] for i in range(n_etapas + 1)}
    for i in range(n_etapas):
        peso = random.randint(*rango_pesos)
        graph[i].append((i + 1, float(peso)))
        for _ in range(k - 1):
            graph[i].append((i + 1, 0.0))
    return graph


def generar_grafo_grid(filas: int, columnas: int, 
                       rango_pesos: Tuple[int, int] = (1, 50)) -> Tuple[Graph, int, int]:
    """Genera un grafo en forma de grid."""
    def nodo_id(f: int, c: int) -> int:
        return f * columnas + c
    
    graph: Graph = {nodo_id(f, c): [] for f in range(filas) for c in range(columnas)}
    
    for f in range(filas):
        for c in range(columnas):
            actual = nodo_id(f, c)
            if c + 1 < columnas:
                peso = random.randint(*rango_pesos)
                graph[actual].append((nodo_id(f, c + 1), float(peso)))
            if f + 1 < filas:
                peso = random.randint(*rango_pesos)
                graph[actual].append((nodo_id(f + 1, c), float(peso)))
            if f + 1 < filas and c + 1 < columnas and random.random() < 0.3:
                peso = random.randint(*rango_pesos)
                graph[actual].append((nodo_id(f + 1, c + 1), float(peso)))
    
    return graph, nodo_id(0, 0), nodo_id(filas - 1, columnas - 1)


def grafica_escalabilidad():
    """Genera gráfica de escalabilidad temporal."""
    print("\n[1/4] Generando gráfica de escalabilidad...")
    
    sizes = [5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 18, 20, 25, 30]
    k = 2
    repeticiones = 3
    
    tiempos_greedy = []
    tiempos_random = []
    tiempos_bf = []
    sizes_bf = []
    
    for n in sizes:
        tg, tr, tb = [], [], []
        
        for _ in range(repeticiones):
            graph = generar_grafo_etapas(n, k)
            s, t = 0, n
            
            # Greedy
            inicio = time.perf_counter()
            greedy_k_edge_disjoint(graph, s, t, k)
            tg.append((time.perf_counter() - inicio) * 1000)
            
            # Randomizado
            inicio = time.perf_counter()
            randomized_greedy(graph, s, t, k, trials=50)
            tr.append((time.perf_counter() - inicio) * 1000)
            
            # Fuerza bruta solo para n pequeño
            if n <= 12:
                inicio = time.perf_counter()
                brute_force_k2(graph, s, t)
                tb.append((time.perf_counter() - inicio) * 1000)
        
        tiempos_greedy.append(statistics.mean(tg))
        tiempos_random.append(statistics.mean(tr))
        if tb:
            tiempos_bf.append(statistics.mean(tb))
            sizes_bf.append(n)
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(sizes, tiempos_greedy, 'o-', color='#2ecc71', linewidth=2, 
            markersize=8, label='Greedy Secuencial')
    ax.plot(sizes, tiempos_random, 's-', color='#3498db', linewidth=2, 
            markersize=8, label='Greedy Aleatorizado (50 trials)')
    ax.plot(sizes_bf, tiempos_bf, '^-', color='#e74c3c', linewidth=2, 
            markersize=8, label='Fuerza Bruta')
    
    ax.set_xlabel('Número de etapas (n)')
    ax.set_ylabel('Tiempo de ejecución (ms)')
    ax.set_title('Escalabilidad de los Algoritmos - Grafo Tipo Reducción')
    ax.set_yscale('log')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # Añadir anotación
    ax.annotate('Crecimiento\nexponencial', 
                xy=(12, tiempos_bf[-1]), 
                xytext=(14, tiempos_bf[-1] * 2),
                arrowprops=dict(arrowstyle='->', color='#e74c3c'),
                fontsize=10, color='#e74c3c')
    
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, 'escalabilidad.pdf')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.savefig(filepath.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Guardado: {filepath}")


def grafica_calidad():
    """Genera gráfica de calidad de soluciones (gap vs óptimo)."""
    print("\n[2/4] Generando gráfica de calidad de soluciones...")
    
    n_grafos = 30
    n_etapas = 8
    k = 2
    
    gaps_greedy = []
    gaps_random = []
    instancias = []
    
    for i in range(n_grafos):
        graph = generar_grafo_etapas(n_etapas, k)
        s, t = 0, n_etapas
        
        # Óptimo
        result_bf = brute_force_k2(graph, s, t)
        if result_bf is None:
            continue
        p1, p2 = result_bf
        opt = max(path_cost(graph, p1), path_cost(graph, p2))
        
        if opt == 0:
            continue
        
        # Greedy
        paths_g = greedy_k_edge_disjoint(graph, s, t, k)
        if paths_g:
            costos = [path_cost(graph, p) for p in paths_g]
            gap_g = ((max(costos) - opt) / opt) * 100
            gaps_greedy.append(gap_g)
        
        # Randomizado
        paths_r = randomized_greedy(graph, s, t, k, trials=50)
        if paths_r:
            costos = [path_cost(graph, p) for p in paths_r]
            gap_r = ((max(costos) - opt) / opt) * 100
            gaps_random.append(gap_r)
        
        instancias.append(i + 1)
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(12, 5))
    
    x = np.arange(len(instancias))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, gaps_greedy[:len(instancias)], width, 
                   label='Greedy Secuencial', color='#2ecc71', alpha=0.8)
    bars2 = ax.bar(x + width/2, gaps_random[:len(instancias)], width, 
                   label='Greedy Aleatorizado', color='#3498db', alpha=0.8)
    
    ax.axhline(y=0, color='#27ae60', linestyle='--', linewidth=2, label='Óptimo')
    ax.axhline(y=statistics.mean(gaps_greedy), color='#2ecc71', linestyle=':', 
               linewidth=2, alpha=0.7)
    ax.axhline(y=statistics.mean(gaps_random), color='#3498db', linestyle=':', 
               linewidth=2, alpha=0.7)
    
    ax.set_xlabel('Instancia de prueba')
    ax.set_ylabel('Gap respecto al óptimo (%)')
    ax.set_title('Calidad de Soluciones - Grafos Tipo Reducción (Peor Caso)')
    ax.set_xticks(x[::5])
    ax.set_xticklabels([str(i) for i in instancias[::5]])
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Estadísticas en el gráfico
    stats_text = f'Greedy: μ={statistics.mean(gaps_greedy):.1f}%\nAleatorizado: μ={statistics.mean(gaps_random):.1f}%'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, 'calidad_soluciones.pdf')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.savefig(filepath.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Guardado: {filepath}")


def grafica_comparacion_grafos():
    """Compara el rendimiento en diferentes tipos de grafos."""
    print("\n[3/4] Generando gráfica de comparación por tipo de grafo...")
    
    tipos = ['Etapas\n(Reducción)', 'Grid 4×4', 'Grid 5×5', 'Grid 6×6']
    
    # Resultados: [greedy_gap, random_gap] para cada tipo
    resultados_greedy = []
    resultados_random = []
    
    k = 2
    repeticiones = 10
    
    # Tipo 1: Etapas
    gaps_g, gaps_r = [], []
    for _ in range(repeticiones):
        graph = generar_grafo_etapas(8, k)
        result_bf = brute_force_k2(graph, 0, 8)
        if result_bf:
            opt = max(path_cost(graph, result_bf[0]), path_cost(graph, result_bf[1]))
            if opt > 0:
                paths_g = greedy_k_edge_disjoint(graph, 0, 8, k)
                paths_r = randomized_greedy(graph, 0, 8, k, trials=50)
                if paths_g:
                    gaps_g.append(((max(path_cost(graph, p) for p in paths_g) - opt) / opt) * 100)
                if paths_r:
                    gaps_r.append(((max(path_cost(graph, p) for p in paths_r) - opt) / opt) * 100)
    resultados_greedy.append(statistics.mean(gaps_g) if gaps_g else 0)
    resultados_random.append(statistics.mean(gaps_r) if gaps_r else 0)
    
    # Tipos 2-4: Grids
    for size in [4, 5, 6]:
        gaps_g, gaps_r = [], []
        for _ in range(repeticiones):
            graph, s, t = generar_grafo_grid(size, size)
            
            # Solo fuerza bruta para grids pequeños
            if size <= 4:
                result_bf = brute_force_k2(graph, s, t)
                if result_bf:
                    opt = max(path_cost(graph, result_bf[0]), path_cost(graph, result_bf[1]))
                else:
                    continue
            else:
                # Usar greedy como aproximación del óptimo
                paths_g = greedy_k_edge_disjoint(graph, s, t, k)
                if paths_g:
                    opt = max(path_cost(graph, p) for p in paths_g)
                else:
                    continue
            
            if opt == 0:
                continue
                
            paths_g = greedy_k_edge_disjoint(graph, s, t, k)
            paths_r = randomized_greedy(graph, s, t, k, trials=100)
            
            if paths_g:
                val_g = max(path_cost(graph, p) for p in paths_g)
                gaps_g.append(((val_g - opt) / opt) * 100 if opt > 0 else 0)
            if paths_r:
                val_r = max(path_cost(graph, p) for p in paths_r)
                gaps_r.append(((val_r - opt) / opt) * 100 if opt > 0 else 0)
        
        resultados_greedy.append(statistics.mean(gaps_g) if gaps_g else 0)
        resultados_random.append(statistics.mean(gaps_r) if gaps_r else 0)
    
    # Crear figura
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(tipos))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, resultados_greedy, width, 
                   label='Greedy Secuencial', color='#2ecc71')
    bars2 = ax.bar(x + width/2, resultados_random, width, 
                   label='Greedy Aleatorizado', color='#3498db')
    
    ax.set_xlabel('Tipo de Grafo')
    ax.set_ylabel('Gap promedio respecto al óptimo (%)')
    ax.set_title('Rendimiento por Tipo de Grafo')
    ax.set_xticks(x)
    ax.set_xticklabels(tipos)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Añadir valores sobre las barras
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, 'comparacion_grafos.pdf')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.savefig(filepath.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Guardado: {filepath}")


def grafica_estructura_reduccion():
    """Genera una visualización del grafo de la reducción NP."""
    print("\n[4/4] Generando diagrama de la reducción...")
    
    fig, ax = plt.subplots(figsize=(12, 4))
    
    n_etapas = 4
    a = [3, 5, 2, 4]  # Valores de ejemplo para la reducción
    
    # Posiciones de nodos
    x_positions = np.linspace(0.5, 9.5, n_etapas + 1)
    y_center = 0.5
    
    # Dibujar nodos
    for i, x in enumerate(x_positions):
        if i == 0:
            label = '$s$'
            color = '#27ae60'
        elif i == n_etapas:
            label = '$t$'
            color = '#e74c3c'
        else:
            label = f'$u_{i}$'
            color = '#3498db'
        
        circle = plt.Circle((x, y_center), 0.25, color=color, zorder=3)
        ax.add_patch(circle)
        ax.text(x, y_center, label, ha='center', va='center', 
                fontsize=14, color='white', fontweight='bold', zorder=4)
    
    # Dibujar aristas
    for i in range(n_etapas):
        x1 = x_positions[i] + 0.25
        x2 = x_positions[i + 1] - 0.25
        
        # Arista superior (peso a_i)
        y_top = y_center + 0.15
        ax.annotate('', xy=(x2, y_top), xytext=(x1, y_top),
                    arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=2))
        ax.text((x1 + x2) / 2, y_top + 0.15, f'$a_{i+1}={a[i]}$', 
                ha='center', va='bottom', fontsize=10, color='#e74c3c')
        
        # Arista inferior (peso 0)
        y_bot = y_center - 0.15
        ax.annotate('', xy=(x2, y_bot), xytext=(x1, y_bot),
                    arrowprops=dict(arrowstyle='->', color='#2ecc71', lw=2))
        ax.text((x1 + x2) / 2, y_bot - 0.15, '$0$', 
                ha='center', va='top', fontsize=10, color='#2ecc71')
    
    # Leyenda y título
    ax.text(5, -0.5, f'$W = \\sum a_i = {sum(a)}$, $T = W/2 = {sum(a)/2}$', 
            ha='center', fontsize=12, style='italic')
    
    red_patch = mpatches.Patch(color='#e74c3c', label=f'Arista costosa ($a_i$)')
    green_patch = mpatches.Patch(color='#2ecc71', label='Arista gratuita (0)')
    ax.legend(handles=[red_patch, green_patch], loc='upper right')
    
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.8, 1.2)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Construcción del Grafo para la Reducción Partition → Convoy ($k=2$)', 
                 fontsize=14, pad=20)
    
    plt.tight_layout()
    filepath = os.path.join(OUTPUT_DIR, 'reduccion_grafo.pdf')
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.savefig(filepath.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Guardado: {filepath}")


def main():
    """Genera todas las gráficas."""
    random.seed(42)
    
    print("=" * 60)
    print("  GENERADOR DE GRÁFICAS - EL CONVOY")
    print("=" * 60)
    
    asegurar_directorio()
    
    grafica_escalabilidad()
    grafica_calidad()
    grafica_comparacion_grafos()
    grafica_estructura_reduccion()
    
    print("\n" + "=" * 60)
    print("  ¡Todas las gráficas generadas exitosamente!")
    print("=" * 60)
    print(f"\nArchivos en: {os.path.abspath(OUTPUT_DIR)}")
    print("Formatos: PDF (para LaTeX) y PNG (para previsualización)")


if __name__ == "__main__":
    main()
