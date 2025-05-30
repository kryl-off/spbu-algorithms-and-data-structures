import sys
import random
import time
import math
from datetime import datetime
from math import sqrt
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QSplitter, QPushButton, QTableWidget, QTableWidgetItem, QLabel,
                             QSpinBox, QPlainTextEdit, QMessageBox, QCheckBox, QTabWidget, QComboBox, QDoubleSpinBox, QLineEdit)
from PyQt5.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.patches import FancyArrowPatch
import networkx as nx

NO_EDGE = 9999

def nearest_neighbor(graph, start=0, NO_EDGE=NO_EDGE):
    n = len(graph)
    visited = [False] * n
    path = [start]
    visited[start] = True
    total_cost = 0
    current = start
    for _ in range(n - 1):
        next_node = None
        min_cost = float('inf')
        for j in range(n):
            if not visited[j]:
                cost = graph[current][j]
                if cost < min_cost:
                    min_cost = cost
                    next_node = j
        if next_node is None or min_cost == NO_EDGE:
            return path, None
        path.append(next_node)
        visited[next_node] = True
        total_cost += min_cost
        current = next_node
    if graph[current][start] == NO_EDGE:
        return path, None
    total_cost += graph[current][start]
    path.append(start)
    return path, total_cost

def modified_nearest_neighbor(graph, NO_EDGE=NO_EDGE):
    best_path = None
    best_total = None
    n = len(graph)
    for start in range(n):
        path, total_cost = nearest_neighbor(graph, start, NO_EDGE)
        if total_cost is not None and (best_total is None or total_cost < best_total):
            best_path = path
            best_total = total_cost
    return best_path, best_total

def nearest_neighbor_two_step(graph, start=0, NO_EDGE=NO_EDGE):
    n = len(graph)
    visited = [False] * n
    path = [start]
    visited[start] = True
    total_cost = 0
    current = start
    for step in range(n - 1):
        candidate = None
        best_value = float('inf')
        chosen_cost = None
        for j in range(n):
            if not visited[j] and graph[current][j] != NO_EDGE:
                cost_current_to_j = graph[current][j]
                remaining = [k for k in range(n) if not visited[k] and k != j]
                if remaining:
                    valid_costs = [graph[j][k] for k in remaining if graph[j][k] != NO_EDGE]
                    cost_lookahead = min(valid_costs) if valid_costs else NO_EDGE
                    candidate_value = cost_current_to_j + cost_lookahead
                else:
                    candidate_value = cost_current_to_j
                if candidate_value < best_value:
                    best_value = candidate_value
                    candidate = j
                    chosen_cost = cost_current_to_j
        if candidate is None:
            return path, None
        path.append(candidate)
        visited[candidate] = True
        total_cost += chosen_cost
        current = candidate
    if graph[current][start] == NO_EDGE:
        return path, None
    total_cost += graph[current][start]
    path.append(start)
    return path, total_cost

def modified_nearest_neighbor_two_step(graph, NO_EDGE=NO_EDGE):
    best_path = None
    best_total = None
    n = len(graph)
    for start in range(n):
        path, total_cost = nearest_neighbor_two_step(graph, start, NO_EDGE)
        if total_cost is not None and (best_total is None or total_cost < best_total):
            best_path = path
            best_total = total_cost
    return best_path, best_total

def nearest_neighbor_random(graph, start=0, random_prob=0.2, NO_EDGE=NO_EDGE):
    n = len(graph)
    visited = [False] * n
    path = [start]
    visited[start] = True
    total_cost = 0
    current = start
    for _ in range(n - 1):
        candidates = [j for j in range(n) if not visited[j] and graph[current][j] != NO_EDGE]
        if not candidates:
            return path, None
        if random.random() < random_prob:
            candidate = random.choice(candidates)
            chosen_cost = graph[current][candidate]
        else:
            candidate = min(candidates, key=lambda j: graph[current][j])
            chosen_cost = graph[current][candidate]
        path.append(candidate)
        visited[candidate] = True
        total_cost += chosen_cost
        current = candidate
    if graph[current][start] == NO_EDGE:
        return path, None
    total_cost += graph[current][start]
    path.append(start)
    return path, total_cost

def modified_nearest_neighbor_random(graph, random_prob=0.2, NO_EDGE=NO_EDGE):
    best_path = None
    best_total = None
    n = len(graph)
    for start in range(n):
        path, total_cost = nearest_neighbor_random(graph, start, random_prob, NO_EDGE)
        if total_cost is not None and (best_total is None or total_cost < best_total):
            best_path = path
            best_total = total_cost
    return best_path, best_total

def nearest_neighbor_two_step_random(graph, start=0, random_prob=0.2, NO_EDGE=NO_EDGE):
    n = len(graph)
    visited = [False] * n
    path = [start]
    visited[start] = True
    total_cost = 0
    current = start
    for step in range(n - 1):
        candidate = None
        best_value = float('inf')
        chosen_cost = None
        for j in range(n):
            if not visited[j] and graph[current][j] != NO_EDGE:
                cost_current_to_j = graph[current][j]
                if random.random() < random_prob:
                    candidate_value = cost_current_to_j
                else:
                    remaining = [k for k in range(n) if not visited[k] and k != j]
                    if remaining:
                        valid_costs = [graph[j][k] for k in remaining if graph[j][k] != NO_EDGE]
                        cost_lookahead = min(valid_costs) if valid_costs else NO_EDGE
                        candidate_value = cost_current_to_j + cost_lookahead
                    else:
                        candidate_value = cost_current_to_j
                if candidate_value < best_value:
                    best_value = candidate_value
                    candidate = j
                    chosen_cost = cost_current_to_j
        if candidate is None:
            return path, None
        path.append(candidate)
        visited[candidate] = True
        total_cost += chosen_cost
        current = candidate
    if graph[current][start] == NO_EDGE:
        return path, None
    total_cost += graph[current][start]
    path.append(start)
    return path, total_cost

def modified_nearest_neighbor_two_step_random(graph, random_prob=0.2, NO_EDGE=NO_EDGE):
    best_path = None
    best_total = None
    n = len(graph)
    for start in range(n):
        path, total_cost = nearest_neighbor_two_step_random(graph, start, random_prob, NO_EDGE)
        if total_cost is not None and (best_total is None or total_cost < best_total):
            best_path = path
            best_total = total_cost
    return best_path, best_total

def simulated_annealing(graph, start_temp=1000, end_temp=1, alpha=0.995, max_iter=1000, 
                        use_nn_start=True, sa_variant="standard",initial_path=None):
    n = len(graph)

    def random_path():
        path = list(range(n))
        random.shuffle(path)
        path.append(path[0])
        return path

    def path_cost(path):
        cost = 0
        for i in range(len(path) - 1):
            w = graph[path[i]][path[i + 1]]
            if w == NO_EDGE:
                return float('inf')
            cost += w
        return cost

    # Инициализация начального пути:
    if initial_path is not None:
        init_path = initial_path
    elif use_nn_start:
        init_path, _ = nearest_neighbor(graph)
        if init_path is None:
            init_path = random_path()
    else:
        init_path = random_path()

    path = init_path[:]  # сохраняем копию начального решения
    current_cost = path_cost(path)
    best_path = path[:]
    best_cost = current_cost
    T = start_temp
    iter_count = 0

    while T > end_temp and iter_count < max_iter:
        i, j = sorted(random.sample(range(1, n), 2))
        new_path = path[:i] + path[i:j+1][::-1] + path[j+1:]
        new_cost = path_cost(new_path)
        delta = new_cost - current_cost

        if delta < 0 or random.random() < math.exp(-delta / T):
            path = new_path
            current_cost = new_cost
            if new_cost < best_cost:
                best_path = new_path
                best_cost = new_cost

        iter_count += 1
        if sa_variant == "standard":
            T *= alpha
        elif sa_variant == "boltzmann":
            T = start_temp / math.log(iter_count + 20)

    best_path.append(best_path[0])
    return init_path, best_path, best_cost

def draw_graph(ax, edge_list, num_vertices, cycle_path=None, title="Graph"):
    ax.clear()
    G = nx.MultiDiGraph()
    for i in range(num_vertices):
        G.add_node(i)
    for (u, v, w) in edge_list:
        if 0 <= u < num_vertices and 0 <= v < num_vertices and u != v:
            G.add_edge(u, v, weight=w)
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, ax=ax)
    nx.draw_networkx_labels(G, pos, ax=ax)

    def draw_broken_arrow(start, end, label, color):
        L = sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        if L == 0:
            L = 0.001
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        dx_unit = dx / L
        dy_unit = dy / L
        gap_fraction = 0.1
        gap_length = gap_fraction * L
        mid = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        mid1 = (mid[0] - (gap_length / 2) * dx_unit, mid[1] - (gap_length / 2) * dy_unit)
        mid2 = (mid[0] + (gap_length / 2) * dx_unit, mid[1] + (gap_length / 2) * dy_unit)
        arrow1 = FancyArrowPatch(start, mid1, arrowstyle='-', mutation_scale=12,
                                 color=color, linewidth=1, shrinkA=5, shrinkB=5)
        ax.add_patch(arrow1)
        arrow2 = FancyArrowPatch(mid2, end, arrowstyle='-|>', mutation_scale=12,
                                 color=color, linewidth=1, shrinkA=5, shrinkB=5)
        ax.add_patch(arrow2)
        ax.text(mid[0], mid[1], str(label), fontsize=10, color='black')

    if cycle_path is None:
        edge_groups = {}
        for u, v, key, data in G.edges(keys=True, data=True):
            edge_groups.setdefault((u, v), []).append((u, v, key, data))
        delta = 0.05
        for (u, v), edges in edge_groups.items():
            base = delta if u < v else -delta
            for i, (u, v, key, data) in enumerate(sorted(edges, key=lambda x: x[3].get('weight', 0))):
                offset_value = base + delta
                x1, y1 = pos[u]
                x2, y2 = pos[v]
                dx = x2 - x1
                dy = y2 - y1
                L = sqrt(dx * dx + dy * dy)
                perp = (-dy / L, dx / L) if L != 0 else (0, 0)
                shift_x = offset_value * perp[0]
                shift_y = offset_value * perp[1]
                start = (x1 + shift_x, y1 + shift_y)
                end = (x2 + shift_x, y2 + shift_y)
                draw_broken_arrow(start, end, data.get('weight', ''), color='black')
    else:
        edge_dict = {}
        for (u, v, w) in edge_list:
            if 0 <= u < num_vertices and 0 <= v < num_vertices and u != v:
                if (u, v) not in edge_dict:
                    edge_dict[(u, v)] = w
                else:
                    edge_dict[(u, v)] = min(edge_dict[(u, v)], w)
        delta = 0.05
        for i in range(len(cycle_path) - 1):
            u = cycle_path[i]
            v = cycle_path[i + 1]
            if (u, v) in edge_dict:
                base = delta if u < v else -delta
                x1, y1 = pos[u]
                x2, y2 = pos[v]
                dx = x2 - x1
                dy = y2 - y1
                L = sqrt(dx * dx + dy * dy)
                perp = (-dy / L, dx / L) if L != 0 else (0, 0)
                shift_x = base * perp[0]
                shift_y = base * perp[1]
                start = (x1 + shift_x, y1 + shift_y)
                end = (x2 + shift_x, y2 + shift_y)
                w = edge_dict[(u, v)]
                draw_broken_arrow(start, end, w, color='red')

    ax.set_title(title)
    ax.axis('off')

def ant_colony_optimization(graph, num_ants=20, num_iterations=100, decay=0.1, alpha=1, beta=2, elite_factor=0):

    n = len(graph)
    pheromone = [[1 for j in range(n)] for i in range(n)]
    best_path = None
    best_cost = float('inf')

    def tour_cost(tour):
        cost = 0
        for i in range(len(tour) - 1):
            w = graph[tour[i]][tour[i+1]]
            if w == NO_EDGE:
                return float('inf')
            cost += w
        return cost

    for i in range(num_iterations):
        tours = []
        costs = []
        for ant in range(num_ants):
            start = random.randint(0, n - 1)
            tour = [start]
            visited = {start}
            for step in range(n - 1):
                curr = tour[-1]
                choices = []
                total_attractiveness = 0.0
                for j in range(n):
                    if j not in visited and graph[curr][j] != NO_EDGE:
                        attractiveness = (pheromone[curr][j] ** alpha) * ((1.0 / graph[curr][j]) ** beta)
                        choices.append((j, attractiveness))
                        total_attractiveness += attractiveness
                if total_attractiveness == 0:
                    break
                r = random.uniform(0, total_attractiveness)
                cumulative = 0.0
                next_vertex = None
                for j, attractiveness in choices:
                    cumulative += attractiveness
                    if cumulative >= r:
                        next_vertex = j
                        break
                if next_vertex is None:
                    break
                tour.append(next_vertex)
                visited.add(next_vertex)
            tour.append(start)
            cost = tour_cost(tour)
            tours.append(tour)
            costs.append(cost)
            if cost < best_cost:
                best_cost = cost
                best_path = tour[:]

        # испарение феромона
        for i in range(n):
            for j in range(n):
                pheromone[i][j] *= (1 - decay)
        
        # осаждение
        for tour, cost in zip(tours, costs):
            if cost == float('inf'):
                continue
            deposit = 1.0 / cost
            for i in range(len(tour) - 1):
                a, b = tour[i], tour[i+1]
                pheromone[a][b] += deposit
                pheromone[b][a] += deposit
        
        # модификация 
        if elite_factor > 0 and best_path is not None and best_cost != 0:
            elite_deposit = elite_factor / best_cost
            for i in range(len(best_path) - 1):
                a, b = best_path[i], best_path[i+1]
                pheromone[a][b] += elite_deposit
                pheromone[b][a] += elite_deposit

    return best_path, best_cost


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Задача о коммивояжере")
        self.num_vertices = 3
        self.default_edges = [
            (0, 1, 5), (1, 0, 6), (0, 2, 8), (2, 0, 7), (1, 2, 4), (2, 1, 9)
        ]
        self.initUI()

    def initUI(self):
        splitter = QSplitter(Qt.Horizontal)
        leftWidget = QWidget()
        leftLayout = QVBoxLayout()
        leftWidget.setLayout(leftLayout)

        self.input_fig, self.input_ax = plt.subplots(figsize=(4, 3))
        self.input_canvas = FigureCanvas(self.input_fig)
        leftLayout.addWidget(QLabel("Входной граф"))
        leftLayout.addWidget(self.input_canvas)

        self.output_fig, self.output_ax = plt.subplots(figsize=(4, 3))
        self.output_canvas = FigureCanvas(self.output_fig)
        leftLayout.addWidget(QLabel("Выходной граф (найденный гамильтонов цикл)"))
        leftLayout.addWidget(self.output_canvas)

        self.cycle_label = QLabel("Путь: ")
        self.distance_label = QLabel("Расстояние: ")
        leftLayout.addWidget(self.cycle_label)
        leftLayout.addWidget(self.distance_label)

        rightWidget = QWidget()
        rightLayout = QVBoxLayout()
        rightWidget.setLayout(rightLayout)

        # Блок настройки графа
        verticesLayout = QHBoxLayout()
        verticesLabel = QLabel("Количество вершин:")
        self.verticesSpinBox = QSpinBox()
        self.verticesSpinBox.setMinimum(3)
        self.verticesSpinBox.setMaximum(50)
        self.verticesSpinBox.setValue(self.num_vertices)
        self.verticesSpinBox.valueChanged.connect(self.change_vertex_count)
        verticesLayout.addWidget(verticesLabel)
        verticesLayout.addWidget(self.verticesSpinBox)
        rightLayout.addLayout(verticesLayout)

        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Исходная вершина", "Целевая вершина", "Расстояние"])
        self.table.setRowCount(0)
        rightLayout.addWidget(QLabel("Задание ребер графа"))
        rightLayout.addWidget(self.table)

        self.addRowButton = QPushButton("Добавить ребро")
        self.addRowButton.clicked.connect(self.add_edge_row)
        rightLayout.addWidget(self.addRowButton)
        self.deleteRowButton = QPushButton("Удалить выбранные ребра")
        self.deleteRowButton.clicked.connect(self.delete_selected_edge_rows)
        rightLayout.addWidget(self.deleteRowButton)

        self.edgesMatrixEdit = QPlainTextEdit()
        rightLayout.addWidget(self.edgesMatrixEdit)
        self.applyEdgesMatrixButton = QPushButton("Применить матрицу ребер")
        self.applyEdgesMatrixButton.clicked.connect(self.apply_edges_matrix_text)
        self.copyEdgesMatrixButton = QPushButton("Обновить матрицу ребер")
        self.copyEdgesMatrixButton.clicked.connect(self.update_edges_matrix_text)
        rightLayout.addWidget(self.applyEdgesMatrixButton)
        rightLayout.addWidget(self.copyEdgesMatrixButton)

        # Блок настройки алгоритма
        rightLayout.addWidget(QLabel("Настройки алгоритма"))
        self.tabs = QTabWidget()
        rightLayout.addWidget(self.tabs)

        # Вкладка для NN
        self.nn_tab = QWidget()
        nn_layout = QVBoxLayout()
        self.nn_tab.setLayout(nn_layout)
        self.checkbox_multi_start = QCheckBox("Многостартовая модификация")
        self.checkbox_two_step = QCheckBox("Двухшаговое предвидение")
        self.checkbox_random = QCheckBox("Рандомный выбор следующей вершины")
        nn_layout.addWidget(self.checkbox_multi_start)
        nn_layout.addWidget(self.checkbox_two_step)
        nn_layout.addWidget(self.checkbox_random)
        self.tabs.addTab(self.nn_tab, "NN")

        # Вкладка для SA
        self.sa_tab = QWidget()
        sa_layout = QVBoxLayout()
        self.sa_tab.setLayout(sa_layout)

        self.sa_start_method = QComboBox()
        self.sa_start_method.addItems(["Случайный", "Ближайший сосед", "Вручную"])  # Добавлена опция «Вручную»
        self.sa_start_method.currentIndexChanged.connect(self.update_manual_path_visibility)
        sa_layout.addWidget(QLabel("Метод начального пути"))
        sa_layout.addWidget(self.sa_start_method)
        self.sa_manual_path_lineedit = QLineEdit()

        self.sa_manual_path_lineedit.setPlaceholderText("Введите номера вершин через запятую, например: 0,2,1")

        self.sa_manual_path_lineedit.setEnabled(False)
        sa_layout.addWidget(QLabel("Ручной ввод начального пути"))
        sa_layout.addWidget(self.sa_manual_path_lineedit)

        self.sa_algorithm = QComboBox()
        self.sa_algorithm.addItems(["Стандартный отжиг", "Больцановский отжиг"])
        sa_layout.addWidget(QLabel("Вариант имитации отжига"))
        sa_layout.addWidget(self.sa_algorithm)

        sa_start_temp_layout = QHBoxLayout()
        sa_start_temp_label = QLabel("Начальная температура:")
        self.sa_start_temp_spinbox = QDoubleSpinBox()
        self.sa_start_temp_spinbox.setMinimum(0)
        self.sa_start_temp_spinbox.setMaximum(1000000)
        self.sa_start_temp_spinbox.setValue(1000)
        sa_start_temp_layout.addWidget(sa_start_temp_label)
        sa_start_temp_layout.addWidget(self.sa_start_temp_spinbox)
        sa_layout.addLayout(sa_start_temp_layout)

        sa_end_temp_layout = QHBoxLayout()
        sa_end_temp_label = QLabel("Конечная температура:")
        self.sa_end_temp_spinbox = QDoubleSpinBox()
        self.sa_end_temp_spinbox.setMinimum(0)
        self.sa_end_temp_spinbox.setMaximum(1000000)
        self.sa_end_temp_spinbox.setValue(1)
        sa_end_temp_layout.addWidget(sa_end_temp_label)
        sa_end_temp_layout.addWidget(self.sa_end_temp_spinbox)
        sa_layout.addLayout(sa_end_temp_layout)

        sa_alpha_layout = QHBoxLayout()
        sa_alpha_label = QLabel("Коэффициент alpha:")
        self.sa_alpha_spinbox = QDoubleSpinBox()
        self.sa_alpha_spinbox.setMinimum(0)
        self.sa_alpha_spinbox.setMaximum(1)
        self.sa_alpha_spinbox.setDecimals(3)
        self.sa_alpha_spinbox.setSingleStep(0.001)
        self.sa_alpha_spinbox.setValue(0.995)
        sa_alpha_layout.addWidget(sa_alpha_label)
        sa_alpha_layout.addWidget(self.sa_alpha_spinbox)
        sa_layout.addLayout(sa_alpha_layout)

        sa_max_iter_layout = QHBoxLayout()
        sa_max_iter_label = QLabel("Макс итераций:")
        self.sa_max_iter_spinbox = QSpinBox()
        self.sa_max_iter_spinbox.setMinimum(1)
        self.sa_max_iter_spinbox.setMaximum(10000000)
        self.sa_max_iter_spinbox.setValue(1000)
        sa_max_iter_layout.addWidget(sa_max_iter_label)
        sa_max_iter_layout.addWidget(self.sa_max_iter_spinbox)
        sa_layout.addLayout(sa_max_iter_layout)

        self.tabs.addTab(self.sa_tab, "SA")

        # Вкладка для ACO
        self.aco_tab = QWidget()
        aco_layout = QVBoxLayout()
        self.aco_tab.setLayout(aco_layout)

        # Параметры АКО
        self.aco_num_ants_spinbox = QSpinBox()
        self.aco_num_ants_spinbox.setMinimum(1)
        self.aco_num_ants_spinbox.setMaximum(1000)
        self.aco_num_ants_spinbox.setValue(20)

        self.aco_num_iterations_spinbox = QSpinBox()
        self.aco_num_iterations_spinbox.setMinimum(1)
        self.aco_num_iterations_spinbox.setMaximum(10000)
        self.aco_num_iterations_spinbox.setValue(100)

        self.aco_decay_spinbox = QDoubleSpinBox()
        self.aco_decay_spinbox.setMinimum(0.0)
        self.aco_decay_spinbox.setMaximum(1.0)
        self.aco_decay_spinbox.setSingleStep(0.01)
        self.aco_decay_spinbox.setValue(0.1)

        self.aco_alpha_spinbox = QDoubleSpinBox()
        self.aco_alpha_spinbox.setMinimum(0.0)
        self.aco_alpha_spinbox.setMaximum(10.0)
        self.aco_alpha_spinbox.setSingleStep(0.1)
        self.aco_alpha_spinbox.setValue(1)

        self.aco_beta_spinbox = QDoubleSpinBox()
        self.aco_beta_spinbox.setMinimum(0.0)
        self.aco_beta_spinbox.setMaximum(10.0)
        self.aco_beta_spinbox.setSingleStep(0.1)
        self.aco_beta_spinbox.setValue(2)

        # Выбор режима ACO: "Обычные" или "Элитные"
        self.aco_mode_combobox = QComboBox()
        self.aco_mode_combobox.addItems(["Обычные", "Элитные"])
        self.aco_mode_combobox.currentIndexChanged.connect(self.update_aco_elite_visibility)

        # Elite factor (активен только для режима "Элитные")
        self.aco_elite_factor_spinbox = QDoubleSpinBox()
        self.aco_elite_factor_spinbox.setMinimum(0.0)
        self.aco_elite_factor_spinbox.setMaximum(10.0)
        self.aco_elite_factor_spinbox.setSingleStep(0.1)
        self.aco_elite_factor_spinbox.setValue(1.0)
        self.aco_elite_factor_spinbox.setEnabled(False)  # по умолчанию отключено, если выбраны "Обычные"

        # Добавляем виджеты на вкладку ACO:
        aco_layout.addWidget(QLabel("Число муравьев:"))
        aco_layout.addWidget(self.aco_num_ants_spinbox)
        aco_layout.addWidget(QLabel("Число итераций:"))
        aco_layout.addWidget(self.aco_num_iterations_spinbox)
        aco_layout.addWidget(QLabel("Коэффициент испарения (decay):"))
        aco_layout.addWidget(self.aco_decay_spinbox)
        aco_layout.addWidget(QLabel("Показатель влияния феромона (alpha):"))
        aco_layout.addWidget(self.aco_alpha_spinbox)
        aco_layout.addWidget(QLabel("Показатель влияния эвристики (beta):"))
        aco_layout.addWidget(self.aco_beta_spinbox)
        aco_layout.addWidget(QLabel("Режим ACO:"))
        aco_layout.addWidget(self.aco_mode_combobox)
        aco_layout.addWidget(QLabel("Elite Factor (для Элитных):"))
        aco_layout.addWidget(self.aco_elite_factor_spinbox)

        self.tabs.addTab(self.aco_tab, "ACO")

        # Кнопка рассчитать
        self.calc_button = QPushButton("Рассчитать")
        self.calc_button.clicked.connect(self.calculate)
        rightLayout.addWidget(self.calc_button)

        splitter.addWidget(leftWidget)
        splitter.addWidget(rightWidget)
        self.setCentralWidget(splitter)

        self.input_ax.clear()
        self.input_canvas.draw()
        self.output_ax.clear()
        self.output_canvas.draw()

        if self.num_vertices == 3 and self.table.rowCount() == 0:
            self.fill_default_example()
        self.update_edges_matrix_text()

    def update_manual_path_visibility(self):
        if self.sa_start_method.currentText() == "Вручную":
            self.sa_manual_path_lineedit.setEnabled(True)
        else:
            self.sa_manual_path_lineedit.setEnabled(False)
            self.sa_manual_path_lineedit.clear()

    def update_aco_elite_visibility(self):
        if self.aco_mode_combobox.currentText() == "Элитные":
            self.aco_elite_factor_spinbox.setEnabled(True)
        else:
            self.aco_elite_factor_spinbox.setEnabled(False)
            self.aco_elite_factor_spinbox.setValue(0)  # для обычного режима elite_factor=0


    def change_vertex_count(self, value):
        self.num_vertices = value
        if self.num_vertices == 3 and self.table.rowCount() == 0:
            self.fill_default_example()
        self.update_edges_matrix_text()

    def add_edge_row(self):
        current_rows = self.table.rowCount()
        self.table.insertRow(current_rows)
        self.table.setItem(current_rows, 0, QTableWidgetItem("0"))
        self.table.setItem(current_rows, 1, QTableWidgetItem("1"))
        self.table.setItem(current_rows, 2, QTableWidgetItem("10"))
        self.update_edges_matrix_text()

    def delete_selected_edge_rows(self):
        selected_rows = list(set([item.row() for item in self.table.selectedItems()]))
        selected_rows.sort(reverse=True)
        for row in selected_rows:
            self.table.removeRow(row)
        self.update_edges_matrix_text()

    def fill_default_example(self):
        if self.num_vertices == 3 and self.table.rowCount() == 0:
            for (u, v, w) in self.default_edges:
                r = self.table.rowCount()
                self.table.insertRow(r)
                self.table.setItem(r, 0, QTableWidgetItem(str(u)))
                self.table.setItem(r, 1, QTableWidgetItem(str(v)))
                self.table.setItem(r, 2, QTableWidgetItem(str(w)))
            self.update_edges_matrix_text()

    def read_edge_list_from_table(self):
        edges = []
        for row in range(self.table.rowCount()):
            try:
                source = int(self.table.item(row, 0).text())
                target = int(self.table.item(row, 1).text())
                weight = float(self.table.item(row, 2).text())
            except Exception:
                continue
            edges.append((source, target, weight))
        return edges

    def read_graph_from_table(self):
        n = self.num_vertices
        graph = [[NO_EDGE if i != j else 0 for j in range(n)] for i in range(n)]
        for (source, target, weight) in self.read_edge_list_from_table():
            if 0 <= source < n and 0 <= target < n and source != target:
                graph[source][target] = min(graph[source][target], weight)
        return graph

    def update_edges_matrix_text(self):
        edges = self.read_edge_list_from_table()
        def format_val(val):
            if isinstance(val, float) and val.is_integer():
                return str(int(val))
            return str(val)
        lines = []
        for edge in edges:
            lines.append(" ".join(format_val(x) for x in edge))
        self.edgesMatrixEdit.setPlainText("\n".join(lines))

    def apply_edges_matrix_text(self):
        text = self.edgesMatrixEdit.toPlainText().strip()
        if not text:
            return
        lines = text.splitlines()
        self.table.setRowCount(0)
        for line in lines:
            parts = line.split()
            if len(parts) != 3:
                continue
            try:
                source = int(parts[0])
                target = int(parts[1])
                weight = float(parts[2])
            except Exception:
                continue
            if 0 <= source < self.num_vertices and 0 <= target < self.num_vertices and source != target:
                r = self.table.rowCount()
                self.table.insertRow(r)
                self.table.setItem(r, 0, QTableWidgetItem(str(source)))
                self.table.setItem(r, 1, QTableWidgetItem(str(target)))
                self.table.setItem(r, 2, QTableWidgetItem(str(weight)))
        self.update_edges_matrix_text()

    def update_graphs(self, cycle_path=None):
        edge_list = self.read_edge_list_from_table()
        draw_graph(self.input_ax, edge_list, self.num_vertices, title="Входной граф")
        self.input_canvas.draw()
        draw_graph(self.output_ax, edge_list, self.num_vertices, cycle_path, title="Выходной граф")
        self.output_canvas.draw()

    def calculate(self):
        graph_matrix = self.read_graph_from_table()
        start_time = time.time()

        current_tab = self.tabs.currentWidget()
        if current_tab == self.nn_tab:
            multi_start = self.checkbox_multi_start.isChecked()
            two_step = self.checkbox_two_step.isChecked()
            random_mod = self.checkbox_random.isChecked()
            random_prob = 0.2

            if random_mod and two_step:
                if multi_start:
                    cycle_path, total_distance = modified_nearest_neighbor_two_step_random(graph_matrix, random_prob, NO_EDGE)
                else:
                    cycle_path, total_distance = nearest_neighbor_two_step_random(graph_matrix, start=0, random_prob=random_prob, NO_EDGE=NO_EDGE)
            elif random_mod:
                if multi_start:
                    cycle_path, total_distance = modified_nearest_neighbor_random(graph_matrix, random_prob, NO_EDGE)
                else:
                    cycle_path, total_distance = nearest_neighbor_random(graph_matrix, start=0, random_prob=random_prob, NO_EDGE=NO_EDGE)
            elif two_step:
                if multi_start:
                    cycle_path, total_distance = modified_nearest_neighbor_two_step(graph_matrix, NO_EDGE)
                else:
                    cycle_path, total_distance = nearest_neighbor_two_step(graph_matrix, start=0, NO_EDGE=NO_EDGE)
            elif multi_start:
                cycle_path, total_distance = modified_nearest_neighbor(graph_matrix, NO_EDGE)
            else:
                cycle_path, total_distance = nearest_neighbor(graph_matrix, start=0, NO_EDGE=NO_EDGE)

            mod_type_str = "NN: "
            mods = []
            if multi_start: mods.append("многостарт")
            if two_step: mods.append("двухшаговое предвидение")
            if random_mod: mods.append("рандом")
            mod_type_str += ", ".join(mods) if mods else "без модификаций"

        elif current_tab == self.sa_tab:
            if self.sa_start_method.currentText() == "Вручную":
                manual_path_text = self.sa_manual_path_lineedit.text().strip()
                if manual_path_text:
                    try:
                        manual_path = [int(x) for x in manual_path_text.split(',')]
                        # Если путь не замкнут (не содержит повтор первого элемента в конце), добавляем
                        if manual_path[0] != manual_path[-1]:
                            manual_path.append(manual_path[0])
                    except Exception as e:
                        QMessageBox.warning(self, "Ошибка", "Неверный формат начального пути! Введите номера вершин через запятую.")
                        return
                else:
                    QMessageBox.warning(self, "Ошибка", "Необходимо ввести начальный путь вручную!")
                    return
            else:
                manual_path = None
            
            use_nn_start = True if self.sa_start_method.currentText() == "Ближайший сосед" else False
            sa_variant = "standard" if self.sa_algorithm.currentText() == "Стандартный отжиг" else "boltzmann"
            start_temp = self.sa_start_temp_spinbox.value()
            end_temp = self.sa_end_temp_spinbox.value()
            alpha = self.sa_alpha_spinbox.value()
            max_iter = self.sa_max_iter_spinbox.value()

            init_path, final_path, total_distance = simulated_annealing(
                graph_matrix,
                start_temp=start_temp,
                end_temp=end_temp,
                alpha=alpha,
                max_iter=max_iter,
                use_nn_start=use_nn_start,
                sa_variant=sa_variant,
                initial_path=manual_path
            )
            mod_type_str = (f"SA: {self.sa_algorithm.currentText()}, "
                            f"начальный путь: {self.sa_start_method.currentText()}, "
                            f"T0={start_temp}, T_end={end_temp}, α={alpha}, max_iter={max_iter}")

            initial_path_str = "Начальный путь: " + ",".join(str(v) for v in init_path)
            final_path_str = "Финальный путь: " + ",".join(str(v) for v in final_path)
            sa_paths_log = initial_path_str + " | " + final_path_str

            initial_path_str = "Начальный путь: " + ",".join(str(v) for v in init_path)
            final_path_str = "Финальный путь: " + ",".join(str(v) for v in final_path)

            sa_paths_log = initial_path_str + " | " + final_path_str

        elif current_tab == self.aco_tab:
            num_ants = self.aco_num_ants_spinbox.value()
            num_iterations = self.aco_num_iterations_spinbox.value()
            decay = self.aco_decay_spinbox.value()
            alpha_val = self.aco_alpha_spinbox.value()
            beta_val = self.aco_beta_spinbox.value()
            mode = self.aco_mode_combobox.currentText()
            if mode == "Элитные":
                elite_factor = self.aco_elite_factor_spinbox.value()
            else:
                elite_factor = 0  # для обычного алгоритма elite_factor=0 (т.е. без элитной модификации)

            path_str, total_distance = ant_colony_optimization(
                graph_matrix,
                num_ants=num_ants,
                num_iterations=num_iterations,
                decay=decay,
                alpha=alpha_val,
                beta=beta_val,
                elite_factor=elite_factor
            )
            mod_type_str = (f"ACO: {mode} муравьи, антов: {num_ants}, итераций: {num_iterations}, "
                            f"decay={decay}, alpha={alpha_val}, beta={beta_val}, elite_factor={elite_factor}")
            cycle_path = path_str
            
        else:
            cycle_path, total_distance = None, None
            mod_type_str = "Не определён"

        end_time = time.time()
        elapsed_time = end_time - start_time

        if current_tab == self.sa_tab:
            log_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = (f"[{log_time}] Run type: {mod_type_str} | {sa_paths_log} | "
                         f"Distance: {total_distance if total_distance is not None else 'N/A'} | "
                         f"Time: {elapsed_time:.6f} sec")
            print(log_entry)
            with open("nearest_neighbor_log.txt", "a", encoding="utf-8") as log_file:
                log_file.write(log_entry + "\n")

            self.cycle_label.setText(initial_path_str + "\n" + final_path_str)
            self.distance_label.setText("Расстояние: " + str(total_distance))
            self.update_graphs(final_path)
        else:
            log_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            path_str = ",".join(str(v) for v in cycle_path) if cycle_path else "путь не найден"
            log_entry = (f"[{log_time}] Run type: {mod_type_str} | Path: {path_str} | "
                         f"Distance: {total_distance if total_distance is not None else 'N/A'} | "
                         f"Time: {elapsed_time:.6f} sec")
            print(log_entry)
            with open("nearest_neighbor_log.txt", "a", encoding="utf-8") as log_file:
                log_file.write(log_entry + "\n")
            if cycle_path is None or total_distance is None:
                self.cycle_label.setText("Путь: Не удалось найти путь")
                self.distance_label.setText("")
                self.update_graphs()
            else:
                self.cycle_label.setText("Путь: " + ",".join(str(v) for v in cycle_path))
                self.distance_label.setText("Расстояние: " + str(total_distance))
                self.update_graphs(cycle_path)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWin = MainWindow()
    mainWin.show()
    sys.exit(app.exec_())
