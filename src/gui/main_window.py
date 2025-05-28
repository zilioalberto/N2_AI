from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QPushButton, 
                            QLabel, QHBoxLayout, QComboBox, QSpinBox, QDoubleSpinBox,
                            QGroupBox, QSlider, QStackedWidget)
from PyQt5.QtCore import Qt, QTimer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import numpy as np

from src.simulation.pendulum_sim import PendulumSimulation
from src.controllers.fis_controller import FISController
from src.controllers.neuro_fuzzy import NeuroFuzzyController
from src.controllers.genetic_fuzzy import GeneticFuzzyController

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Controle de Pêndulo Invertido")
        self.setGeometry(100, 100, 1200, 800)
        
        # Widget central
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Layout principal
        layout = QHBoxLayout()
        central_widget.setLayout(layout)
        
        # Painel de controle
        control_panel = QWidget()
        control_layout = QVBoxLayout()
        control_panel.setLayout(control_layout)
        control_panel.setFixedWidth(300)
        
        # Título
        title = QLabel("Sistema de Controle de Pêndulo Invertido")
        title.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(title)
        
        # Seleção do controlador
        controller_group = QGroupBox("Controlador")
        controller_layout = QVBoxLayout()
        controller_group.setLayout(controller_layout)
        
        controller_label = QLabel("Tipo:")
        self.controller_combo = QComboBox()
        self.controller_combo.addItems(["FIS", "Neuro-Fuzzy", "Genetic-Fuzzy"])
        controller_layout.addWidget(controller_label)
        controller_layout.addWidget(self.controller_combo)
        
        # Widget empilhado para os parâmetros dos controladores
        self.controller_params_stack = QStackedWidget()
        
        # Parâmetros do FIS
        fis_widget = QWidget()
        fis_layout = QVBoxLayout()
        fis_widget.setLayout(fis_layout)
        
        # Ganho do FIS
        gain_layout = QHBoxLayout()
        gain_label = QLabel("Ganho:")
        self.gain_spin = QDoubleSpinBox()
        self.gain_spin.setRange(0.1, 2.0)
        self.gain_spin.setValue(0.5)
        self.gain_spin.setSingleStep(0.1)
        gain_layout.addWidget(gain_label)
        gain_layout.addWidget(self.gain_spin)
        fis_layout.addLayout(gain_layout)
        
        # Limites dos conjuntos fuzzy
        fuzzy_limits_group = QGroupBox("Limites Fuzzy")
        fuzzy_limits_layout = QVBoxLayout()
        fuzzy_limits_group.setLayout(fuzzy_limits_layout)
        
        # Ângulo
        angle_layout = QHBoxLayout()
        angle_label = QLabel("Ângulo (±):")
        self.angle_spin = QDoubleSpinBox()
        self.angle_spin.setRange(0.1, np.pi)
        self.angle_spin.setValue(np.pi/2)
        self.angle_spin.setSingleStep(0.1)
        angle_layout.addWidget(angle_label)
        angle_layout.addWidget(self.angle_spin)
        fuzzy_limits_layout.addLayout(angle_layout)
        
        # Velocidade angular
        velocity_layout = QHBoxLayout()
        velocity_label = QLabel("Vel. Angular (±):")
        self.velocity_spin = QDoubleSpinBox()
        self.velocity_spin.setRange(1, 10)
        self.velocity_spin.setValue(5)
        self.velocity_spin.setSingleStep(0.5)
        velocity_layout.addWidget(velocity_label)
        velocity_layout.addWidget(self.velocity_spin)
        fuzzy_limits_layout.addLayout(velocity_layout)
        
        # Força
        force_layout = QHBoxLayout()
        force_label = QLabel("Força (±):")
        self.force_spin = QDoubleSpinBox()
        self.force_spin.setRange(5, 20)
        self.force_spin.setValue(10)
        self.force_spin.setSingleStep(1)
        force_layout.addWidget(force_label)
        force_layout.addWidget(self.force_spin)
        fuzzy_limits_layout.addLayout(force_layout)
        
        fis_layout.addWidget(fuzzy_limits_group)
        
        # Parâmetros do Neuro-Fuzzy
        neuro_fuzzy_widget = QWidget()
        neuro_fuzzy_layout = QVBoxLayout()
        neuro_fuzzy_widget.setLayout(neuro_fuzzy_layout)
        
        # Taxa de aprendizado
        lr_layout = QHBoxLayout()
        lr_label = QLabel("Taxa de Aprendizado:")
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.001, 0.1)
        self.lr_spin.setValue(0.01)
        self.lr_spin.setSingleStep(0.001)
        lr_layout.addWidget(lr_label)
        lr_layout.addWidget(self.lr_spin)
        neuro_fuzzy_layout.addLayout(lr_layout)
        
        # Número de regras
        rules_layout = QHBoxLayout()
        rules_label = QLabel("Número de Regras:")
        self.rules_spin = QSpinBox()
        self.rules_spin.setRange(4, 16)
        self.rules_spin.setValue(9)
        self.rules_spin.setSingleStep(1)
        rules_layout.addWidget(rules_label)
        rules_layout.addWidget(self.rules_spin)
        neuro_fuzzy_layout.addLayout(rules_layout)
        
        # Parâmetros do Genetic-Fuzzy
        genetic_fuzzy_widget = QWidget()
        genetic_fuzzy_layout = QVBoxLayout()
        genetic_fuzzy_widget.setLayout(genetic_fuzzy_layout)
        
        # Tamanho da população
        pop_layout = QHBoxLayout()
        pop_label = QLabel("Tamanho da População:")
        self.pop_spin = QSpinBox()
        self.pop_spin.setRange(20, 200)
        self.pop_spin.setValue(50)
        self.pop_spin.setSingleStep(10)
        pop_layout.addWidget(pop_label)
        pop_layout.addWidget(self.pop_spin)
        genetic_fuzzy_layout.addLayout(pop_layout)
        
        # Taxa de mutação
        mut_layout = QHBoxLayout()
        mut_label = QLabel("Taxa de Mutação:")
        self.mut_spin = QDoubleSpinBox()
        self.mut_spin.setRange(0.01, 0.5)
        self.mut_spin.setValue(0.1)
        self.mut_spin.setSingleStep(0.01)
        mut_layout.addWidget(mut_label)
        mut_layout.addWidget(self.mut_spin)
        genetic_fuzzy_layout.addLayout(mut_layout)
        
        # Tamanho da elite
        elite_layout = QHBoxLayout()
        elite_label = QLabel("Tamanho da Elite:")
        self.elite_spin = QSpinBox()
        self.elite_spin.setRange(1, 20)
        self.elite_spin.setValue(5)
        self.elite_spin.setSingleStep(1)
        elite_layout.addWidget(elite_label)
        elite_layout.addWidget(self.elite_spin)
        genetic_fuzzy_layout.addLayout(elite_layout)
        
        # Botão de evolução
        self.evolve_button = QPushButton("Evoluir")
        genetic_fuzzy_layout.addWidget(self.evolve_button)
        
        # Adiciona os widgets ao stack
        self.controller_params_stack.addWidget(fis_widget)
        self.controller_params_stack.addWidget(neuro_fuzzy_widget)
        self.controller_params_stack.addWidget(genetic_fuzzy_widget)
        
        # Parâmetros do pêndulo
        pendulum_group = QGroupBox("Parâmetros do Pêndulo")
        pendulum_layout = QVBoxLayout()
        pendulum_group.setLayout(pendulum_layout)
        
        # Massa
        mass_layout = QHBoxLayout()
        mass_label = QLabel("Massa:")
        self.mass_spin = QDoubleSpinBox()
        self.mass_spin.setRange(0.1, 10.0)
        self.mass_spin.setValue(1.0)
        self.mass_spin.setSingleStep(0.1)
        mass_layout.addWidget(mass_label)
        mass_layout.addWidget(self.mass_spin)
        pendulum_layout.addLayout(mass_layout)
        
        # Comprimento
        length_layout = QHBoxLayout()
        length_label = QLabel("Comprimento:")
        self.length_spin = QDoubleSpinBox()
        self.length_spin.setRange(0.1, 5.0)
        self.length_spin.setValue(1.0)
        self.length_spin.setSingleStep(0.1)
        length_layout.addWidget(length_label)
        length_layout.addWidget(self.length_spin)
        pendulum_layout.addLayout(length_layout)
        
        # Botões de controle
        self.start_button = QPushButton("Iniciar")
        self.stop_button = QPushButton("Parar")
        self.reset_button = QPushButton("Resetar")
        
        # Adiciona todos os grupos ao painel de controle
        control_layout.addWidget(controller_group)
        control_layout.addWidget(self.controller_params_stack)
        control_layout.addWidget(pendulum_group)
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.stop_button)
        control_layout.addWidget(self.reset_button)
        
        # Adiciona o painel de controle ao layout principal
        layout.addWidget(control_panel)
        
        # Área de visualização
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        # Inicialização dos sistemas
        self.simulation = None
        self.controller = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_simulation)
        
        # Conecta os sinais
        self.start_button.clicked.connect(self.start_simulation)
        self.stop_button.clicked.connect(self.stop_simulation)
        self.reset_button.clicked.connect(self.reset_simulation)
        self.controller_combo.currentTextChanged.connect(self.change_controller)
        self.evolve_button.clicked.connect(self.evolve_controller)
        
        # Conecta os sinais dos parâmetros
        self.gain_spin.valueChanged.connect(self.update_controller_params)
        self.angle_spin.valueChanged.connect(self.update_controller_params)
        self.velocity_spin.valueChanged.connect(self.update_controller_params)
        self.force_spin.valueChanged.connect(self.update_controller_params)
        self.lr_spin.valueChanged.connect(self.update_controller_params)
        self.rules_spin.valueChanged.connect(self.update_controller_params)
        self.pop_spin.valueChanged.connect(self.update_controller_params)
        self.mut_spin.valueChanged.connect(self.update_controller_params)
        self.elite_spin.valueChanged.connect(self.update_controller_params)
        self.mass_spin.valueChanged.connect(self.update_simulation_params)
        self.length_spin.valueChanged.connect(self.update_simulation_params)
        
        # Inicializa o sistema
        self.initialize_systems()
        
    def initialize_systems(self):
        """Inicializa os sistemas de simulação e controle"""
        self.simulation = PendulumSimulation(
            mass=self.mass_spin.value(),
            length=self.length_spin.value()
        )
        self.change_controller(self.controller_combo.currentText())
        
    def change_controller(self, controller_name):
        """Muda o controlador atual"""
        if controller_name == "FIS":
            self.controller = FISController()
            self.controller_params_stack.setCurrentIndex(0)
            self.update_controller_params()
        elif controller_name == "Neuro-Fuzzy":
            self.controller = NeuroFuzzyController(
                learning_rate=self.lr_spin.value(),
                num_rules=self.rules_spin.value()
            )
            self.controller_params_stack.setCurrentIndex(1)
        elif controller_name == "Genetic-Fuzzy":
            self.controller = GeneticFuzzyController(
                population_size=self.pop_spin.value(),
                mutation_rate=self.mut_spin.value(),
                elite_size=self.elite_spin.value()
            )
            self.controller_params_stack.setCurrentIndex(2)
            
    def update_controller_params(self):
        """Atualiza os parâmetros do controlador"""
        if isinstance(self.controller, FISController):
            self.controller.update_parameters(
                gain=self.gain_spin.value(),
                angle_range=self.angle_spin.value(),
                velocity_range=self.velocity_spin.value(),
                force_range=self.force_spin.value()
            )
        elif isinstance(self.controller, NeuroFuzzyController):
            self.controller.learning_rate = self.lr_spin.value()
            self.controller.num_rules = self.rules_spin.value()
        elif isinstance(self.controller, GeneticFuzzyController):
            self.controller.update_parameters(
                population_size=self.pop_spin.value(),
                mutation_rate=self.mut_spin.value(),
                elite_size=self.elite_spin.value()
            )
            
    def evolve_controller(self):
        """Realiza uma geração de evolução do controlador Genetic-Fuzzy"""
        if isinstance(self.controller, GeneticFuzzyController):
            # Gera casos de teste para evolução
            test_cases = []
            for _ in range(100):
                angle = np.random.uniform(-np.pi/2, np.pi/2)
                velocity = np.random.uniform(-5, 5)
                # Força alvo baseada em um controlador PID simples
                target_force = -2 * angle - 1 * velocity
                test_cases.append((angle, velocity, target_force))
            
            # Realiza a evolução
            self.controller.evolve(test_cases)
            
    def update_simulation_params(self):
        """Atualiza os parâmetros da simulação"""
        if self.simulation:
            self.simulation.mass = self.mass_spin.value()
            self.simulation.length = self.length_spin.value()
            
    def start_simulation(self):
        """Inicia a simulação"""
        if not self.timer.isActive():
            self.timer.start(10)  # 10ms = 100Hz
            
    def stop_simulation(self):
        """Para a simulação"""
        self.timer.stop()
        
    def reset_simulation(self):
        """Reseta a simulação"""
        self.stop_simulation()
        self.initialize_systems()
        self.update_plot()
        
    def update_simulation(self):
        """Atualiza a simulação em um passo"""
        try:
            state = self.simulation.update(
                self.controller.compute_control(
                    self.simulation.angle,
                    self.simulation.angular_velocity
                )
            )
            self.update_plot()
        except Exception as e:
            print(f"Erro na simulação: {str(e)}")
            self.stop_simulation()
        
    def update_plot(self):
        """Atualiza o gráfico do pêndulo"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Desenha o pêndulo
        x = self.simulation.cart_position
        y = 0
        pendulum_x = x + self.simulation.length * np.sin(self.simulation.angle)
        pendulum_y = -self.simulation.length * np.cos(self.simulation.angle)
        
        # Desenha trilhos
        ax.plot([-10, 10], [0, 0], 'k--', linewidth=1)
        
        # Desenha o carrinho
        ax.plot([x-0.5, x+0.5], [y, y], 'b-', linewidth=4)
        
        # Desenha o pêndulo
        ax.plot([x, pendulum_x], [y, pendulum_y], 'r-', linewidth=2)
        
        # Configura o gráfico fixo
        ax.set_xlim(-10, 10)
        ax.set_ylim(-7, 7)
        ax.set_aspect('equal')
        ax.grid(True)
        
        self.canvas.draw() 