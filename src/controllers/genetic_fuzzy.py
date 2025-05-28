import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import random

class GeneticFuzzyController:
    def __init__(self, population_size=50, mutation_rate=0.1, elite_size=5):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size
        
        # Parâmetros do sistema fuzzy
        self.angle_range = np.arange(-2*np.pi, 2*np.pi, 0.01)
        self.angular_velocity_range = np.arange(-5, 5, 0.1)
        self.force_range = np.arange(-20, 20, 0.1)
        
        # Inicializa a população
        self.population = self._initialize_population()
        self.best_individual = None
        self.best_fitness = float('-inf')
        
        # Sistema fuzzy atual
        self._initialize_fuzzy_system()
        
    def _initialize_population(self):
        """Inicializa a população de indivíduos"""
        population = []
        for _ in range(self.population_size):
            individual = {
                # Centros dos conjuntos fuzzy para ângulo
                'angle_centers': np.random.uniform(-np.pi/2, np.pi/2, 5),
                # Larguras dos conjuntos fuzzy para ângulo
                'angle_widths': np.random.uniform(0.1, np.pi/4, 5),
                # Centros dos conjuntos fuzzy para velocidade angular
                'velocity_centers': np.random.uniform(-5, 5, 5),
                # Larguras dos conjuntos fuzzy para velocidade angular
                'velocity_widths': np.random.uniform(0.5, 2.5, 5),
                # Pesos das regras
                'rule_weights': np.random.uniform(-1, 1, 25)
            }
            population.append(individual)
        return population
    
    def _initialize_fuzzy_system(self):
        """Inicializa o sistema fuzzy com os parâmetros do melhor indivíduo"""
        if self.best_individual is None:
            self.best_individual = self.population[0]
            
        try:
            # Conjuntos fuzzy para ângulo
            self.angle = ctrl.Antecedent(self.angle_range, 'angle')
            for i, (center, width) in enumerate(zip(self.best_individual['angle_centers'], 
                                                 self.best_individual['angle_widths'])):
                self.angle[f'set_{i}'] = fuzz.gaussmf(self.angle_range, center, width)
            
            # Conjuntos fuzzy para velocidade angular
            self.angular_velocity = ctrl.Antecedent(self.angular_velocity_range, 'angular_velocity')
            for i, (center, width) in enumerate(zip(self.best_individual['velocity_centers'],
                                                 self.best_individual['velocity_widths'])):
                self.angular_velocity[f'set_{i}'] = fuzz.gaussmf(self.angular_velocity_range, center, width)
            
            # Conjuntos fuzzy para força
            self.force = ctrl.Consequent(self.force_range, 'force')
            self.force['negative'] = fuzz.trimf(self.force_range, [-20, -10, 0])
            self.force['zero'] = fuzz.trimf(self.force_range, [-10, 0, 10])
            self.force['positive'] = fuzz.trimf(self.force_range, [0, 10, 20])
            
            # Regras fuzzy
            rules = []
            rule_idx = 0
            for i in range(5):  # Para cada conjunto de ângulo
                for j in range(5):  # Para cada conjunto de velocidade
                    weight = self.best_individual['rule_weights'][rule_idx]
                    if weight > 0:
                        rules.append(ctrl.Rule(
                            self.angle[f'set_{i}'] & self.angular_velocity[f'set_{j}'],
                            self.force['positive']
                        ))
                    else:
                        rules.append(ctrl.Rule(
                            self.angle[f'set_{i}'] & self.angular_velocity[f'set_{j}'],
                            self.force['negative']
                        ))
                    rule_idx += 1
            
            # Sistema de controle
            self.control_system = ctrl.ControlSystem(rules)
            self.simulation = ctrl.ControlSystemSimulation(self.control_system)
            
        except Exception as e:
            print(f"Erro ao inicializar sistema fuzzy: {str(e)}")
            # Reinicializa com valores padrão
            self.best_individual = {
                'angle_centers': np.array([-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2]),
                'angle_widths': np.array([np.pi/4, np.pi/4, np.pi/4, np.pi/4, np.pi/4]),
                'velocity_centers': np.array([-5, -2.5, 0, 2.5, 5]),
                'velocity_widths': np.array([2.5, 2.5, 2.5, 2.5, 2.5]),
                'rule_weights': np.zeros(25)
            }
            # Tenta inicializar novamente com os valores padrão
            try:
                self._initialize_fuzzy_system()
            except Exception as e:
                print(f"Erro fatal ao inicializar sistema fuzzy: {str(e)}")
                raise
    
    def compute_control(self, angle, angular_velocity):
        """
        Computa a força de controle usando o sistema fuzzy otimizado
        """
        try:
            # Limita apenas a velocidade angular
            angular_velocity = np.clip(angular_velocity, -10, 10)
            
            self.simulation.input['angle'] = angle
            self.simulation.input['angular_velocity'] = angular_velocity
            self.simulation.compute()
            
            # Limita a força de saída
            return np.clip(self.simulation.output['force'], -20, 20)
            
        except Exception as e:
            print(f"Erro no controlador Genetic-Fuzzy: {str(e)}")
            return 0.0
    
    def evaluate_fitness(self, individual, test_cases):
        """
        Avalia o fitness de um indivíduo usando casos de teste
        """
        try:
            # Configura o sistema fuzzy com os parâmetros do indivíduo
            self.best_individual = individual
            self._initialize_fuzzy_system()
            
            total_error = 0
            for angle, velocity, target_force in test_cases:
                # Simula o sistema para cada caso
                force = self.compute_control(angle, velocity)
                # Penaliza o desvio do pêndulo e do carrinho
                # Aqui, supomos que o objetivo é manter o pêndulo em pé (angle ~ 0) e o carrinho no centro (posição ~ 0)
                # Como não temos a posição do carrinho no teste, penalizamos apenas o ângulo e a força aplicada
                error = abs(angle) + 0.1 * abs(force)  # Peso maior para o ângulo
                total_error += error
            # Fitness é o inverso do erro total
            return 1.0 / (1.0 + total_error)
            
        except Exception as e:
            print(f"Erro na avaliação do fitness: {str(e)}")
            return 0.0
    
    def crossover(self, parent1, parent2):
        """Realiza o crossover entre dois indivíduos"""
        child = {}
        for key in parent1.keys():
            if random.random() < 0.5:
                child[key] = parent1[key].copy()
            else:
                child[key] = parent2[key].copy()
        return child
    
    def mutate(self, individual):
        """Aplica mutação em um indivíduo"""
        mutated = individual.copy()
        for key in individual.keys():
            if random.random() < self.mutation_rate:
                if 'centers' in key:
                    mutated[key] += np.random.normal(0, 0.1, size=individual[key].shape)
                elif 'widths' in key:
                    mutated[key] += np.random.normal(0, 0.05, size=individual[key].shape)
                else:  # rule_weights
                    mutated[key] += np.random.normal(0, 0.2, size=individual[key].shape)
        return mutated
    
    def evolve(self, test_cases):
        """
        Realiza uma geração de evolução
        """
        # Avalia todos os indivíduos
        fitness_scores = [self.evaluate_fitness(ind, test_cases) for ind in self.population]
        
        # Encontra o melhor indivíduo
        best_idx = np.argmax(fitness_scores)
        if fitness_scores[best_idx] > self.best_fitness:
            self.best_fitness = fitness_scores[best_idx]
            self.best_individual = self.population[best_idx].copy()
            self._initialize_fuzzy_system()
        
        # Seleciona os melhores indivíduos (elite)
        elite_indices = np.argsort(fitness_scores)[-self.elite_size:]
        new_population = [self.population[i].copy() for i in elite_indices]
        
        # Completa a nova população com crossover e mutação
        while len(new_population) < self.population_size:
            # Seleção por torneio
            idx1 = random.randint(0, len(self.population)-1)
            idx2 = random.randint(0, len(self.population)-1)
            parent1 = self.population[idx1]
            parent2 = self.population[idx2]
            
            # Crossover
            child = self.crossover(parent1, parent2)
            
            # Mutação
            child = self.mutate(child)
            
            new_population.append(child)
        
        self.population = new_population 
    
    def update_parameters(self, population_size=None, mutation_rate=None, elite_size=None):
        """Atualiza os parâmetros do controlador"""
        if population_size is not None:
            self.population_size = population_size
            self.population = self._initialize_population()
            
        if mutation_rate is not None:
            self.mutation_rate = mutation_rate
            
        if elite_size is not None:
            self.elite_size = elite_size
            
        # Reinicializa o sistema fuzzy com os parâmetros atuais
        self._initialize_fuzzy_system() 