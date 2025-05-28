import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

class FISController:
    def __init__(self):
        # Fator de ganho para ajuste fino do controle
        self.gain = 0.5
        
        # Universo de discurso mais preciso próximo do zero
        self.angle_range = np.arange(-np.pi/2, np.pi/2, 0.01)  # Reduzido para melhor precisão
        self.angular_velocity_range = np.arange(-5, 5, 0.1)    # Reduzido para melhor controle
        self.force_range = np.arange(-10, 10, 0.1)            # Força mais suave
        
        self._initialize_fuzzy_system()
    
    def _initialize_fuzzy_system(self):
        """Inicializa ou reinicializa o sistema fuzzy com os parâmetros atuais"""
        # Conjuntos fuzzy para ângulo - mais precisos próximos do zero
        self.angle = ctrl.Antecedent(self.angle_range, 'angle')
        self.angle['negative_large'] = fuzz.trimf(self.angle_range, [-np.pi/2, -np.pi/4, -np.pi/8])
        self.angle['negative_small'] = fuzz.trimf(self.angle_range, [-np.pi/4, -np.pi/8, 0])
        self.angle['zero'] = fuzz.trimf(self.angle_range, [-np.pi/8, 0, np.pi/8])
        self.angle['positive_small'] = fuzz.trimf(self.angle_range, [0, np.pi/8, np.pi/4])
        self.angle['positive_large'] = fuzz.trimf(self.angle_range, [np.pi/8, np.pi/4, np.pi/2])
        
        # Conjuntos fuzzy para velocidade angular - mais precisos próximos do zero
        self.angular_velocity = ctrl.Antecedent(self.angular_velocity_range, 'angular_velocity')
        self.angular_velocity['negative_large'] = fuzz.trimf(self.angular_velocity_range, [-5, -2.5, -1])
        self.angular_velocity['negative_small'] = fuzz.trimf(self.angular_velocity_range, [-2.5, -1, 0])
        self.angular_velocity['zero'] = fuzz.trimf(self.angular_velocity_range, [-1, 0, 1])
        self.angular_velocity['positive_small'] = fuzz.trimf(self.angular_velocity_range, [0, 1, 2.5])
        self.angular_velocity['positive_large'] = fuzz.trimf(self.angular_velocity_range, [1, 2.5, 5])
        
        # Conjuntos fuzzy para força - mais suaves
        self.force = ctrl.Consequent(self.force_range, 'force')
        self.force['negative_large'] = fuzz.trimf(self.force_range, [-10, -5, -2.5])
        self.force['negative_small'] = fuzz.trimf(self.force_range, [-5, -2.5, 0])
        self.force['zero'] = fuzz.trimf(self.force_range, [-2.5, 0, 2.5])
        self.force['positive_small'] = fuzz.trimf(self.force_range, [0, 2.5, 5])
        self.force['positive_large'] = fuzz.trimf(self.force_range, [2.5, 5, 10])
        
        # Regras fuzzy - mais ênfase no controle próximo do equilíbrio
        rules = []
        
        # Regras para ângulo negativo grande
        rules.append(ctrl.Rule(self.angle['negative_large'] & self.angular_velocity['negative_large'], self.force['positive_large']))
        rules.append(ctrl.Rule(self.angle['negative_large'] & self.angular_velocity['negative_small'], self.force['positive_large']))
        rules.append(ctrl.Rule(self.angle['negative_large'] & self.angular_velocity['zero'], self.force['positive_small']))
        rules.append(ctrl.Rule(self.angle['negative_large'] & self.angular_velocity['positive_small'], self.force['zero']))
        rules.append(ctrl.Rule(self.angle['negative_large'] & self.angular_velocity['positive_large'], self.force['negative_small']))
        
        # Regras para ângulo negativo pequeno - mais suave
        rules.append(ctrl.Rule(self.angle['negative_small'] & self.angular_velocity['negative_large'], self.force['positive_small']))
        rules.append(ctrl.Rule(self.angle['negative_small'] & self.angular_velocity['negative_small'], self.force['positive_small']))
        rules.append(ctrl.Rule(self.angle['negative_small'] & self.angular_velocity['zero'], self.force['positive_small']))
        rules.append(ctrl.Rule(self.angle['negative_small'] & self.angular_velocity['positive_small'], self.force['zero']))
        rules.append(ctrl.Rule(self.angle['negative_small'] & self.angular_velocity['positive_large'], self.force['negative_small']))
        
        # Regras para ângulo zero - controle mais preciso
        rules.append(ctrl.Rule(self.angle['zero'] & self.angular_velocity['negative_large'], self.force['positive_small']))
        rules.append(ctrl.Rule(self.angle['zero'] & self.angular_velocity['negative_small'], self.force['positive_small']))
        rules.append(ctrl.Rule(self.angle['zero'] & self.angular_velocity['zero'], self.force['zero']))
        rules.append(ctrl.Rule(self.angle['zero'] & self.angular_velocity['positive_small'], self.force['negative_small']))
        rules.append(ctrl.Rule(self.angle['zero'] & self.angular_velocity['positive_large'], self.force['negative_small']))
        
        # Regras para ângulo positivo pequeno - mais suave
        rules.append(ctrl.Rule(self.angle['positive_small'] & self.angular_velocity['negative_large'], self.force['positive_small']))
        rules.append(ctrl.Rule(self.angle['positive_small'] & self.angular_velocity['negative_small'], self.force['zero']))
        rules.append(ctrl.Rule(self.angle['positive_small'] & self.angular_velocity['zero'], self.force['negative_small']))
        rules.append(ctrl.Rule(self.angle['positive_small'] & self.angular_velocity['positive_small'], self.force['negative_small']))
        rules.append(ctrl.Rule(self.angle['positive_small'] & self.angular_velocity['positive_large'], self.force['negative_small']))
        
        # Regras para ângulo positivo grande
        rules.append(ctrl.Rule(self.angle['positive_large'] & self.angular_velocity['negative_large'], self.force['positive_small']))
        rules.append(ctrl.Rule(self.angle['positive_large'] & self.angular_velocity['negative_small'], self.force['zero']))
        rules.append(ctrl.Rule(self.angle['positive_large'] & self.angular_velocity['zero'], self.force['negative_small']))
        rules.append(ctrl.Rule(self.angle['positive_large'] & self.angular_velocity['positive_small'], self.force['negative_small']))
        rules.append(ctrl.Rule(self.angle['positive_large'] & self.angular_velocity['positive_large'], self.force['negative_large']))
        
        # Sistema de controle
        self.control_system = ctrl.ControlSystem(rules)
        self.simulation = ctrl.ControlSystemSimulation(self.control_system)
    
    def update_parameters(self, gain=None, angle_range=None, velocity_range=None, force_range=None):
        """Atualiza os parâmetros do controlador"""
        if gain is not None:
            self.gain = gain
        if angle_range is not None:
            self.angle_range = np.arange(-angle_range, angle_range, 0.01)
        if velocity_range is not None:
            self.angular_velocity_range = np.arange(-velocity_range, velocity_range, 0.1)
        if force_range is not None:
            self.force_range = np.arange(-force_range, force_range, 0.1)
        
        # Reinicializa o sistema fuzzy com os novos parâmetros
        self._initialize_fuzzy_system()
    
    def compute_control(self, angle, angular_velocity):
        """
        Computa a força de controle baseada no ângulo e velocidade angular
        """
        try:
            # Garante que as entradas estejam dentro dos ranges definidos
            angle = float(np.clip(angle, self.angle_range[0], self.angle_range[-1]))
            angular_velocity = float(np.clip(angular_velocity, self.angular_velocity_range[0], self.angular_velocity_range[-1]))
            
            self.simulation.input['angle'] = angle
            self.simulation.input['angular_velocity'] = angular_velocity
            self.simulation.compute()
            
            # Verifica se a saída 'force' existe
            if 'force' in self.simulation.output:
                force = self.simulation.output['force'] * self.gain
                return np.clip(force, -20, 20)
            else:
                print("Aviso: saída 'force' não encontrada no sistema fuzzy.")
                return 0.0
        except Exception as e:
            print(f"Erro no controlador FIS: {str(e)}")
            return 0.0  # Retorna força zero em caso de erro 