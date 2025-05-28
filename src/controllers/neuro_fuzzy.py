import torch
import torch.nn as nn
import numpy as np

class NeuroFuzzySystem(nn.Module):
    def __init__(self, num_inputs=2, num_membership=3, num_rules=9):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_membership = num_membership
        self.num_rules = num_rules
        
        # Camada de fuzzificação
        self.membership_centers = nn.Parameter(torch.randn(num_inputs, num_membership))
        self.membership_widths = nn.Parameter(torch.ones(num_inputs, num_membership))
        
        # Camada de regras
        self.rule_weights = nn.Parameter(torch.randn(num_rules, num_membership * num_inputs))
        
        # Camada de consequentes
        self.consequent_weights = nn.Parameter(torch.randn(num_rules))
        
    def gaussian_membership(self, x, centers, widths):
        return torch.exp(-0.5 * ((x.unsqueeze(-1) - centers) / widths) ** 2)
    
    def forward(self, x):
        # Fuzzificação
        membership_values = self.gaussian_membership(x, self.membership_centers, self.membership_widths)
        
        # Achatamento dos valores de pertinência
        flattened_membership = membership_values.reshape(-1)
        
        # Computação das regras
        rule_outputs = torch.matmul(self.rule_weights, flattened_membership)
        rule_outputs = torch.sigmoid(rule_outputs)
        
        # Defuzzificação
        output = torch.matmul(rule_outputs, self.consequent_weights)
        
        return output

class NeuroFuzzyController:
    def __init__(self, learning_rate=0.01, num_rules=9):
        self.learning_rate = learning_rate
        self.num_rules = num_rules
        self.model = NeuroFuzzySystem(num_rules=num_rules)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Inicializa os parâmetros com valores mais estáveis
        with torch.no_grad():
            # Centros dos conjuntos fuzzy
            self.model.membership_centers.data = torch.tensor([
                [-np.pi/2, 0, np.pi/2],  # Ângulo
                [-5, 0, 5]               # Velocidade angular
            ], dtype=torch.float32)
            
            # Larguras dos conjuntos fuzzy
            self.model.membership_widths.data = torch.tensor([
                [np.pi/4, np.pi/8, np.pi/4],  # Ângulo
                [2.5, 1.0, 2.5]              # Velocidade angular
            ], dtype=torch.float32)
            
            # Pesos das regras
            self.model.rule_weights.data = torch.randn(num_rules, 6) * 0.1
            self.model.consequent_weights.data = torch.randn(num_rules) * 0.1
        
    def compute_control(self, angle, angular_velocity):
        """
        Computa a força de controle usando o sistema neuro-fuzzy
        """
        try:
            # Limita os valores de entrada
            angle = np.clip(angle, -np.pi/2, np.pi/2)
            angular_velocity = np.clip(angular_velocity, -10, 10)
            
            inputs = torch.tensor([angle, angular_velocity], dtype=torch.float32)
            with torch.no_grad():
                force = self.model(inputs)
            
            # Limita a força de saída
            return np.clip(force.item(), -20, 20)
            
        except Exception as e:
            print(f"Erro no controlador Neuro-Fuzzy: {str(e)}")
            return 0.0
    
    def train_step(self, angle, angular_velocity, target_force):
        """
        Realiza um passo de treinamento do sistema neuro-fuzzy.
        
        Args:
            angle (float): Ângulo atual do pêndulo
            angular_velocity (float): Velocidade angular atual
            target_force (float): Força alvo desejada
            
        Returns:
            float: Valor da função de perda
        """
        try:
            self.optimizer.zero_grad()
            
            inputs = torch.tensor([angle, angular_velocity], dtype=torch.float32)
            target = torch.tensor([target_force], dtype=torch.float32)
            
            output = self.model(inputs)
            loss = self.criterion(output, target)
            
            loss.backward()
            self.optimizer.step()
            
            return loss.item()
            
        except Exception as e:
            print(f"Erro no treinamento Neuro-Fuzzy: {str(e)}")
            return float('inf') 