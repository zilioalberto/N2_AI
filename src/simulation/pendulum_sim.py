import numpy as np

class PendulumSimulation:
    def __init__(self, mass=1.0, length=1.0, cart_mass=1.0, gravity=9.81, dt=0.01):
        self.mass = mass  # massa do pêndulo (m)
        self.length = length  # comprimento do pêndulo (l)
        self.cart_mass = cart_mass  # massa do carrinho (M)
        self.gravity = gravity  # gravidade (g)
        self.dt = dt
        
        # Estado inicial
        self.angle = 0.1  # ângulo em radianos (0 = para cima)
        self.angular_velocity = 0.0  # velocidade angular
        self.cart_position = 0.0  # posição do carrinho
        self.cart_velocity = 0.0  # velocidade do carrinho
        
    def update(self, force):
        """
        Atualiza o estado do pêndulo usando as equações de movimento
        """
        try:
            # Limita a força aplicada
            force = np.clip(force, -20, 20)
            
            # Aceleração do carrinho
            cart_acceleration = force / (self.cart_mass + self.mass)
            
            # Atualiza a posição e velocidade do carrinho
            self.cart_velocity += cart_acceleration * self.dt
            self.cart_velocity = np.clip(self.cart_velocity, -10, 10)  # Limita a velocidade do carrinho
            self.cart_position += self.cart_velocity * self.dt
            self.cart_position = np.clip(self.cart_position, -10, 10)  # Limita a posição do carrinho
            
            # Aceleração angular do pêndulo
            angular_acceleration = (self.gravity * np.sin(self.angle) - 
                                  cart_acceleration * np.cos(self.angle)) / self.length
            
            # Atualiza a velocidade angular e o ângulo
            self.angular_velocity += angular_acceleration * self.dt
            self.angular_velocity = np.clip(self.angular_velocity, -10, 10)  # Limita a velocidade angular
            self.angle += self.angular_velocity * self.dt
            
            return {
                'cart_position': self.cart_position,
                'cart_velocity': self.cart_velocity,
                'angle': self.angle,
                'angular_velocity': self.angular_velocity
            }
            
        except Exception as e:
            print(f"Erro na simulação do pêndulo: {str(e)}")
            return None
    
    def reset(self):
        """
        Reseta o estado do pêndulo para as condições iniciais
        """
        self.angle = 0.1  # ligeiramente fora do equilíbrio
        self.angular_velocity = 0.0
        self.cart_position = 0.0
        self.cart_velocity = 0.0 