import numpy as np

class PendulumSimulation:
    def __init__(self, mass=1.0, length=1.0, cart_mass=1.0, gravity=9.81, dt=0.01, inertia=None):
        self.mass = mass  # massa do pêndulo (m_p)
        self.length = length  # comprimento do pêndulo (l)
        self.cart_mass = cart_mass  # massa do carrinho (m_c)
        self.gravity = gravity  # gravidade (g)
        self.dt = dt
        self.inertia = inertia if inertia is not None else self.mass * self.length ** 2  # I
        
        # Estado inicial
        self.angle = 0.1  # ângulo em radianos (0 = para cima)
        self.angular_velocity = 0.0  # velocidade angular
        self.cart_position = 0.0  # posição do carrinho
        self.cart_velocity = 0.0  # velocidade do carrinho
        
    def update(self, force):
        """
        Atualiza o estado do pêndulo usando as equações de movimento completas
        """
        try:
            # Limita a força aplicada
            force = np.clip(force, -20, 20)

            m_p = self.mass
            m_c = self.cart_mass
            l = self.length
            g = self.gravity
            I = self.inertia
            theta = self.angle
            theta_dot = self.angular_velocity
            x_dot = self.cart_velocity

            # Sistema de equações:
            # 1) x_ddot = (m_p * l * (theta_dot**2 * sin(theta) - theta_ddot * cos(theta)) + F) / (m_c + m_p)
            # 2) theta_ddot = (m_p * l * (g * sin(theta) - x_ddot * cos(theta))) / (I + m_p * l**2)

            # Resolvendo o sistema:
            # Primeiro, expressa x_ddot em função de theta_ddot
            # x_ddot = (m_p * l * (theta_dot**2 * sin(theta) - theta_ddot * cos(theta)) + force) / (m_c + m_p)
            # Substitui x_ddot na equação de theta_ddot e resolve para theta_ddot

            denom = (I + m_p * l**2) * (m_c + m_p) - (m_p**2) * (l**2) * (np.cos(theta)**2)
            
            # Numerador de theta_ddot
            num_theta_ddot = (m_p * l * (g * np.sin(theta)) * (m_c + m_p)
                              + m_p * l * (np.cos(theta)) * (force)
                              - m_p**2 * l**2 * theta_dot**2 * np.sin(theta) * np.cos(theta))
            theta_ddot = num_theta_ddot / denom

            # Agora x_ddot
            num_x_ddot = (m_p * l * (theta_dot**2 * np.sin(theta) - theta_ddot * np.cos(theta)) + force)
            x_ddot = num_x_ddot / (m_c + m_p)

            # Atualiza os estados
            self.cart_velocity += x_ddot * self.dt
            self.cart_velocity = np.clip(self.cart_velocity, -10, 10)
            self.cart_position += self.cart_velocity * self.dt
            self.cart_position = np.clip(self.cart_position, -10, 10)

            self.angular_velocity += theta_ddot * self.dt
            self.angular_velocity = np.clip(self.angular_velocity, -10, 10)
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