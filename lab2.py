import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter:
    def __init__(self, initial_state, initial_uncertainty, process_variance, measurement_variance):
        self.state = initial_state
        self.uncertainty = initial_uncertainty
        self.process_variance = process_variance
        self.measurement_variance = measurement_variance

    def predict(self, velocity):
        self.state += velocity
        self.uncertainty += self.process_variance

    def update(self, measurement):
        kalman_coef = self.uncertainty / (self.uncertainty + self.measurement_variance)
        self.state = self.state + kalman_coef * (measurement - self.state)
        self.uncertainty = (1 - kalman_coef) * self.uncertainty
        return self.state


# Параметры симуляции
initial_position = 0.0
initial_uncertainty = 500.0
process_variance = 0.1
measurement_variance = 5.0
num_iterations = 35

kf = KalmanFilter(initial_position, initial_uncertainty, process_variance, measurement_variance)

true_positions = []
measured_positions = []
estimated_positions = []

true_position = initial_position
velocity = 1.0

for i in range(num_iterations):
    velocity += 1.5 if i % 2 != 0 and i % 5 != 0 else -1
    true_position += velocity
    true_positions.append(true_position)

    measurement_noise = np.random.normal(0, np.sqrt(measurement_variance))
    measurement = true_position + measurement_noise
    measured_positions.append(measurement)

    kf.predict(velocity)
    estimated_position = kf.update(measurement)
    estimated_positions.append(estimated_position)

time_steps = np.arange(num_iterations)

plt.figure(figsize=(10, 6))
plt.plot(time_steps, true_positions, 'g-', label='Истинное положение', linewidth=2)
plt.plot(time_steps, measured_positions, 'r.', label='Измерения с шумом', markersize=6)
plt.plot(time_steps, estimated_positions, 'b--', label='Оценка Калмана', linewidth=2)

plt.title('Фильтр Калмана: Оценка положения')
plt.xlabel('Шаги времени')
plt.ylabel('Положение')
plt.legend()
plt.grid(True)
plt.show()