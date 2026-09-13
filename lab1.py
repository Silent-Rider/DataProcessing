import numpy as np
import matplotlib.pyplot as plt

def trajectory(t, theta):
    c0, c1, c2, c3 = theta
    return c0 * np.exp(c1 * t + c2) + c3


def normal_density(x, mean, sigma):
    coefficient = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
    exponent = -((x - mean) ** 2) / (2.0 * sigma ** 2)
    return coefficient * np.exp(exponent)


def delta(coordinate, origin):
    return np.sqrt((coordinate - origin) ** 2)


def normalize_weights(weights):
    weights = np.asarray(weights, dtype=float)
    total = np.sum(weights)

    if total == 0:
        return np.ones_like(weights) / len(weights)

    return weights / total


def survival_probability(weight, number_of_draws):
    return 1.0 - (1.0 - weight) ** number_of_draws


def run_particle_filter(
        thetas: np.ndarray,
        n_steps: int,
        sigma: float,
        origin: float,
        survival_threshold: float,
        seed: int
):
    rng = np.random.default_rng(seed)

    t = np.arange(n_steps)

    n_objects = len(thetas)
    n_particles = n_objects - 1

    formula_coordinates = np.zeros((n_objects, n_steps))

    for i, theta in enumerate(thetas):
        formula_coordinates[i] = trajectory(t, theta)

    noise = rng.normal(
        loc=0.0,
        scale=sigma,
        size=(n_objects, n_steps)
    )

    coordinates = 1.0 + formula_coordinates + noise

    measurements = delta(coordinates, origin)

    robot_measurements = measurements[0]

    weights = np.zeros((n_particles, n_steps))

    for j in range(n_particles):
        particle_measurements = measurements[j + 1]

        weights[j] = normal_density(
            particle_measurements,
            mean=robot_measurements,
            sigma=sigma
        )

    normalized_weights = np.zeros_like(weights)

    for step in range(n_steps):
        normalized_weights[:, step] = normalize_weights(
            weights[:, step]
        )

    survival_probabilities = survival_probability(
        normalized_weights,
        number_of_draws=n_particles
    )

    survived = (
        survival_probabilities >= survival_threshold
    ).astype(int)

    return {
        "t": t,
        "formula_coordinates": formula_coordinates,
        "noise": noise,
        "coordinates": coordinates,
        "measurements": measurements,
        "weights": weights,
        "normalized_weights": normalized_weights,
        "survival_probabilities": survival_probabilities,
        "survived": survived,
    }


def plot_coordinates(result, origin=1.0):
    t = result["t"]
    coordinates = result["coordinates"]

    plt.figure(figsize=(12, 6))

    plt.plot(
        t,
        coordinates[0],
        color="#00B050",
        linewidth=2.5,
        label="1 — Робот"
    )

    for i in range(1, coordinates.shape[0]):
        plt.plot(
            t,
            coordinates[i],
            linewidth=1.8,
            label=f"{i + 1} — Частица"
        )

    plt.axhline(
        y=origin,
        color="#FF0000",
        linewidth=2,
        label="Ориентир"
    )

    plt.title("Координаты робота и частиц")
    plt.xlabel("Шаг")
    plt.ylabel("Координата")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def print_results(result, i):
    print("Шаг:", result["t"][i])

    print("\nКоординаты:")
    print(result["coordinates"][:, i])

    print("\nИзмерения — расстояния до ориентира:")
    print(result["measurements"][:, i])

    print("\nВеса частиц 2–5:")
    print(result["weights"][:, i])

    print("\nНормализованные веса частиц 2–5:")
    print(result["normalized_weights"][:, i])

    print("\nВероятности пережить отсев:")
    print(result["survival_probabilities"][:, i])

    print("\nВыжили ли частицы при пороге 0.3:")
    print(result["survived"][:, i])
    print("-------------------------------------\n")


if __name__ == "__main__":

    np.set_printoptions(suppress=True, precision=6)

    THETAS = np.array([
        [0.0313, 0.1137, 0.0070, 0.0029],  # робот
        [0.0022, 0.2000, 0.0806, 0.0058],  # частица 1
        [0.1350, 0.0017, 0.2600, 0.0073],  # частица 2
        [0.0023, 0.0021, 0.0064, 0.0143],  # частица 3
        [0.0021, 0.1352, 0.1225, 0.0034],  # частица 4
    ])

    result = run_particle_filter(
        thetas=THETAS,
        n_steps=30,
        sigma=0.01,
        origin=1,
        survival_threshold=0.3,
        seed=42
    )

    plot_coordinates(result)

    for i in range(3):
        print_results(result, i)
