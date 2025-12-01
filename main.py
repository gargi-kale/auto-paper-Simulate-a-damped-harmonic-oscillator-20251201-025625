# ==== main.py ====
import numpy as np
import matplotlib.pyplot as plt

def simulate(m, k, c, x0, v0, t_max, dt):
    n = int(t_max / dt) + 1
    t = np.linspace(0, t_max, n)
    x = np.empty(n)
    v = np.empty(n)
    x[0] = x0
    v[0] = v0
    for i in range(1, n):
        y = np.array([x[i-1], v[i-1]])
        def f(y):
            return np.array([y[1], -(c/m) * y[1] - (k/m) * y[0]])
        k1 = f(y)
        k2 = f(y + 0.5 * dt * k1)
        k3 = f(y + 0.5 * dt * k2)
        k4 = f(y + dt * k3)
        y_next = y + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        x[i] = y_next[0]
        v[i] = y_next[1]
    return t, x, v

def main():
    m = 1.0
    k = 1.0
    c_crit = 2.0
    dampings = {
        'Underdamped': 0.5,
        'Critically damped': c_crit,
        'Overdamped': 3.0
    }
    t_max = 20.0
    dt = 0.01
    x0 = 1.0
    v0 = 0.0

    results = {}
    for label, c in dampings.items():
        t, x, v = simulate(m, k, c, x0, v0, t_max, dt)
        results[label] = (t, x, v)

    # Plot displacement vs time
    plt.figure()
    for label, (t, x, _) in results.items():
        plt.plot(t, x, label=label)
    plt.xlabel('Time (s)')
    plt.ylabel('Displacement x(t)')
    plt.title('Displacement vs Time for Different Damping Regimes')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('displacement_vs_time.png')
    plt.close()

    # Plot phase space trajectories
    plt.figure()
    for label, (_, x, v) in results.items():
        plt.plot(x, v, label=label)
    plt.xlabel('Displacement x')
    plt.ylabel('Velocity v')
    plt.title('Phase Space Trajectories')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('phase_space.png')
    plt.close()

    # Primary numeric answer (critical damping coefficient)
    print('Answer:', c_crit)

if __name__ == '__main__':
    main()

