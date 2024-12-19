import numpy as np
from matplotlib import pyplot as plt

from deltastep import (
    RK4,
    Integrator,
    State,
    explicit_euler,
    semi_implicit_euler,
    semi_implicit_euler2,
    verlet,
)


class TestState(State[float]):
    def __init__(self, x: float, dxdt: float) -> None:
        super().__init__(x, dxdt)
        self.positions: list[float] = []

    def update(
        self,
        dt: float,
        integrator: Integrator[float],
    ) -> None:
        integrator(self, acceleration, dt)
        self.positions.append(self.x)


def acceleration(state: State[float]) -> float:
    return -15 * state.x


t = 0.0
dt = 1 / 4
times: list[float] = []

x_0 = 1000.0
v_0 = 0.0
rk4_test = TestState(x_0, v_0)
explicit_euler_test = TestState(x_0, v_0)
semi_implicit_euler_test = TestState(x_0, v_0)
semi_implicit_euler2_test = TestState(x_0, v_0)
verlet_test = TestState(x_0, v_0)

while t <= 100:
    rk4_test.update(dt, RK4)
    explicit_euler_test.update(dt, explicit_euler)
    semi_implicit_euler_test.update(dt, semi_implicit_euler)
    semi_implicit_euler2_test.update(dt, semi_implicit_euler2)
    verlet_test.update(dt, verlet)

    times.append(t)
    t += dt


x = np.arange(0, 100, 0.001)
y = 1000 * np.cos(x * np.sqrt(15))  # * np.exp(-0.05 * x)
plt.plot(x, y, color="black", label="Exact")

plt.plot(times, rk4_test.positions, color="red", label="RK4")
# plt.plot(
#     times, explicit_euler_test.positions, color="blue", label="Explicit Euler"
# )
plt.plot(
    times,
    semi_implicit_euler_test.positions,
    color="green",
    label="Semi-implicit Euler",
)
plt.plot(
    times,
    semi_implicit_euler2_test.positions,
    color="yellow",
    label="Euler variant",
)
plt.plot(times, verlet_test.positions, color="purple", label="Verlet")
plt.legend(loc="upper right")
plt.show()
