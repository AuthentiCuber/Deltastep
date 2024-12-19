import pytest

from deltastep.integrators import (
    RK4,
    Integrator,
    explicit_euler,
    semi_implicit_euler,
    semi_implicit_euler2,
    verlet,
)
from deltastep.state import State


def acceleration(state: State[float]) -> float:
    return 10


@pytest.mark.parametrize(
    "integrator, dt, iterations, expected",
    [
        (explicit_euler, 0.1, 10, State(4.5, 10.0)),
    ],
)
def test_integrators(
    integrator: Integrator[float],
    iterations: int,
    dt: float,
    expected: State[float],
) -> None:
    initial_state = State(0.0, 0.0)
    state = initial_state
    for _ in range(iterations):
        integrator(state, acceleration, dt)

    assert (
        round(state[0], 7) == expected[0] and round(state[1], 7) == expected[1]
    )
