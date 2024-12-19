from __future__ import annotations

__all__ = [
    "RK4",
    "explicit_euler",
    "semi_implicit_euler",
    "semi_implicit_euler2",
    "verlet",
    "Integrator",
    "DiffEq",
]

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from deltastep.state import State, VectorLike


type DiffEq[V: VectorLike] = Callable[[State[V]], V]


class Integrator[V: VectorLike](Protocol):
    def __call__(
        self, state: State[V], diff_eq: DiffEq[V], dt: float
    ) -> None: ...


def explicit_euler[V: VectorLike](
    state: State[V], diff_eq: DiffEq[V], dt: float
) -> None:
    # https://en.wikipedia.org/wiki/Euler_method#Higher-order_process
    # derivative at inital state
    k = state.derivative(diff_eq)

    state[-1] += diff_eq(state) * dt
    for i in range(state.order - 1, -1, -1):
        state[i] += k[i] * dt


def semi_implicit_euler[V: VectorLike](
    state: State[V], diff_eq: DiffEq[V], dt: float
) -> None:
    # https://en.wikipedia.org/wiki/Semi-implicit_Euler_method#The_method
    state[-1] += diff_eq(state) * dt
    for i in range(state.order - 1, -1, -1):
        state[i] += state[i + 1] * dt


def semi_implicit_euler2[V: VectorLike](
    state: State[V], diff_eq: DiffEq[V], dt: float
) -> None:
    # https://en.wikipedia.org/wiki/Semi-implicit_Euler_method#The_method
    for i in range(state.order):
        state[i] += state[i + 1] * dt
    state[-1] += diff_eq(state) * dt


def verlet[V: VectorLike](
    state: State[V], diff_eq: DiffEq[V], dt: float
) -> None:
    # https://en.wikipedia.org/wiki/Verlet_integration#Velocity_Verlet
    old_accel = diff_eq(state)
    state[0] += state[1] * dt + 0.5 * old_accel * dt * dt
    state[1] += 0.5 * (old_accel + diff_eq(state)) * dt


def RK4[V: VectorLike](state: State[V], diff_eq: DiffEq[V], dt: float) -> None:
    # https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods#The_Runge%E2%80%93Kutta_method
    state_len = state.order + 1

    # derivative at inital state
    k1 = state.derivative(diff_eq)
    # state at half step along initial tangent
    s1 = State(*[state[i] + k1[i] * dt / 2 for i in range(state_len)])
    # derivative at half step
    k2 = s1.derivative(diff_eq)
    # state at half step along k2 tangent
    s2 = State(*[state[i] + k2[i] * dt / 2 for i in range(state_len)])
    # derivative at new half step
    k3 = s2.derivative(diff_eq)
    # state at end using k3 tangent
    s3 = State(*[state[i] + k3[i] * dt for i in range(state_len)])
    # slope at end
    k4 = s3.derivative(diff_eq)

    for i in range(state_len):
        state[i] += (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]) * dt / 6
