from __future__ import annotations

__all__ = [
    "VectorLike",
    "State",
]


from typing import TYPE_CHECKING, Protocol, Self, runtime_checkable

if TYPE_CHECKING:
    from deltastep.integrators import DiffEq, Integrator


@runtime_checkable
class VectorLike(Protocol):
    def __add__(self, other: Self) -> Self: ...
    def __radd__(self, other: Self) -> Self: ...
    def __sub__(self, other: Self) -> Self: ...
    def __rsub__(self, other: Self) -> Self: ...
    def __mul__(self, other: float) -> Self: ...
    def __rmul__(self, other: float) -> Self: ...
    def __truediv__(self, other: float) -> Self: ...
    def __rtruediv__(self, other: float) -> Self: ...


class State[V: VectorLike]:
    def __init__(self, x: V, dxdt: V, *derivatives: V) -> None:
        self.derivatives = [x, dxdt, *derivatives]
        self.order = len(self.derivatives) - 1

    @property
    def x(self) -> V:
        """Convenient alias for State.derivatives[0]"""
        return self.derivatives[0]

    @x.setter
    def x(self, value: V) -> None:
        self.derivatives[0] = value

    @property
    def dxdt(self) -> V:
        """Alias for State.derivatives[1]"""
        return self.derivatives[1]

    @dxdt.setter
    def dxdt(self, value: V) -> None:
        self.derivatives[1] = value

    def __getitem__(self, index: int) -> V:
        return self.derivatives[index]

    def __setitem__(self, index: int, value: V) -> None:
        if index > self.order:
            raise IndexError("Index out of range")
        self.derivatives[index] = value

    def integrate(
        self,
        integrator: Integrator[V],
        diff_eq: DiffEq[V],
        dt: float,
    ) -> None:
        integrator(self, diff_eq, dt)

    def derivative(self, diff_eq: DiffEq[V]) -> list[V]:
        return [self.derivatives[i] for i in range(1, self.order + 1)] + [
            diff_eq(self)
        ]

    def __eq__(self, value: object) -> bool:
        return (
            isinstance(value, State) and self.derivatives == value.derivatives
        )

    def __repr__(self) -> str:
        return f"State({self.derivatives}, order={self.order})"
