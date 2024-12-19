import pytest

from deltastep.state import State


@pytest.mark.parametrize(
    "state, expected_order, index, expected_value",
    [
        (State(3.5, 7.0), 1, 0, 3.5),
        (State(10.3, 11.4, 63.5), 2, -1, 63.5),
    ],
)
def test_getitem(
    state: State[float], expected_order: int, index: int, expected_value: float
) -> None:
    assert state.order == expected_order
    assert state[index] == expected_value


@pytest.mark.parametrize(
    "state, index, value, expected",
    [
        (State(3.5, 7.0), 1, 0.0, State(3.5, 0.0)),
        (State(10.3, 11.4, 63.5), -1, 12.0, State(10.3, 11.4, 12.0)),
    ],
)
def test_setitem(
    state: State[float], index: int, value: float, expected: State[float]
) -> None:
    state[index] = value
    assert state == expected
