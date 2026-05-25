"""Color parser accepts strings, tuples, and named forms."""
import pytest

from server.colors import parse_color


def test_hex_rgb():
    assert parse_color("#ff8800") == (255, 136, 0, 255)


def test_hex_rgba():
    assert parse_color("#ff880080") == (255, 136, 0, 128)


def test_named_color():
    assert parse_color("red") == (255, 0, 0, 255)


def test_transparent_sentinel():
    assert parse_color("transparent") == (0, 0, 0, 0)


def test_tuple_rgb():
    assert parse_color([10, 20, 30]) == (10, 20, 30, 255)


def test_tuple_rgba():
    assert parse_color((10, 20, 30, 40)) == (10, 20, 30, 40)


def test_bad_input_raises():
    with pytest.raises((ValueError, TypeError)):
        parse_color(object())
