"""Tests for stdlib MathModule."""

import math as pymath
import pytest
from src.zexus.stdlib.math import MathModule


@pytest.fixture
def m():
    return MathModule()


# ── Constants ────────────────────────────────────────────────────────────
class TestConstants:
    def test_pi(self, m):
        assert m.PI == pytest.approx(pymath.pi)

    def test_e(self, m):
        assert m.E == pytest.approx(pymath.e)

    def test_tau(self, m):
        assert m.TAU == pytest.approx(pymath.tau)

    def test_inf(self, m):
        assert m.INF == float("inf")

    def test_nan(self, m):
        assert pymath.isnan(m.NAN)


# ── Trigonometry ─────────────────────────────────────────────────────────
class TestTrig:
    def test_sin(self, m):
        assert m.sin(0) == pytest.approx(0.0)
        assert m.sin(pymath.pi / 2) == pytest.approx(1.0)

    def test_cos(self, m):
        assert m.cos(0) == pytest.approx(1.0)

    def test_tan(self, m):
        assert m.tan(0) == pytest.approx(0.0)

    def test_asin(self, m):
        assert m.asin(1) == pytest.approx(pymath.pi / 2)

    def test_acos(self, m):
        assert m.acos(1) == pytest.approx(0.0)

    def test_atan(self, m):
        assert m.atan(0) == pytest.approx(0.0)

    def test_atan2(self, m):
        assert m.atan2(1, 1) == pytest.approx(pymath.pi / 4)


# ── Hyperbolic ───────────────────────────────────────────────────────────
class TestHyperbolic:
    def test_sinh(self, m):
        assert m.sinh(0) == pytest.approx(0.0)

    def test_cosh(self, m):
        assert m.cosh(0) == pytest.approx(1.0)

    def test_tanh(self, m):
        assert m.tanh(0) == pytest.approx(0.0)

    def test_asinh(self, m):
        assert m.asinh(0) == pytest.approx(0.0)

    def test_acosh(self, m):
        assert m.acosh(1) == pytest.approx(0.0)

    def test_atanh(self, m):
        assert m.atanh(0) == pytest.approx(0.0)


# ── Power / Log ──────────────────────────────────────────────────────────
class TestPowerLog:
    def test_exp(self, m):
        assert m.exp(0) == pytest.approx(1.0)
        assert m.exp(1) == pytest.approx(pymath.e)

    def test_pow(self, m):
        assert m.pow(2, 10) == pytest.approx(1024.0)

    def test_sqrt(self, m):
        assert m.sqrt(144) == pytest.approx(12.0)

    def test_cbrt(self, m):
        assert m.cbrt(27) == pytest.approx(3.0)

    def test_log(self, m):
        assert m.log(pymath.e) == pytest.approx(1.0)

    def test_log10(self, m):
        assert m.log10(100) == pytest.approx(2.0)

    def test_log2(self, m):
        assert m.log2(8) == pytest.approx(3.0)

    def test_log1p(self, m):
        assert m.log1p(0) == pytest.approx(0.0)


# ── Rounding ─────────────────────────────────────────────────────────────
class TestRounding:
    def test_ceil(self, m):
        assert m.ceil(2.3) == 3

    def test_floor(self, m):
        assert m.floor(2.9) == 2

    def test_trunc(self, m):
        assert m.trunc(2.9) == 2

    def test_round(self, m):
        assert m.round(2.567, 2) == pytest.approx(2.57)

    def test_abs(self, m):
        assert m.abs(-42) == 42


# ── Number theory ────────────────────────────────────────────────────────
class TestNumberTheory:
    def test_factorial(self, m):
        assert m.factorial(5) == 120

    def test_gcd(self, m):
        assert m.gcd(12, 8) == 4

    def test_lcm(self, m):
        assert m.lcm(4, 6) == 12


# ── Conversion ───────────────────────────────────────────────────────────
class TestConversion:
    def test_degrees(self, m):
        assert m.degrees(pymath.pi) == pytest.approx(180.0)

    def test_radians(self, m):
        assert m.radians(180) == pytest.approx(pymath.pi)


# ── Checks ───────────────────────────────────────────────────────────────
class TestChecks:
    def test_isnan(self, m):
        assert m.isnan(float("nan")) is True
        assert m.isnan(1.0) is False

    def test_isinf(self, m):
        assert m.isinf(float("inf")) is True
        assert m.isinf(1.0) is False

    def test_isfinite(self, m):
        assert m.isfinite(1.0) is True
        assert m.isfinite(float("inf")) is False


# ── Statistics ───────────────────────────────────────────────────────────
class TestStatistics:
    def test_sum(self, m):
        assert m.sum([1, 2, 3, 4]) == 10

    def test_mean(self, m):
        assert m.mean([2, 4, 6]) == pytest.approx(4.0)

    def test_median(self, m):
        assert m.median([1, 3, 5]) == pytest.approx(3.0)

    def test_mode(self, m):
        assert m.mode([1, 2, 2, 3]) == 2

    def test_variance(self, m):
        result = m.variance([2, 4, 4, 4, 5, 5, 7, 9])
        assert isinstance(result, float)

    def test_stdev(self, m):
        result = m.stdev([2, 4, 4, 4, 5, 5, 7, 9])
        assert isinstance(result, float)

    def test_min(self, m):
        assert m.min([5, 1, 9]) == 1

    def test_max(self, m):
        assert m.max([5, 1, 9]) == 9


# ── Interpolation / Clamping ─────────────────────────────────────────────
class TestInterpolation:
    def test_clamp(self, m):
        assert m.clamp(15, 0, 10) == 10
        assert m.clamp(-5, 0, 10) == 0
        assert m.clamp(5, 0, 10) == 5

    def test_lerp(self, m):
        assert m.lerp(0, 10, 0.5) == pytest.approx(5.0)
        assert m.lerp(0, 10, 0) == pytest.approx(0.0)
        assert m.lerp(0, 10, 1) == pytest.approx(10.0)


# ── Random ───────────────────────────────────────────────────────────────
class TestRandom:
    def test_random_in_range(self, m):
        val = m.random()
        assert 0 <= val < 1

    def test_randint_in_range(self, m):
        val = m.randint(1, 10)
        assert 1 <= val <= 10

    def test_randrange(self, m):
        val = m.randrange(0, 10, 2)
        assert val in range(0, 10, 2)

    def test_choice(self, m):
        val = m.choice([10, 20, 30])
        assert val in [10, 20, 30]

    def test_shuffle(self, m):
        result = m.shuffle([1, 2, 3, 4, 5])
        assert sorted(result) == [1, 2, 3, 4, 5]


# ── Misc ─────────────────────────────────────────────────────────────────
class TestMisc:
    def test_copysign(self, m):
        assert m.copysign(1, -1) == -1.0

    def test_fmod(self, m):
        assert m.fmod(10, 3) == pytest.approx(1.0)

    def test_modf(self, m):
        result = m.modf(3.75)
        assert result[0] == pytest.approx(0.75)
        assert result[1] == pytest.approx(3.0)
