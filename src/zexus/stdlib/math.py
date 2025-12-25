"""Math module for Zexus standard library."""

import math
import random
from typing import List


class MathModule:
    """Provides advanced mathematical operations."""

    # Constants
    PI = math.pi
    E = math.e
    TAU = math.tau
    INF = math.inf
    NAN = math.nan

    # Trigonometric functions
    @staticmethod
    def sin(x: float) -> float:
        """Sine of x radians."""
        return math.sin(x)

    @staticmethod
    def cos(x: float) -> float:
        """Cosine of x radians."""
        return math.cos(x)

    @staticmethod
    def tan(x: float) -> float:
        """Tangent of x radians."""
        return math.tan(x)

    @staticmethod
    def asin(x: float) -> float:
        """Arc sine in radians."""
        return math.asin(x)

    @staticmethod
    def acos(x: float) -> float:
        """Arc cosine in radians."""
        return math.acos(x)

    @staticmethod
    def atan(x: float) -> float:
        """Arc tangent in radians."""
        return math.atan(x)

    @staticmethod
    def atan2(y: float, x: float) -> float:
        """Arc tangent of y/x in radians."""
        return math.atan2(y, x)

    # Hyperbolic functions
    @staticmethod
    def sinh(x: float) -> float:
        """Hyperbolic sine."""
        return math.sinh(x)

    @staticmethod
    def cosh(x: float) -> float:
        """Hyperbolic cosine."""
        return math.cosh(x)

    @staticmethod
    def tanh(x: float) -> float:
        """Hyperbolic tangent."""
        return math.tanh(x)

    @staticmethod
    def asinh(x: float) -> float:
        """Inverse hyperbolic sine."""
        return math.asinh(x)

    @staticmethod
    def acosh(x: float) -> float:
        """Inverse hyperbolic cosine."""
        return math.acosh(x)

    @staticmethod
    def atanh(x: float) -> float:
        """Inverse hyperbolic tangent."""
        return math.atanh(x)

    # Power and logarithmic functions
    @staticmethod
    def exp(x: float) -> float:
        """e raised to power x."""
        return math.exp(x)

    @staticmethod
    def log(x: float, base: float = math.e) -> float:
        """Logarithm of x to given base."""
        return math.log(x, base)

    @staticmethod
    def log10(x: float) -> float:
        """Base-10 logarithm."""
        return math.log10(x)

    @staticmethod
    def log2(x: float) -> float:
        """Base-2 logarithm."""
        return math.log2(x)

    @staticmethod
    def log1p(x: float) -> float:
        """Natural logarithm of 1+x."""
        return math.log1p(x)

    @staticmethod
    def pow(x: float, y: float) -> float:
        """x raised to power y."""
        return math.pow(x, y)

    @staticmethod
    def sqrt(x: float) -> float:
        """Square root."""
        return math.sqrt(x)

    @staticmethod
    def cbrt(x: float) -> float:
        """Cube root."""
        return math.copysign(1, x) * abs(x) ** (1/3)

    # Rounding and absolute functions
    @staticmethod
    def ceil(x: float) -> int:
        """Ceiling (smallest integer >= x)."""
        return math.ceil(x)

    @staticmethod
    def floor(x: float) -> int:
        """Floor (largest integer <= x)."""
        return math.floor(x)

    @staticmethod
    def trunc(x: float) -> int:
        """Truncate to integer."""
        return math.trunc(x)

    @staticmethod
    def round(x: float, ndigits: int = 0) -> float:
        """Round to n digits."""
        return round(x, ndigits)

    @staticmethod
    def abs(x: float) -> float:
        """Absolute value."""
        return abs(x)

    # Other functions
    @staticmethod
    def factorial(n: int) -> int:
        """Factorial of n."""
        return math.factorial(n)

    @staticmethod
    def gcd(a: int, b: int) -> int:
        """Greatest common divisor."""
        return math.gcd(a, b)

    @staticmethod
    def lcm(a: int, b: int) -> int:
        """Least common multiple."""
        return abs(a * b) // math.gcd(a, b) if a and b else 0

    @staticmethod
    def degrees(x: float) -> float:
        """Convert radians to degrees."""
        return math.degrees(x)

    @staticmethod
    def radians(x: float) -> float:
        """Convert degrees to radians."""
        return math.radians(x)

    @staticmethod
    def copysign(x: float, y: float) -> float:
        """Return x with sign of y."""
        return math.copysign(x, y)

    @staticmethod
    def fmod(x: float, y: float) -> float:
        """Floating point remainder of x/y."""
        return math.fmod(x, y)

    @staticmethod
    def modf(x: float) -> tuple:
        """Return fractional and integer parts."""
        return math.modf(x)

    @staticmethod
    def isnan(x: float) -> bool:
        """Check if x is NaN."""
        return math.isnan(x)

    @staticmethod
    def isinf(x: float) -> bool:
        """Check if x is infinite."""
        return math.isinf(x)

    @staticmethod
    def isfinite(x: float) -> bool:
        """Check if x is finite."""
        return math.isfinite(x)

    # Statistics functions
    @staticmethod
    def sum(numbers: List[float]) -> float:
        """Sum of numbers."""
        return sum(numbers)

    @staticmethod
    def mean(numbers: List[float]) -> float:
        """Arithmetic mean."""
        return sum(numbers) / len(numbers) if numbers else 0

    @staticmethod
    def median(numbers: List[float]) -> float:
        """Median value."""
        sorted_nums = sorted(numbers)
        n = len(sorted_nums)
        if n == 0:
            return 0
        mid = n // 2
        if n % 2 == 0:
            return (sorted_nums[mid - 1] + sorted_nums[mid]) / 2
        return sorted_nums[mid]

    @staticmethod
    def mode(numbers: List[float]) -> float:
        """Most common value."""
        if not numbers:
            return 0
        from collections import Counter
        return Counter(numbers).most_common(1)[0][0]

    @staticmethod
    def variance(numbers: List[float]) -> float:
        """Population variance."""
        if not numbers:
            return 0
        m = MathModule.mean(numbers)
        return sum((x - m) ** 2 for x in numbers) / len(numbers)

    @staticmethod
    def stdev(numbers: List[float]) -> float:
        """Population standard deviation."""
        return math.sqrt(MathModule.variance(numbers))

    @staticmethod
    def min(numbers: List[float]) -> float:
        """Minimum value."""
        return min(numbers) if numbers else 0

    @staticmethod
    def max(numbers: List[float]) -> float:
        """Maximum value."""
        return max(numbers) if numbers else 0

    @staticmethod
    def clamp(x: float, min_val: float, max_val: float) -> float:
        """Clamp x between min and max."""
        return max(min_val, min(max_val, x))

    @staticmethod
    def lerp(a: float, b: float, t: float) -> float:
        """Linear interpolation between a and b."""
        return a + (b - a) * t

    # Random functions
    @staticmethod
    def random() -> float:
        """Random float in [0, 1)."""
        return random.random()

    @staticmethod
    def randint(a: int, b: int) -> int:
        """Random integer in [a, b]."""
        return random.randint(a, b)

    @staticmethod
    def randrange(start: int, stop: int = None, step: int = 1) -> int:
        """Random integer from range."""
        if stop is None:
            return random.randrange(start)
        return random.randrange(start, stop, step)

    @staticmethod
    def choice(seq: List) -> any:
        """Random element from sequence."""
        return random.choice(seq) if seq else None

    @staticmethod
    def shuffle(seq: List) -> List:
        """Shuffle sequence (returns new list)."""
        result = seq.copy()
        random.shuffle(result)
        return result


# Export all functions and constants
PI = MathModule.PI
E = MathModule.E
TAU = MathModule.TAU
INF = MathModule.INF
NAN = MathModule.NAN
sin = MathModule.sin
cos = MathModule.cos
tan = MathModule.tan
asin = MathModule.asin
acos = MathModule.acos
atan = MathModule.atan
atan2 = MathModule.atan2
sinh = MathModule.sinh
cosh = MathModule.cosh
tanh = MathModule.tanh
asinh = MathModule.asinh
acosh = MathModule.acosh
atanh = MathModule.atanh
exp = MathModule.exp
log = MathModule.log
log10 = MathModule.log10
log2 = MathModule.log2
log1p = MathModule.log1p
pow = MathModule.pow
sqrt = MathModule.sqrt
cbrt = MathModule.cbrt
ceil = MathModule.ceil
floor = MathModule.floor
trunc = MathModule.trunc
round = MathModule.round
abs = MathModule.abs
factorial = MathModule.factorial
gcd = MathModule.gcd
lcm = MathModule.lcm
degrees = MathModule.degrees
radians = MathModule.radians
copysign = MathModule.copysign
fmod = MathModule.fmod
modf = MathModule.modf
isnan = MathModule.isnan
isinf = MathModule.isinf
isfinite = MathModule.isfinite
sum = MathModule.sum
mean = MathModule.mean
median = MathModule.median
mode = MathModule.mode
variance = MathModule.variance
stdev = MathModule.stdev
min = MathModule.min
max = MathModule.max
clamp = MathModule.clamp
lerp = MathModule.lerp
random = MathModule.random
randint = MathModule.randint
randrange = MathModule.randrange
choice = MathModule.choice
shuffle = MathModule.shuffle
