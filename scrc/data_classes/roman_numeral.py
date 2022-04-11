from typing import Union

ROMAN_CONSTANTS = (
    ("", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"),
    ("", "X", "XX", "XXX", "XL", "L", "LX", "LXX", "LXXX", "XC"),
    ("", "C", "CC", "CCC", "CD", "D", "DC", "DCC", "DCCC", "CM"),
    ("", "M", "MM", "MMM", "", "", "-", "", "", ""),
)

ROMAN_SYMBOL_MAP = dict(I=1, V=5, X=10, L=50, C=100, D=500, M=1000)

CUTOFF = 4000
BIG_DEC = 2900
BIG_ROMAN = "MMCM"
ROMAN_NOUGHT = "nulla"


class RomanNumeral(int):
    """https://stackoverflow.com/questions/20973546/check-if-an-input-is-a-valid-roman-numeral"""

    def __init__(self, number_or_numeral: Union[int, str]):
        super().__init__()
        if isinstance(number_or_numeral, int):
            self.number = number_or_numeral
        elif isinstance(number_or_numeral, str):
            self.number = self.parse(number_or_numeral)
        else:
            raise ValueError(f"Please either supply a str or an int. {type(number_or_numeral)} is not supported.")

    def __str__(self):
        return RomanNumeral.to_string(self.number)

    @staticmethod
    def digits(num):
        if num < 0:
            raise Exception('range error: negative numbers not supported')
        if num % 1 != 0.0:
            raise Exception('floating point numbers not supported')
        res = []
        while num > 0:
            res.append(num % 10)
            num //= 10
        return res

    @staticmethod
    def to_string(num: int, emptyZero=False):
        if num < CUTOFF:
            digitlist = RomanNumeral.digits(num)
            if digitlist:
                res = reversed([ROMAN_CONSTANTS[order][digit] for order, digit in enumerate(digitlist)])
                return "".join(res)
            else:
                return "" if emptyZero else ROMAN_NOUGHT
        else:
            if num % 1 != 0.0:
                raise Exception('floating point numbers not supported')
            # For numbers over or equal the CUTOFF, the remainder of division by 2900
            # is represented as above, prepended with the multiples of MMCM (2900 in Roman),
            # which guarantees no more than 3 repetitive Ms.
            return BIG_ROMAN * (num // BIG_DEC) + RomanNumeral.to_string(num % BIG_DEC, emptyZero=True)

    @staticmethod
    def parse(numeral: str):
        numeral = numeral.upper()
        result = 0
        if numeral == ROMAN_NOUGHT.upper():
            return result
        lastVal = 0
        lastCount = 0
        subtraction = False
        for symbol in numeral[::-1]:
            value = ROMAN_SYMBOL_MAP.get(symbol)
            if not value:
                raise Exception('incorrect symbol')
            if lastVal == 0:
                lastCount = 1
                lastVal = value
            elif lastVal == value:
                lastCount += 1
                # exceptions
            else:
                result += (-1 if subtraction else 1) * lastVal * lastCount
                subtraction = lastVal > value
                lastCount = 1
                lastVal = value
        return result + (-1 if subtraction else 1) * lastVal * lastCount


if __name__ == '__main__':
    print(RomanNumeral.parse("IV"))
    print(RomanNumeral.to_string(6))

    numeral1 = RomanNumeral(12)
    numeral2 = RomanNumeral(34)
    print(numeral1)
    print(numeral1 > numeral2)
