def format_number(value: float) -> str:
    """Integers as integers (``3``); any other value at full precision (Python ``repr``)."""
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return repr(number)
