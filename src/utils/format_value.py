def format_value(value) -> str:

    if value >= 1_000_000:
        value /= 1_000_000
        unit = 'M'
    elif value >= 1_000:
        value /=  1_000
        unit = 'k'
    else:
        unit = ''

    if value.is_integer():
        value = int(value)

    return str(value) + unit
