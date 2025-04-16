def tuple_type(strings):
    strings = strings.replace("(", "").replace(")", "").replace(" ", "")
    mapped_int = map(int, strings.split(","))
    return tuple(mapped_int)
