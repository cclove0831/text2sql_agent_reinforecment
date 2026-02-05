def fmt_value(x):
    if x is None:
        return ""
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        if abs(x - round(x)) < 1e-9:
            return str(int(round(x)))
        return f"{x:.2f}"
    return str(x)

def rows_to_answer(rows):
    if len(rows) == 0:
        return ""
    if len(rows) == 1 and len(rows[0]) == 1:
        return fmt_value(rows[0][0])
    if all(len(r) == 1 for r in rows):
        return " | ".join(fmt_value(r[0]) for r in rows)
    if len(rows) == 1:
        return ", ".join(fmt_value(x) for x in rows[0])
    return str(rows)
