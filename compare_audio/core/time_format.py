"""Clock-style time formatting shared by the analysis text, UI and reports."""


def format_clock(seconds: float, decimals: int = 1) -> str:
    """``83.42`` -> ``"1:23.4"``; hours are added only when needed."""
    sign = "-" if seconds < 0 else ""
    seconds = abs(seconds)
    scale = 10**decimals
    total = round(seconds * scale)
    whole, fraction = divmod(total, scale)
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    tail = f".{fraction:0{decimals}d}" if decimals else ""
    if hours:
        return f"{sign}{hours}:{minutes:02d}:{secs:02d}{tail}"
    return f"{sign}{minutes}:{secs:02d}{tail}"
