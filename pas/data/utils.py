from typing import TypedDict

# Typed structure of a single solution.
SolutionType = dict[str, tuple[tuple[int, int], ...]]
# Error dictionary
ErrorDict = TypedDict("ErrorDict", {"C1": list[str], "C3": list[str], "C4": list[str], "C5": list[str]})