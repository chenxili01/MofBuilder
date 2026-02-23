from pathlib import Path
from typing import List, Sequence, Any


def fetch_pdbfile(
    dir_name: str,
    keywords: Sequence[str],
    nokeywords: Sequence[str],
    ostream: Any,
) -> List[str]:
    """Finds PDB files in a directory matching keyword filters.

    Recursively searches for `.pdb` files under the specified directory whose filenames
    contain all substrings in `keywords` and do not contain any substring from `nokeywords`.

    Args:
        dir_name (str): The directory path to search within.
        keywords (Sequence[str]): List of substrings that must be present in the filename.
        nokeywords (Sequence[str]): List of substrings that must NOT be present in the filename.
        ostream (Any): Output stream object with `print_info` method for logging information.

    Returns:
        List[str]: List containing the name(s) of matching PDB file(s).

    Raises:
        ValueError: If no matching PDB file is found.

    Example:
        >>> fetch_pdbfile('mydir', ['ABC'], ['bad'], ostream)
        ["ABC1.pdb"]

    Note:
        If multiple files match, all are returned. If one match, a single-element list is returned.
    """
    candidates: List[str] = []
    for pdb in Path(dir_name).rglob("*.pdb"):
        name = pdb.name
        if all(i in name for i in keywords) and all(j not in name for j in nokeywords):
            candidates.append(pdb.name)

    if len(candidates) == 0:
        raise ValueError(f"Cannot find a file including '{keywords}'")
    elif len(candidates) == 1:
        ostream.print_info(f"Found the file including {keywords}: {candidates[0]}")
        return candidates
    else:  # len(candidates) > 1
        ostream.print_info(f"Found many files including {keywords}: {candidates}")
        return candidates
