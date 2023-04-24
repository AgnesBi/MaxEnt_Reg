# Last updated: 04/17/2023

import numpy as np


def get_info(
    filename: str,
):
    """Reads in a file of OTSoft format and stores the information.

    OTSoft format files:
        (1) Tab-delimited.
        (2) The first two lines are the constraint names (full names, abbreviated names), starting with three empty tabs.
        (3) A series of tableau
            (i) First column: UR, listed once on the first row and a tab for the remaining rows in the same tableau.
            (ii) Second column: CANDIDATES, one per row.
            (iii) Third column: WINNER/FREQUENCY; 1 or n (n>1 relative frequency) for a winner, and 0 or blank for a loser.
            Note: Possible to have more than one winners to indicate optionality/variation.

    Args:
        filename: name of the input file

    Sample returns:

        constraints: ['Ident (asp)', '*dh', '*[-son/+voice]']

        urs: ['ta', 'tha', 'ada', 'atha', 'at', 'tada']

        candidates: [array(['ta', 'da', 'tha', 'dha'], dtype='<U16'),
                    array(['tha', 'ta', 'da', 'dha'], dtype='<U16'),
                    array(['ada', 'ata', 'atha', 'adha'], dtype='<U16')]

        observed_prob: [np.array([[1/3, 2/3, 0, 0]]),
                np.array([[0, 1, 0, 0]])]

        violations: [array([[0, 0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 0, 1, 0, 0, 1, 0],
                            [1, 0, 1, 0, 0, 0, 0, 1],
                            [1, 1, 1, 1, 0, 1, 1, 1]]),
                    array([[0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 1, 0, 0, 0, 0, 0],
                            [1, 1, 1, 1, 0, 0, 1, 0],
                            [0, 1, 0, 1, 0, 1, 1, 1]])]


    """
    # Reads in a txt file.
    with open(filename, "r") as f:
        df = f.read()

    # Stores each row as a item in a list.
    df = df.splitlines()

    # For each row, splits the string (by tab) into a list where each cell is a list item.
    # df is now a list of lists, which is then turned into a np array of size |row| x |columns|.
    df = np.array([r.split("\t") for r in df])

    # Extracts the constraint names.
    constraints = df[0, 3:]
    constraints_short = df[1, 3:]

    # Extracts the URs and their indices.
    urs_column = df[:, 0]
    urs = [x for x in urs_column if x]
    urs_idx = [i for i, x in enumerate(urs_column) if x]

    # Records the range of indices for each tableau in a list.
    tableaux = [range(urs_idx[i], urs_idx[i + 1]) for i in range(len(urs_idx) - 1)] + [
        range(urs_idx[-1], df.shape[0])
    ]

    # Extracts candidates for each tableau.
    candidates = [df[tableaux[id_range], 1] for id_range, _ in enumerate(tableaux)]

    # Records the observed probabilities for each candidate.
    observed_prob_str = [
        df[tableaux[id_range], 2] for id_range, _ in enumerate(tableaux)
    ]

    observed_prob = []
    for t in observed_prob_str:
        # Replaces empty strings with 0's.
        t[t == ""] = 0
        observed_prob.append(t.astype(float))

    # Extracts violation profiles for each tableau.
    violations_str = []
    for i, _ in enumerate(tableaux):
        violations_str.append(np.stack([df[id, 3:] for id in tableaux[i]], axis=0))

    violations = []
    for arr in violations_str:
        arr[arr == ""] = 0
        arr = arr.astype(int)
        violations.append(arr)

    return constraints, urs, candidates, violations, observed_prob
