def similarity_with_levenshtein_distance(x, y):
    rows, columns = len(x) + 1, len(y) + 1
    arr = [list(range(columns)) if not i else [i] + [0] * (columns - 1) for i in range(rows)]

    for i in range(1, rows):
        for j in range(1, columns):
            if x[i - 1] == y[j - 1]:
                arr[i][j] = arr[i - 1][j - 1]
            else:
                arr[i][j] = min(arr[i - 1][j], arr[i][j - 1], arr[i - 1][j - 1]) + 1

    return round(1 - arr[rows - 1][columns - 1] / max(len(x), len(y)), 2)
