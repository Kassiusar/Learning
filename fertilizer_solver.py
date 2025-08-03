"""Compute fertilizer doses to reach target nutrient concentrations.

This script solves a non-negative least squares problem to determine the
amount of each fertilizer product needed. It replicates the functionality
of a small nutrient mixing calculator without requiring third‑party
packages such as NumPy.
"""

data = {
    "meta": {
        "bowlSize_L": 10_000,
        "concentration_percent": 95,
        "recircPercent": 48,
    },
    "targets": {
        "NO3": 14.56, "P": 1.15, "Cl": 4.49, "SO4": 4.90,
        "K": 9.23, "Ca": 7.48, "Mg": 2.90,
        "Fe": 49.0, "Mn": 13, "Zn": 5, "B": 85,
        "Cu": 1.7, "Mo": 1.05,
    },
    "drain": {
        "NO3": 9.81, "P": 0.24, "Cl": 3.33, "SO4": 3.71,
        "K": 4.71, "Ca": 4.81, "Mg": 2.05,
        "Fe": 12.14, "Mn": 1.57, "Zn": 11.38, "B": 65.24,
        "Cu": 1.33, "Mo": 0.48,
    },
    "products": [
        ("A", "Kalksalpeter 8.7%N"),
        ("A", "Kaliumchloride"),
        ("A", "Iron DTPA 6%"),
        ("B", "Nitric acid 38%"),
        ("B", "Kaliloog 50%"),
        ("B", "MKP"),
        ("B", "Potassium sulfate"),
        ("B", "Bitterzout"),
        ("B", "Mn-EDTA 32%"),
        ("B", "Zn-EDTA 35%"),
        ("B", "Borax 11.3%"),
        ("B", "Cu-EDTA 25%"),
        ("B", "Mo-Na 39.5%"),
    ],
}

fert = {
    "Kalksalpeter 8.7%N": {"NO3": 8.7, "Ca": 5.9},
    "Kaliumchloride": {"K": 7.5, "Cl": 7.5},
    "Iron DTPA 6%": {"Fe": 60},
    "Nitric acid 38%": {"NO3": 8.9},
    "Kaliloog 50%": {"K": 13.2},
    "MKP": {"P": 8.8, "K": 4.4},
    "Potassium sulfate": {"SO4": 9.3, "K": 4.1},
    "Bitterzout": {"SO4": 4.1, "Mg": 4.1},
    "Mn-EDTA 32%": {"Mn": 320},
    "Zn-EDTA 35%": {"Zn": 350},
    "Borax 11.3%": {"B": 113},
    "Cu-EDTA 25%": {"Cu": 250},
    "Mo-Na 39.5%": {"Mo": 395},
}

nutrients = list(data["targets"].keys())

# Matrix A (nutrients x products)
A = []
for ion in nutrients:
    row = []
    for _, name in data["products"]:
        row.append(fert.get(name, {}).get(ion, 0.0))
    A.append(row)

A_T = [list(col) for col in zip(*A)]


def matmul(X, Y):
    m, n, p = len(X), len(X[0]), len(Y[0])
    res = [[0.0] * p for _ in range(m)]
    for i in range(m):
        for k in range(n):
            for j in range(p):
                res[i][j] += X[i][k] * Y[k][j]
    return res


def matvec(X, v):
    m, n = len(X), len(X[0])
    res = [0.0] * m
    for i in range(m):
        s = 0.0
        for k in range(n):
            s += X[i][k] * v[k]
        res[i] = s
    return res


def invert(M):
    n = len(M)
    aug = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(M)]
    for i in range(n):
        pivot = aug[i][i]
        if abs(pivot) < 1e-12:
            for j in range(i + 1, n):
                if abs(aug[j][i]) > abs(pivot):
                    aug[i], aug[j] = aug[j], aug[i]
                    pivot = aug[i][i]
                    break
            if abs(pivot) < 1e-12:
                raise ValueError("Singular matrix")
        factor = aug[i][i]
        for k in range(2 * n):
            aug[i][k] /= factor
        for j in range(n):
            if j == i:
                continue
            factor = aug[j][i]
            if factor:
                for k in range(2 * n):
                    aug[j][k] -= factor * aug[i][k]
    return [row[n:] for row in aug]

# Solve normal equations
ATA = matmul(A_T, A)
f = data["meta"]["recircPercent"] / 100.0
drn = [data["drain"].get(n, 0.0) for n in nutrients]
need = [data["targets"][n] - f * drn[i] for i, n in enumerate(nutrients)]
AT_need = matvec(A_T, need)
ATA_inv = invert(ATA)
x0 = matvec(ATA_inv, AT_need)
x0 = [max(0.0, xi) for xi in x0]

# Greedy non-negative adjustment
for _ in range(10):
    residual = [need[i] - sum(A[i][j] * x0[j] for j in range(len(x0))) for i in range(len(need))]
    if max(abs(r) for r in residual) < 1e-3:
        break
    AT_res = matvec(A_T, residual)
    add = matvec(ATA_inv, AT_res)
    add = [max(0.0, a) for a in add]
    x0 = [x0[i] + add[i] for i in range(len(x0))]

doses = x0

# Convert to kg/L (approx)
conv = 0.1

dose_table = []
for (tank, name), mmol in zip(data["products"], doses):
    unit = "kg" if name in ("Kaliumchloride", "MKP", "Potassium sulfate", "Bitterzout", "Borax 11.3%") else "L"
    dose_table.append((tank, name, round(mmol / conv, 3), unit))

obtained = [f * drn[i] + sum(A[i][j] * doses[j] for j in range(len(doses))) for i in range(len(nutrients))]

print("\n=== DOSES POR TANQUE (stock-tank) ===")
for tank in ("A", "B"):
    for t, n, q, u in dose_table:
        if t == tank and q > 0:
            print(f"{tank}  {n:<30} {q:>10} {u}")

print("\n=== NUTRIENTES – alvo × obtido (mmol L-1 / µmol L-1) ===")
for ion, t, o in zip(nutrients, [data['targets'][n] for n in nutrients], obtained):
    print(f"{ion:<4}  target {t:>6.2f}   →   {o:>6.2f}   Δ={o-t:+.2f}")
