# `math_utils.py` — What It Does & Why You Need It

Since you're using **no NumPy**, you need to build your own math engine. This file is your **calculator** — every other file (`model.py`, `normalizer.py`) will import from here.

---

## Why This File Exists

Normal Python lists **can't do math** on each other:
```python
[1, 2, 3] + [4, 5, 6]  →  [1, 2, 3, 4, 5, 6]  ❌ (just joins them!)
[1, 2, 3] * 2           →  [1, 2, 3, 1, 2, 3]  ❌ (just repeats!)
```
So you need to write functions that treat lists **as vectors/matrices** and do proper math.

---

## The 7 Functions You Need to Write

---

### 1️⃣ `dot_product(a, b)` — Vector × Vector

**What it does:** Multiplies two 1D lists element-by-element and sums the result.

**Why you need it:** This is the core of prediction —
`y_hat = w1*x1 + w2*x2 + w3*x3 + w4*x4`
that's literally a dot product of `w` and `x`.

```
a = [w1, w2, w3, w4]
b = [x1, x2, x3, x4]
result = w1*x1 + w2*x2 + w3*x3 + w4*x4  → single number
```

---

### 2️⃣ `vec_add(a, b)` — Vector + Vector

**What it does:** Adds two 1D lists element by element.

**Why you need it:** When accumulating gradients across all samples —
you keep adding each sample's gradient contribution to a running total.

```
[0.1, 0.2, 0.3, 0.4]
+ [0.05, 0.1, 0.15, 0.2]
= [0.15, 0.3, 0.45, 0.6]
```

---

### 3️⃣ `vec_subtract(a, b)` — Vector − Vector

**What it does:** Subtracts two 1D lists element by element.

**Why you need it:** Weight update step —
`new_w = old_w - learning_rate * gradient`
That subtraction on lists needs this function.

```
[0.5, 0.3, 0.7, 0.2]
- [0.01, 0.02, 0.01, 0.03]
= [0.49, 0.28, 0.69, 0.17]
```

---

### 4️⃣ `scalar_multiply(v, scalar)` — Number × Vector

**What it does:** Multiplies every element of a list by a single number.

**Why you need it:** Applying the learning rate to gradients —
`lr * dw` where `lr = 0.01` and `dw` is a list of 4 gradients.

```
scalar = 0.01
v      = [0.5, 1.2, 0.3, 0.8]
result = [0.005, 0.012, 0.003, 0.008]
```

---

### 5️⃣ `vec_scalar_divide(v, scalar)` — Vector ÷ Number

**What it does:** Divides every element by a single number.

**Why you need it:** Averaging gradients over all `n` samples —
`dw = total_gradient / n`

```
v      = [100.0, 200.0, 150.0, 50.0]
scalar = 500   (n = 500 samples)
result = [0.2, 0.4, 0.3, 0.1]
```

---

### 6️⃣ `mat_vec_multiply(M, v)` — Matrix × Vector

**What it does:** Multiplies a 2D list (matrix) by a 1D list (vector).

**Why you need it:** When computing predictions for **all samples at once** instead of looping one by one — makes your training loop cleaner.

```
M = [[x1, x2, x3, x4],   ← sample 1
     [x1, x2, x3, x4],   ← sample 2
     ...]

v = [w1, w2, w3, w4]     ← weights

result = [y_hat_1, y_hat_2, ...]  ← predictions for all samples
```

---

### 7️⃣ `transpose(M)` — Flip a Matrix

**What it does:** Converts rows into columns and columns into rows.

**Why you need it:** During gradient computation —
`dw = Xᵀ · error` — you need X transposed so each **feature's** error contribution is grouped together.

```
M = [[1, 2, 3, 4],    →   Mᵀ = [[1, 5],
     [5, 6, 7, 8]]              [2, 6],
                                 [3, 7],
                                 [4, 8]]
```

---

## How Other Files Will Use This

| File | Functions Used |
|------|---------------|
| `data_loader.py` | *(no math needed)* |
| `normalizer.py` | `scalar_multiply`, `vec_subtract`, `vec_scalar_divide` |
| `model.py` | ALL — `dot_product`, `vec_subtract`, `scalar_multiply`, `vec_add`, `mat_vec_multiply`, `transpose` |
| `evaluate.py` | `dot_product`, `vec_subtract` |
| `main.py` | `dot_product` (for final prediction) |

---

## 🧠 Key Rules While Writing These

- Every function takes **plain Python lists** as input
- Every function returns a **plain Python list** (or a number for `dot_product`)
- Always check lengths match — raise a `ValueError` if they don't
- No imports needed in this file — **pure Python only**
