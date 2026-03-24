# Role of `normalizer.py` — Why & What It Does

---

## The Core Problem It Solves

Look at your dataset's feature ranges:

| Feature | Typical Range | Example Value |
|---|---|---|
| Avg. Session Length | 29 – 38 | 34.49 |
| Time on App | 8 – 15 | 12.65 |
| Time on Website | 33 – 43 | 39.57 |
| Length of Membership | 0 – 7 | 4.08 |
| Yearly Amount Spent (Y) | 250 – 750 | 587.95 |

These features live on **completely different scales**. This causes two big problems:

---

### Problem 1 — Gradient Descent Goes Haywire

Gradient descent updates weights based on how much each feature influences the error.
If one feature has values like `39.57` and another has `4.08`, the gradients will be
**hugely different in size** — the model will overshoot on large-scale features and
barely move on small ones.

Visually, without normalization your loss landscape looks like this:
```
Elongated bowl shape  →  gradients zigzag  →  very slow or never converges
```
With normalization:
```
Round bowl shape  →  gradients go straight to minimum  →  fast convergence ✅
```

### Problem 2 — Weights Become Incomparable

Without scaling, a weight of `0.5` on `Time on App (range 8–15)` means something
totally different from `0.5` on `Length of Membership (range 0–7)`.
You can't compare or interpret them.

---

## What Normalization Actually Does

You'll use **Min-Max Normalization** — it squishes every feature into the range `[0, 1]`:

```
x_scaled = (x - x_min) / (x_max - x_min)
```

Example on `Avg. Session Length`:
```
min = 29.0,  max = 38.0
x   = 34.49

x_scaled = (34.49 - 29.0) / (38.0 - 29.0)
         = 5.49 / 9.0
         = 0.61  ✅  (now between 0 and 1)
```

After normalization, **all 4 features + Y live in [0, 1]** — same scale, fair competition.

---

## The 3 Functions Inside This File

---

### 1️⃣ `get_min_max(data)`

**What it does:** Scans each feature column and finds its minimum and maximum value.

**Input:** X → 2D list `[500 samples][4 features]`

**Output:** Two lists —
```
mins = [min_col0, min_col1, min_col2, min_col3]
maxs = [max_col0, max_col1, max_col2, max_col3]
```

**Why save these?** You MUST reuse the exact same min/max values later when a new
input comes in for prediction. If you recompute min/max on new data, the scaling
will be wrong.

---

### 2️⃣ `normalize(data, mins, maxs)`

**What it does:** Applies the min-max formula to every value in every column.

**Input:** Raw X (2D list) + the mins and maxs from above

**Output:** Scaled X where every value is between 0 and 1

**Also used for Y** — you call this on your target list too, just pass it as a
single-column structure.

---

### 3️⃣ `denormalize(value, min_val, max_val)`

**What it does:** Reverses the scaling — converts a normalized prediction back to
the real world value.

**Formula:**
```
x_original = (x_scaled * (max - min)) + min
```

**Why you need it:** After your model predicts a scaled Y (e.g. `0.74`), you need
to convert it back to actual dollars (e.g. `~$612.00`). Without this, your
prediction is meaningless to the user.

---

## The Flow in Your Project

```
Raw CSV Data
     ↓
get_min_max(X)  →  save mins, maxs
     ↓
normalize(X)    →  X_scaled  (used for training)
normalize(Y)    →  Y_scaled  (used for training)
     ↓
         [ model trains on scaled data ]
     ↓
New user input arrives (raw values)
     ↓
normalize(new_input, same mins, maxs)
     ↓
model predicts  →  y_scaled
     ↓
denormalize(y_scaled, Y_min, Y_max)  →  actual dollar amount  ✅
```

---

## How Other Files Use This

| File | What it calls |
|------|--------------|
| `main.py` | `get_min_max()`, `normalize()`, `denormalize()` |
| `model.py` | receives already-normalized X and Y — doesn't call normalizer directly |
| `evaluate.py` | uses `denormalize()` to convert predictions back before computing metrics |

---

## 🧠 Key Rule to Remember

> The `mins` and `maxs` computed from training data must be **saved and reused**
> for every future prediction. Never recompute them on new data.
