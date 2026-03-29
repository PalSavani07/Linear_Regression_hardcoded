# `model.py` — The Heart of Linear Regression

This file is where the actual **learning** happens. It contains everything
related to making predictions, measuring error, computing gradients, and
updating weights through training.

---

## What This File Is Responsible For

```
Given X (features) and Y (targets)  →  find the best w and b
such that  y_hat = w · x + b  is as close to y as possible
```

---

## The 4 Functions Inside This File

---

### 1️⃣ `predict(x_row, w, b)` — Make a Single Prediction

**What it does:** Takes one sample's features, multiplies each by its
corresponding weight, sums them up, and adds the bias.

**Formula:**
```
y_hat = w1*x1 + w2*x2 + w3*x3 + w4*x4 + b
```

**Input:**
```
x_row  →  1D list  [session_len, time_app, time_web, membership]  (scaled)
w      →  1D list  [w1, w2, w3, w4]  (current weights)
b      →  float    (current bias)
```

**Output:**
```
y_hat  →  single float  (scaled prediction)
```

**Uses from math_utils:** `dot_product(w, x_row)`

---

### 2️⃣ `compute_loss(X, Y, w, b)` — Measure How Wrong the Model Is

**What it does:** Runs predictions on all samples and computes the
average squared error — called **Mean Squared Error (MSE)**.

**Formula:**
```
MSE = (1/n) * Σ (y_hat_i - y_i)²
```

**Why squared?**
- Squaring makes all errors positive
- Penalizes large errors more than small ones
- Smooth and differentiable — gradient descent needs this

**Input:**
```
X  →  2D list  [n_samples][4 features]  (scaled)
Y  →  1D list  [n_samples]              (scaled)
w  →  1D list  [w1, w2, w3, w4]
b  →  float
```

**Output:**
```
loss  →  single float  (lower = better model)
```

**What to watch:** Print this every 100 epochs. It must go **down** over
time. If it goes up, your learning rate is too high.

---

### 3️⃣ `compute_gradients(X, Y, w, b)` — Find Which Direction to Move

**What it does:** Computes how much the loss changes when you nudge each
weight or the bias slightly. This is the **derivative of MSE** with respect
to each parameter.

**Formulas:**
```
For each weight wj:
    dL/dw_j = (2/n) * Σ (y_hat_i - y_i) * x_ij

For bias b:
    dL/db   = (2/n) * Σ (y_hat_i - y_i)
```

**Intuition:**
- If `error * x_j` is positive → weight is too high → gradient pushes it down
- If `error * x_j` is negative → weight is too low  → gradient pushes it up
- Bias gradient is just the average error across all samples

**Input:**
```
X  →  2D list  [n_samples][4 features]  (scaled)
Y  →  1D list  [n_samples]              (scaled)
w  →  1D list  current weights
b  →  float    current bias
```

**Output:**
```
dw  →  1D list  [dL/dw1, dL/dw2, dL/dw3, dL/dw4]
db  →  float    dL/db
```

**Uses from math_utils:** `dot_product`, `vec_add`, `scalar_multiply`,
`vec_scalar_divide`

---

### 4️⃣ `train(X, Y, lr, epochs)` — The Training Loop

**What it does:** Repeatedly calls `compute_gradients` and updates the
weights and bias until the model converges.

**The Update Rule (Gradient Descent):**
```
w = w - lr * dw
b = b - lr * db
```

**What it does step by step each epoch:**
```
1. Compute gradients  →  dw, db
2. Update weights     →  w = w - lr * dw
3. Update bias        →  b = b - lr * db
4. Every 100 epochs   →  print loss
```

**Input:**
```
X       →  2D list  scaled features
Y       →  1D list  scaled targets
lr      →  float    learning rate  (e.g. 0.01)
epochs  →  int      number of iterations  (e.g. 1000)
```

**Output:**
```
w  →  1D list  final trained weights  [w1, w2, w3, w4]
b  →  float    final trained bias
```

**Uses from math_utils:** `vec_subtract`, `scalar_multiply`

---

## How the 4 Functions Connect

```
train()
  │
  ├── compute_gradients()
  │       │
  │       └── predict()  ← called for every sample
  │               │
  │               └── dot_product()  [from math_utils]
  │
  └── updates w and b using vec_subtract + scalar_multiply
```

---

## Hyperparameters — What to Set

| Parameter | Recommended Value | What happens if too high | What happens if too low |
|-----------|------------------|--------------------------|-------------------------|
| `lr` (learning rate) | `0.01` | Loss explodes / diverges | Learns very slowly |
| `epochs` | `1000` | Wastes compute (overfits) | Model not fully trained |

---

## What Good Training Looks Like

```
Epoch 0    →  Loss: 0.2451
Epoch 100  →  Loss: 0.0832
Epoch 200  →  Loss: 0.0341
Epoch 300  →  Loss: 0.0187
...
Epoch 900  →  Loss: 0.0031
Epoch 1000 →  Loss: 0.0028  ✅  converged
```

Loss must decrease every epoch. If it:
- **Spikes up** → learning rate too high, reduce `lr`
- **Barely moves** → learning rate too low, increase `lr`
- **Drops then flattens** → increase `epochs`

---

## Imports This File Needs

```python
from math_utils import dot_product, vec_add, vec_subtract,
                       scalar_multiply, vec_scalar_divide
```

---

## What Other Files Use From Here

| File | What it calls |
|------|--------------|
| `main.py` | `train(X, Y, lr, epochs)` → gets back final `w, b` |
| `main.py` | `predict(new_input, w, b)` → for new user input |
| `evaluate.py` | `predict()` → runs on all samples to compute metrics |
