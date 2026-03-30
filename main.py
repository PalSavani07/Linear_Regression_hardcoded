from data_loader  import load_data
from normalizer   import get_min_max, normalize, normalize_1d, denormalize
from model        import train, predict
from evalute     import evaluate

print("\n" + "=" * 55)
print("  STEP 1 — Loading Data")
print("=" * 55)
 
X, Y = load_data("Ecommerce Customers")

print("\n" + "=" * 55)
print("  STEP 2 — Normalizing Data")
print("=" * 55)

x_mins, x_maxs = get_min_max(X)

y_min = min(Y)
y_max = max(Y)

X_scaled = normalize(X, x_mins, x_maxs)
Y_scaled  = normalize_1d(Y, y_min, y_max)
 
print(f"  X normalized  →  sample row : {[round(v, 4) for v in X_scaled[0]]}")
print(f"  Y normalized  →  sample val : {round(Y_scaled[0], 4)}")
print(f"  Y range       →  min=${y_min:.2f}  max=${y_max:.2f}")

print("\n" + "=" * 55)
print("  STEP 3 — Training Model")
print("=" * 55)
 
# Hyperparameters — tune these if needed
LEARNING_RATE = 0.1
EPOCHS        = 1000
 
w, b = train(X_scaled, Y_scaled, lr=LEARNING_RATE, epochs=EPOCHS)

print("\n" + "=" * 55)
print("  STEP 4 — Evaluating Model")
print("=" * 55)
 
mae, r2 = evaluate(X_scaled, Y_scaled, w, b, y_min, y_max)

print("\n" + "=" * 55)
print("  STEP 5 — Predict from New Input")
print("=" * 55)
def predict_new(raw_input, w, b, x_mins, x_maxs, y_min, y_max):
    if len(raw_input) != 4:
        raise ValueError(
            f"[predict_new] Expected 4 features, got {len(raw_input)}.\n"
            f"  Order: [Avg Session Length, Time on App, "
            f"Time on Website, Length of Membership]"
        )
    scaled_input = normalize([raw_input], x_mins, x_maxs)[0]
    y_scaled = predict(scaled_input, w, b)
    y_real = denormalize(y_scaled, y_min, y_max)
 
    return y_real

while True:
 
    print("-" * 55)
    user_input = input("  Enter 4 values separated by commas\n  > ").strip()
 
    if user_input.lower() == 'quit':
        print("\n  Exiting. Goodbye!\n")
        break
 
    try:
        # Parse the input string into a list of floats
        parts = [float(v.strip()) for v in user_input.split(',')]
 
        # Run prediction
        prediction = predict_new(parts, w, b, x_mins, x_maxs, y_min, y_max)
 
        print(f"\n  ✅ Predicted Yearly Amount Spent : ${prediction:.2f}")
        if r2 >= 0.95:
                print(f"     Model confidence : High  (R²={r2:.4f})")
        elif r2 >= 0.85:
                print(f"     Model confidence : Good  (R²={r2:.4f})")
        else:
                print(f"     Model confidence : Low   (R²={r2:.4f}) ⚠️")
    
    except ValueError as e:
        print(f"\n  ❌ Invalid input: {e}")
        print("     Please enter exactly 4 numeric values separated by commas.")