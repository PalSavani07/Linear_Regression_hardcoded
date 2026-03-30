from model import predict
from normalizer import denormalize

def mean_absolute_error(Y_actual,Y_predicted):
    if len(Y_actual) != len(Y_predicted):
        raise ValueError(
            f"[mean_absolute_error] Length mismatch: "
            f"Y_actual={len(Y_actual)}, Y_predicted={len(Y_predicted)}"
        )
    
    n = len(Y_actual)
    total = 0.0
 
    for i in range(n):
        total += abs(Y_predicted[i] - Y_actual[i])
 
    return total / n

def r2_score(Y_actual,Y_predicted):

    if len(Y_actual) != len(Y_predicted):
        raise ValueError(
            f"[r2_score] Length mismatch: "
            f"Y_actual={len(Y_actual)}, Y_predicted={len(Y_predicted)}"
        )
    
    n = len(Y_actual)
    y_mean=sum(Y_actual)/n

    ss_res=0.0
    ss_tot=0.0

    for i in range(n):
        ss_res+=(Y_actual[i]-Y_predicted[i])**2
        ss_tot+=(Y_actual[i]-y_mean)**2
    
    if ss_tot == 0:
        raise ValueError(
            "[r2_score] All Y_actual values are identical — R² is undefined."
        )
    
    return 1.0 - (ss_res / ss_tot)

def evaluate(X, Y_actual_scaled, w, b, y_min, y_max):
    n = len(X)

    Y_predicted_scaled = []
    for i in range(n):
        y_hat = predict(X[i], w, b)
        Y_predicted_scaled.append(y_hat)

    Y_actual    = [denormalize(y, y_min, y_max) for y in Y_actual_scaled]
    Y_predicted = [denormalize(y, y_min, y_max) for y in Y_predicted_scaled]

    mae = mean_absolute_error(Y_actual, Y_predicted)
    r2  = r2_score(Y_actual, Y_predicted)
    print("\n" + "=" * 55)
    print("  Model Evaluation Report")
    print("=" * 55)
    print(f"  Samples Evaluated     : {n}")
    print(f"  Mean Absolute Error   : ${mae:.2f}  ← avg dollar error")
    print(f"  R² Score              : {r2:.4f}   ← 1.0 = perfect")
    print("=" * 55)

    if r2 >= 0.95:
        verdict = "Excellent ✅"
    elif r2 >= 0.85:
        verdict = "Good ✅"
    elif r2 >= 0.70:
        verdict = "Acceptable ⚠️ — try more epochs or tune lr"
    else:
        verdict = "Poor ❌ — check data, lr, or epochs"
 
    print(f"  Verdict               : {verdict}")
    print("=" * 55)

    print("\n  Sample Predictions (first 5):")
    print(f"  {'#':<5} {'Actual ($)':<15} {'Predicted ($)':<15} {'Error ($)'}")
    print("  " + "-" * 48)
    for i in range(min(5, n)):
        err = abs(Y_actual[i] - Y_predicted[i])
        print(f"  {i+1:<5} {Y_actual[i]:<15.2f} {Y_predicted[i]:<15.2f} {err:.2f}")
 
    return mae, r2