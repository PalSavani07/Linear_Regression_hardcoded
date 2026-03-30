from math_utils import dot_product, vec_add, vec_subtract,scalar_multiply,scalar_divide


def predict(x_row,w,b):
    return dot_product(w,x_row)+b

def compute_loss(X,Y,w,b):

    n = len(X)
 
    if n == 0:
        raise ValueError("[compute_loss] X is empty.")
 
    total_error = 0.0

    for i in range(n):
        y_hat = predict(X[i], w, b)
        error = y_hat - Y[i]
        total_error += error ** 2
 
    return total_error / n

def compute_gradients(X,Y,w,b):

    n=len(X)
    n_features=len(w)

    dw=[0.0]*n_features
    db=0.0

    for i in range(n):
        y_hat=predict(X[i],w,b)
        error=y_hat-Y[i]

        for j in range(n_features):
            dw[j]+=error * X[i][j]
        
        db+=error
    
    scale=2.0/n
    dw=scalar_multiply(dw,scale)
    db=db*scale

    return dw,db

def train(X,Y,lr,epochs):
    
    n_features = len(X[0])
    w = [0.0] * n_features
    b = 0.0
 
    print("=" * 55)
    print("  Training Started")
    print(f"  Samples   : {len(X)}")
    print(f"  Features  : {n_features}")
    print(f"  Learning Rate        : {lr}")
    print(f"  Epochs    : {epochs}")
    print("=" * 55)

    for epoch in range(epochs):
        dw,db=compute_gradients(X,Y,w,b)
        w=vec_subtract(w,scalar_multiply(dw,lr))

        b=b-lr*db

        if epoch%100==0:
            loss = compute_loss(X, Y, w, b)
            print(f"  Epoch {epoch:>5}  |  Loss (MSE): {loss:.6f}")

    final_loss = compute_loss(X, Y, w, b)
    print("=" * 55)
    print(f"  Training Complete!")
    print(f"  Final Loss (MSE) : {final_loss:.6f}")
    print(f"  Final Weights    : {[round(wi, 4) for wi in w]}")
    print(f"  Final Bias       : {round(b, 4)}")
    print("=" * 55)
 
    return w, b