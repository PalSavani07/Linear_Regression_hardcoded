def dot_product(a,b):
    """"
    Multiplies two 1D lists element-by-element and returns their sum.
    """

    if len(a) != len(b):
        raise ValueError(
            f"[dot_product] Length mismatch: len(a)={len(a)}, len(b)={len(b)}"
        )
    
    result=0.0
    for i in range(len(a)):
        result+=a[i]*b[i]

    return result

def vec_add(a,b):
    if len(a) != len(b):
        raise ValueError(
                f"[vec_add] Length mismatch: len(a)={len(a)}, len(b)={len(b)}"
        )

    result=[]
    for i in range(len(a)):
        result.append(a[i]+b[i])
    
    return result

def vec_subtract(a, b):
   
    if len(a) != len(b):
        raise ValueError(
            f"[vec_subtract] Length mismatch: len(a)={len(a)}, len(b)={len(b)}"
        )
 
    result = []
    for i in range(len(a)):
        result.append(a[i] - b[i])
 
    return result

def scalar_multiply(v,scalar):

    if not isinstance(v, list):
        raise TypeError(
            f"[scalar_multiply] Expected a list, got {type(v)}"
        )
 
    result = []
    for i in range(len(v)):
        result.append(v[i] * scalar)
 
    return result

def scalar_divide(v,scalar):
    if not isinstance(v, list):
        raise TypeError(
            f"[scalar_divide] Expected a list, got {type(v)}"
        )
 
    result = []
    for i in range(len(v)):
        result.append(v[i] / scalar)
 
    return result


def mat_vec_multiply(M, v):
    if len(M[0]) != len(v):
        raise ValueError(
            f"[mat_vec_multiply] Column count of M ({len(M[0])}) "
            f"must equal length of v ({len(v)})"
        )
 
    result = []
    for row in M:
        result.append(dot_product(row, v))
 
    return result

def transpose(M):
    if len(M) == 0:
        raise ValueError("[transpose] Cannot transpose an empty matrix.")
 
    n_rows = len(M)
    n_cols = len(M[0])
 
    
    result = [[0.0] * n_rows for _ in range(n_cols)]
 
    for i in range(n_rows):
        for j in range(n_cols):
            result[j][i] = M[i][j]
 
    return result