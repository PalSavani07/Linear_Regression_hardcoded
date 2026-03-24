def get_min_max(data):
    if len(data)==0:
        raise ValueError("[get_min_max] Data is empty.")
    
    n_cols=len(data[0])

    mins=[data[0][j] for j in range(n_cols)]
    maxs=[data[0][j] for j in range(n_cols)]

    for i in range(1,len(data)):
        for j in range(n_cols):
            if data[i][j]<min[j]:
                 mins[j] = data[i][j]
            if data[i][j] > maxs[j]:
                maxs[j] = data[i][j]
    
    return mins,maxs

def normalize(data,mins,maxs):
    