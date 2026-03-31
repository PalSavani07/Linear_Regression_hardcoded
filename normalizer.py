def get_min_max(data):
    if len(data)==0:
        raise ValueError("[get_min_max] Data is empty.")
    
    n_cols=len(data[0])

    mins=[data[0][j] for j in range(n_cols)]
    maxs=[data[0][j] for j in range(n_cols)]

    for i in range(1,len(data)):
        for j in range(n_cols):
            if data[i][j]<mins[j]:
                 mins[j] = data[i][j]
            if data[i][j] > maxs[j]:
                maxs[j] = data[i][j]
    
    return mins,maxs


def normalize(data,mins,maxs):
    if len(mins) != len(maxs):
        raise ValueError("[normalize] mins and maxs must have the same length.")
 
    if len(data[0]) != len(mins):
        raise ValueError(
            f"[normalize] data has {len(data[0])} columns "
            f"but mins/maxs have {len(mins)} values."
        )
 
    scaled = []
 
    for i in range(len(data)):
        scaled_row = []
        for j in range(len(data[i])):
            range_val = maxs[j] - mins[j]
 
            if range_val == 0:
                
                scaled_row.append(0.0)
            else:
                scaled_row.append((data[i][j] - mins[j]) / range_val)
 
        scaled.append(scaled_row)
 
    return scaled


def normalize_1d(data,mins,maxs):


    range_val=maxs-mins

    if range_val==0:
        raise ValueError("[normalize_1d] max_val == min_val, cannot normalize.")
    
    scaled=[]
    for val in data:
        scaled.append((val-mins)/range_val)

    return scaled

def denormalize(scaled_val,mins,maxs):

    return (scaled_val*(maxs-mins))+mins


