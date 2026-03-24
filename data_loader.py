def load_data(filepath):
    X=[]
    Y=[]

    skipped=0

    with open(filepath,'r') as f:
        lines=f.readlines()

    data_lines=lines[1:]

    for line in data_lines:
        
        line=line.strip()

        if not line:
            continue

        parts=line.split(',')

        if len(parts)!=8:
            print(f"[WARNING] Skipping malformed row (expected 8 cols, got {len(parts)})")
        
            skipped +=1
            continue
        try:
            avg_session_length   = float(parts[3])
            time_on_app          = float(parts[4])
            time_on_website      = float(parts[5])
            length_of_membership = float(parts[6])
            yearly_amount_spent  = float(parts[7])
 
            x_row = [avg_session_length, time_on_app, time_on_website, length_of_membership]
 
            X.append(x_row)
            Y.append(yearly_amount_spent)
 
        except ValueError:
            #Skipping row with non-numeric value
            skipped += 1
            continue
 
    print(f"\n[INFO] Data loaded successfully!")
    print(f"       Total samples loaded : {len(X)}")
    print(f"       Rows skipped          : {skipped}")
 
    return X, Y


