import pandas as pd

def data_clean(x,y,target_column_name,df):
    
    c_x = x.columns.tolist() #columns name

    non_numeric_columns_x = []
    numeric_x = []

    for column in x.columns:
        if not pd.api.types.is_numeric_dtype(x[column]):
            non_numeric_columns_x.append(column)
        else:
            numeric_x.append(column)

    #for x data checking nan for non_numeric
    for i in range(0,len(non_numeric_columns_x)):    
        is_column_all_nan = x[non_numeric_columns_x[i]].isnull().all()
        if is_column_all_nan:
            x[non_numeric_columns_x[i]].fillna(0, inplace=True)
            
    #for x data checking nan for numeric
    for i in range(0,len(numeric_x)):    
        is_all_nan = x[numeric_x[i]].isnull().all()
        if is_all_nan:
            x[numeric_x[i]].fillna(0, inplace=True)
            
            
    #for x data
    for i in range(0,len(non_numeric_columns_x)):  
        x[non_numeric_columns_x[i]].fillna(method='ffill', inplace=True) # Forward fill missing string values
        x[non_numeric_columns_x[i]].fillna(method='bfill', inplace=True) # Backward fill missing string values
    
    
    #for x data
    for i in range(0,len(numeric_x)):
        missing_percentage = x[numeric_x[i]].isnull().sum() / len(x) * 100
        # Handle missing values using different methods
        if missing_percentage < 5: 
            x[numeric_x[i]].fillna(x[numeric_x[i]].mean(), inplace=True) # If missing values are less than 5%, use mean imputation
        elif missing_percentage < 30: 
            x[numeric_x[i]].fillna(x[numeric_x[i]].median(), inplace=True) # If missing values are less than 30%, use median imputation
        else:
            x[numeric_x[i]].interpolate(inplace=True) # If missing values are 30% or more, use interpolation
        
    #for y data
    try:
        missing_percentage = df[target_column_name].isnull().sum() / len(y) * 100
        if missing_percentage < 5:
            df[target_column_name].fillna(df[target_column_name].mean(), inplace=True) # If missing values are less than 5%, use mean imputation
        elif missing_percentage < 30:
            df[target_column_name].fillna(df[target_column_name].median(), inplace=True) # If missing values are less than 30%, use median imputation
        else:
            df[target_column_name].interpolate(inplace=True) # If missing values are 30% or more, use interpolation
    except Exception as e:
        print()
    
    
    #for x data
    for i in range(0,len(non_numeric_columns_x)):
        x[non_numeric_columns_x[i]] = x[non_numeric_columns_x[i]].astype('category').cat.codes

    #for y data
    try:
        y = y.astype('category').cat.codes
    except Exception as e:
        print()
        
        
    return(x,y)