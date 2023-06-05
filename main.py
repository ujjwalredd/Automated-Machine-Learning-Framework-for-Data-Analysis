from feature import feature
from data_clean import data_clean
from models import models
from Scores import Scores
from H_optmization import H_optmization
import pandas as pd


#dataset path 
df = pd.read_csv("-----------------------") #Enter your dataset path 

#target column name
target_column_name = "---------------------" #change to specific name

x = df.drop(target_column_name, axis=1)
y = df[target_column_name]

st = int(input("Enter value (yes:1 or no:0) for using the Hyperparamter Optimized models fro better results: "))

def main():
    xx,yy = data_clean(x,y,target_column_name,df)
    featur = feature(xx,yy)
    print("\n-------------------xxxx-------------------------")
    print("\n This features are selected: ", feature(xx,yy))
    print("\n-------------------xxxx-------------------------")
    if st == 0:
        c,r,bc,br = models(xx,yy,featur)
        Scores(c,r,bc,br)
    else:
        cc,rr,bbc,bbr = H_optmization(xx,yy,featur)
        Scores(cc,rr,bbc,bbr)
    
        
    
if __name__ == "__main__":
    main()