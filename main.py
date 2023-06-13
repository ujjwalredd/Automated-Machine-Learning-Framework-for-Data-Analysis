from feature import feature
from data_clean import data_clean
from models import models
from Scores import Scores
from H_optmization import H_optmization
import pandas as pd


#dataset path 
def process_data(dataset_path, target_column_name1, use_hyperparameter_optimization):
    df = pd.read_csv(dataset_path) #Enter your dataset path 

    #target column name
    target_column_name = target_column_name1 #change to specific name

    x = df.drop(target_column_name, axis=1)
    y = df[target_column_name]

    st = use_hyperparameter_optimization


    xx,yy = data_clean(x,y,target_column_name,df)
    featur = feature(xx,yy)
    print("\n-------------------xxxx-------------------------")
    print("\n This features are selected: ", feature(xx,yy))
    print("\n-------------------xxxx-------------------------")
    if st == 0:
        c,r,bc,br = models(xx,yy,featur)
        cool = Scores(c,r,bc,br)
    else:
        cc,rr,bbc,bbr = H_optmization(xx,yy,featur)
        cool = Scores(cc,rr,bbc,bbr)

    return cool
