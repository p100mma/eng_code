import pandas as pd
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import sqlite3
from IPython.display import clear_output
import pickle

def read_database(Database_name='JAN_FEB_MAR_2021_EW1-EW3.db', Timestamp_col_name="Timestamp",
                 Y_var_name="'ew1_czynna_turbiny_wiatrowej_mw'",
                  X_var_names=["'ew1_kierunek_wiatru_na_zewnatrz_gondoli_deg'",
                              "'ew1_predkosc_wiatru_m_s'",
                              "'ew1_stan_turbiny'",
                              "'ew1_temperatura_na_zewnatrz_gondoli_c'"],
                 Y_alias="p",
                 X_aliases=["dir","v","state","temp"]):
    """ Reads data from SQLite database, one table for each variable.
      Each table has a column with timestamps named Timestamp_col_name,
        and a column with numeric values (one name in X_var_names for each table)
      Funcion returns dictionary, where values are tables and keys are supplied aliases or
        original variable names from tables in database.""" 
    dbConnection = sqlite3.connect(Database_name)
    Y_raw=pd.read_sql_query("SELECT * FROM "+Y_var_name, dbConnection)
    if X_aliases is None:
        X_aliases=X_var_names
    if Y_alias is None:
        Y_alias= Y_var_name
    variables_dict={}
    variables_dict[Y_alias]=Y_raw
    for i in range(len(X_var_names)):
        variables_dict[X_aliases[i]]=pd.read_sql_query("SELECT * FROM "+X_var_names[i] , dbConnection)
    return variables_dict
        


def avg_time_diff(timestamp_col):
    """Computes average time difference of pd.TimeSeries data series."""
    right=timestamp_col[1:]                           
    left=timestamp_col[:(timestamp_col.shape[0]-1)] 
    right.index=left.index
    differences= right- left
    delta=differences.mean()
    return delta
def make_regular_timesteps(variables_dict, #output from 
                           Y_name='p',
                           delta=None, min_t= None, max_t= None,
                           eps= pd.Timedelta(pd.offsets.Milli(652)),
                           partial_fill_Y=False,
                          save=True,
                          out_nameX="X_series_turbine.csv",
                          out_nameY="Y_series_turbine.csv"):
    """ Input: data in uneven time intervals,
       Output: data in evenly spaced intervals (data frames for X and Y)
       Args:
           variables_dict - output of read_database, python dictionary,
                            each entry contains a table of some sort with
                            a column of timestamps (pd.timeseries) and numeric values.
           Y_name - key of table containing values and timestamps for Y (value to be predicted),
           delta - desired width of new even time interval, pd.Timedelta,
                if not given- calculate it from the Y variable data and do not change the time intervals between Y's
           min_t, max_t - first and last moment to consider in converting data,pd.timeseries,
                if not given- calculate it from the Y variable data.
           eps - value to append to last boundary of time intervals, detail best left as default.
           partial_fill_Y == True -> do the same thing to Y as to X variables in terms of missing values
           partial_fill_Y == False -> if interval contains any null in Y, put null in that interval
           save == True -> save the result to file on hard drive. """
    no_delta=False
    if min_t is None:
        min_t= variables_dict[Y_name]["Timestamp"].min()
    if max_t is None:
        max_t= variables_dict[Y_name]["Timestamp"].max()
    if delta is None:
        if min_t is not None or max_t is not None:
           print("Warning: min_t and max_t were given but delta not.")
           print("Delta will be calculated from max and min timestamp in Y variable,")
           print("Even though min_t and max_t are specified by the user!")
        delta=avg_time_diff(variables_dict[Y_name]["Timestamp"])
        no_delta=True
    maxtplus=max_t+eps
    left_bounds=[min_t]
    left_bounds[1:]=[min_t +delta*i for i in range(1, np.floor((max_t-min_t)/delta).astype('int64'))]
    right_bounds=left_bounds[1:]
    right_bounds.append(maxtplus)
    result_dict={}
    k=0
    X_vars=list(variables_dict.keys())
    X_vars.remove(Y_name)
    for var_name in X_vars:        #for k-th variable, k from 0 to n_variables...
        vals=[]
        df=variables_dict[var_name]       #take variable's table and put in temporary df
        for j in range(len(left_bounds)): #for every time interval of length delta...
            if j%10_000==0:    #print progress
                clear_output()
                print("In progress of making regular timesteps...")
                print(j)
                print(k)
            chunk=df[((df["Timestamp"]>= left_bounds[j])&(df["Timestamp"]< right_bounds[j]))] #...collect matching data
            if chunk.empty:                                   #depending on found data
                chunk.Value=None                              #if no data mataches, put NULL at j-th row, kth col
            else:
                no_nulls=chunk.Value.isnull().sum()           #count how many nulls there are
                if no_nulls>0:
                    if no_nulls==chunk.shape[0]:
                        chunk.Value=None                      #if all matching data is null, put NULL at j-th row, kth col
                    elif no_nulls< chunk.shape[0]:            #if some data is null, some not...
                        to_draw_from=chunk.Value[chunk.Value.isnull()==False] #...draw with replacement from non-missing data
                        chunk.Value[chunk.Value.isnull()]=to_draw_from.sample(no_nulls, replace=True)# and take their mean
            vals.append(chunk.Value.mean())                   #and put it at j-th row, kth col
        result_dict[k]=vals
        k+=1
    if not no_delta:                                       # if custom delta was given, do the same for Y variable
        y_vals=[]
        df= variables_dict[Y_name]
        for j in range(len(left_bounds)):
            if j%10_000==0:
                clear_output()
                print("In progress of making regular timesteps...")
                print(j)
                print("Y")
            chunk=df[((df["Timestamp"]>= left_bounds[j])&(df["Timestamp"]< right_bounds[j]))]
            if chunk.empty:
                chunk.Value=None
            else:
                no_nulls=chunk.Value.isnull().sum()
                if no_nulls>0 and partial_fill_Y:
                    if no_nulls==chunk.shape[0]:
                        chunk.Value=None
                    elif no_nulls< chunk.shape[0]:
                        to_draw_from=chunk.Value[chunk.Value.isnull()==False]
                        chunk.Value[chunk.Value.isnull()]=to_draw_from.sample(no_nulls, replace=True)
                elif no_nulls>0 and not partial_fill_Y:
                    chunk.Value=None
            y_vals.append(chunk.Value.mean())
        Y_series=pd.DataFrame({"Timestamp": left_bounds,
                              Y_name: y_vals})
    else:
        Y_series=pd.DataFrame({"Timestamp": variables_dict[Y_name]["Timestamp"],
                               Y_name: y_vals})
    X_series_dict=({"Timestamp": left_bounds})
    k=0
    for var_name in  X_vars:
        X_series_dict[var_name]=result_dict[k]
        k+=1
    X_series=pd.DataFrame(X_series_dict)
    if no_delta:                #if no delta was given, there are n-1 rows, so duplicate last row.
        X_series=X_series.append(X_series.loc[X_series.shape[0]-1,], ignore_index=True)
        X_series.iloc[X_series.shape[0]-1,0]=max_t
    if save:
        print("Saving...")
        X_series.to_csv(out_nameX,index=False)
        Y_series.to_csv(out_nameY,index=False)
        print("Success.")
    Y_series.index=[i for i in range(Y_series.shape[0])]
    X_series.index=Y_series.index
    if save:
        print("Saving...")
        X_series.to_csv(out_nameX,index=False)
        Y_series.to_csv(out_nameY,index=False)
        print("Success.")
    return X_series, Y_series
	
	
def read_XY_series(X_filename="X_series_turbine.csv",
                   Y_filename="Y_series_turbine.csv"):
    """Convenience function reading X and Y series from memory. Default settings load files just like they were saved by
        make_regular_timesteps"""
    X_series=pd.read_csv(X_filename)
    X_series.Timestamp=pd.to_datetime(X_series.Timestamp)
    Y_series=pd.read_csv(Y_filename)
    Y_series.Timestamp=pd.to_datetime(Y_series.Timestamp)
    assert X_series.shape[0]==Y_series.shape[0], "Y and X have different number of rows!"
    return X_series, Y_series
	
	
def find_available_boundariesY(Y_series, i_len, o_len, Y_name="p"):
    """Based on Y variable, find regions available as inputs to neural network.
       i_len is the size of desired input to the network (one unit of data X in number of records).
       o_len is the desired length of the output from the network, i.e. how many values of Y to predict.
       Y data is assumed to contain all available Y values (not skipping beginning 
       i_len Y's that cannot be used as reference, since i_len X's are used for predicting future Y)
       Outputs 2 lists of indexes in the form of:
       [start1, start2, ... startn]
       [end1, end2, ... endn], such that i-th training region in Y_series can be accesed as:
       Y_series[starti:endi]
       Y has to have indexes in the form of 0,1... n"""
    cuts_left=[]                                                
    cuts_right=[]                                               
    null_idx=Y_series.loc[pd.isna(Y_series[Y_name]), "Timestamp"].index        #null idx
    if len(null_idx)>0:
        for j in range(len(null_idx)):                                          #take every null idx
            if (null_idx[j]>i_len):   #if there are enough not nulls to make a prediction from the beggining or we are already past that...
                if (j==0):            #if it is first null idx,
                    if ((null_idx[j] - 1) - (i_len -1)) >= o_len: #check if region between i_len_th element and first null is big enough
                        cuts_left.append(i_len)
                        cuts_right.append(null_idx[j]) #right slice idx marks last non null element in Y
                elif ((null_idx[j]-1) - null_idx[j-1]  ) >= o_len:  #if region between current null and previous is big enough to pick a o_len Y sample
                    cuts_left.append(null_idx[j-1]+1)               #(and we already ruled out that its not case with 0 and first null)
                    cuts_right.append(null_idx[j])
        if ((Y_series.shape[0]-1)- null_idx[len(null_idx) -1] ) >= o_len: #if we can fit o_len between last null and last index in Y,
            cuts_left.append(null_idx[len(null_idx) -1] + 1)
            cuts_right.append(Y_series.shape[0])
    else:
      cuts_left=[i_len]
      cuts_right=[Y_series.shape[0]]
    return cuts_left, cuts_right
	
	
	
def Xbounds_fromYbounds(cuts_leftY, cuts_rightY, i_len):
    """ Take output from find_available_boundaries
        and i_len, and based on that calculate
        equivalent of find_available_boundaries but for X.
        It is assumed that i-th observation of X data corresponds to i-th observation of Y data.
        (in terms of time)"""
    clY, crY= np.array(cuts_leftY), np.array(cuts_rightY)
    clX= clY - i_len 
    crX= crY - 1
    return clX.tolist(), crX.tolist()
	
	
	
	
def interpolate_by_parts( parts_left, parts_right, X_series,lim_dir="both", print_prog=False):
    """Call .interpolate method on slices of X_series with lim_dir. Slices are specified by parts_left, parts_right.
     parts_left are starting indexes, parts_right are ending indexes thus both iterables must have same length."""
    X_series_filled=X_series.copy()
    for i in range(len(parts_left)):
        X_series_filled[parts_left[i]:
                    parts_right[i]]=X_series[parts_left[i]:
                                                 parts_right[i]].interpolate(limit_direction=lim_dir)
        if print_prog:
            print(i)
        for j in range(X_series_filled.shape[1]):
            if X_series_filled.iloc[parts_left[i]:parts_right[i],j].isnull().sum()!=0:
                print("Warning, in rows in slice(",parts_left[i],",",parts_right[i],") filling NaNs did not work. ")
    return X_series_filled

def scale_column_wise01(cols):
    return (cols - cols.min())/(cols.max()- cols.min())

def scale_by_parts01(parts_leftX, parts_rightX, parts_leftY, parts_rightY,
                     X_series_filled, Y_series):
    """Call scale_column_wise01 on slices of X_series_filled and Y_series_filled. Slices are specified by parts_left, parts_right.
     parts_left are starting indexes, parts_right are ending indexes thus both iterables must have same length."""
    X_series_scaled=X_series_filled.copy()
    Y_series_scaled=Y_series.copy()
    for i in range(len(parts_leftX)):
        X_series_scaled.iloc[parts_leftX[i]:
                    parts_rightX[i], 1:]=scale_column_wise01(X_series_filled.iloc[parts_leftX[i]:
                                                 parts_rightX[i], 1:])
        Y_series_scaled.iloc[parts_leftY[i]:parts_rightY[i], 1:]= scale_column_wise01( Y_series.iloc[parts_leftY[i]:
                                                                                                        parts_rightY[i], 1:])  
        for j in range(X_series_filled.shape[1]):
            if X_series_filled.iloc[parts_leftX[i]:parts_rightX[i],j].isnull().sum()!=0:
                print("Warning, in rows in X slice(",parts_leftX[i],",",parts_rightX[i],") NaNs encountered ")
                print("after scaling. Check if the values in that region are not constant.")
        if Y_series_scaled.iloc[parts_leftY[i]:parts_rightY[i], 1].isnull().sum()!=0:
            print("Warning, in rows in Y slice(",parts_leftY[i],",",parts_rightY[i],") NaNs encountered ")
            print("after scaling. Check if the values in that region are not constant.")
    return X_series_scaled, Y_series_scaled
	
	
	
def make_XY_sample_pairs(data_X, data_Y, i_len, o_len, nr=None):
    """Create 3 dimensional array of shape (samples, i_len,n_features) of X values
           from  2 dimensional data frame of shape(samples, n_features),
       and a 2 dimensional array of shape(samples, o_len) of Y values
           from  series of shape(samples)"""
    no_f = data_X.shape[1]
    no_samples= data_Y.shape[0] - o_len + 1
    X= np.empty(shape=(no_samples, i_len, no_f))
    Y= np.empty(shape=(no_samples, o_len))
    for i in range(no_samples):
      X[i,:,:] = data_X.iloc[i:i+i_len, :]
      Y[i, :] = data_Y.iloc[i:i+o_len]
      if (i%1000==0):
          print(i,"/",no_samples)
          if (nr is not None):
              print("of set ",nr)
    return X, Y
def make_XY_samples_by_parts(parts_leftX, parts_rightX, parts_leftY, parts_rightY,
                            X_series,
                            Y_series,
                            i_len, o_len,
                            save=True):
    """Apply make_XY_sample_pairs to X_series and Y_series subsets indexed by their respective parts_left and parts_right
       arguments (outputs from find_available_boundaries and XboundsfromYbounds).
       X_series and Y_series are data frames, first column of each should be Timestamp column,
       another columns of X_series are numerical values for descriptors, second column of Y should be numerical values of 
       value to predict.
       i_len and o_len specify input length to the network (how many timesteps to use for prediction)
       and output length of the network (how many future timesteps of Y to predict).
       Returns: list of arrays of made from X_series (X_sets) and list of arrays made from Y_series (Y_sets). 
       X_sets[i] and Y_sets[i] can be used as an input pair to the network with input shape (i_len, X_sets[i].shape[2])
       and output shape (o_len,1)"""
    X_sets=[]
    Y_sets=[]
    for i in range(len(parts_leftX)):
        X_samples, Y_samples= make_XY_sample_pairs(X_series.iloc[parts_leftX[i]:parts_rightX[i],1:],
                                                   Y_series.iloc[parts_leftY[i]:parts_rightY[i],1],i_len,o_len, nr=i)
        X_sets.append(X_samples)
        Y_sets.append(Y_samples)
    if save==True:
        with open("X_sets.txt", "wb") as fp:   #Pickling
            print("Saving...")
            pickle.dump(X_sets, fp)
        with open("Y_sets.txt", "wb") as fp:
            pickle.dump(Y_sets, fp)
            print("Done.")
    return X_sets, Y_sets
	
	
def Dumb_predict(Y, OneValueBeforeFirstY):
    dumb_prediction= np.empty(shape=(Y.shape[0], Y.shape[1]))
    dumb_prediction[0,:] = np.array([OneValueBeforeFirstY for i in range(Y.shape[1])])
    for j in range(1, Y.shape[0]):
        dumb_prediction[j,:]= np.array([Y[j-1,0] for i in range(Y.shape[1])])
    return dumb_prediction

def find_longest_chunks(cuts_leftY, cuts_rightY):
    chunk_lengths=np.array([cuts_rightY[i] - cuts_leftY[i] for i in range(len(cuts_leftY))])
    all_in_one_chunk = True if len(chunk_lengths)==1 else False
    if not all_in_one_chunk:
        longidx1, longidx2= np.argsort(-chunk_lengths)[0], np.argsort(-chunk_lengths)[1]
        return all_in_one_chunk, longidx1, longidx2
    else:
        return all_in_one_chunk, 0, 1
    
def TrainValTestPartition():
    """Ask the user to pick indexes of training, validation and testing sets. Returns dictionary of boundaries of parts of
    each of the training, validation and test set."""
    PartsLeft, PartsRight = {}, {}
    PartsLeft["train"], PartsLeft["validation"], PartsLeft["test"] = [], [], []
    PartsRight["train"], PartsRight["validation"], PartsRight["test"] = [], [], []
    cont = True
    while cont:
        print(" Getting parts to be used for training.")
        temp= int(input("Enter starting idx : "))
        PartsLeft["train"].append(temp)
        temp2= int(input("Enter ending idx : "))
        PartsRight["train"].append(temp2)
        ans= input("End entering training parts? (y/n)")
        if ans=="y":
            cont= False
    cont= True
    while cont:
        print(" Getting parts to be used for validation.")
        temp= int(input("Enter starting idx : "))
        PartsLeft["validation"].append(temp)
        temp2= int(input("Enter ending idx : "))
        PartsRight["validation"].append(temp2)
        ans= input("End entering validation parts? (y/n)")
        if ans=="y":
            cont= False
    cont = True
    while cont:
        print(" Getting parts to be used for testing.")
        temp= int(input("Enter starting idx : "))
        PartsLeft["test"].append(temp)
        temp2= int(input("Enter ending idx : "))
        PartsRight["test"].append(temp2)
        ans= input("End entering testing parts? (y/n)")
        if ans=="y":
            cont= False
    return PartsLeft, PartsRight


def Reg2PInterface(cl, cr):
    """ Helper function displaying avalaible boundaries in data set, from which user can choose train / val/ test split."""
    AllInOneChunk, a, b= find_longest_chunks(cl, cr)
    if AllInOneChunk:
        print("No Y contained missing values.")
        print("The available indexes of data for slicing are {}:{}, n_samples = {}".format(cl[0], cr[0], cr[0]-cl[0]))
    else:
        Lidx1, Lidx2= a , b
        print("Longest available slice of Y is {}:{}, second longest is {}:{}".format(cl[Lidx1],cr[Lidx1],cl[Lidx2],cr[Lidx2]))
        print("All available slices look as follows: ")
        for k in range(len(cl)):
            print("{}:{}, n_samples = ".format(cl[k], cr[k], cr[k]-cl[k]))
            
            
def Regularized2Partition( Y, i_len, o_len):
    """ wrapper going from load_XY_series() or make_regular_timesteps  -> Partition of data to train, validate and test sets."""
    cl, cr =find_available_boundariesY(Y, i_len, o_len)
    Reg2PInterface(cl, cr)
    PartitionLeftsY, PartitionRightsY = TrainValTestPartition()
    PartitionLeftsX, PartitionRightsX = {}, {}
    for key in ["train", "validation", "test"]:
        currentYL, currentYR = PartitionLeftsY[key], PartitionRightsY[key] 
        PartitionLeftsX[key], PartitionRightsX[key] =  Xbounds_fromYbounds(currentYL, currentYR, i_len)
    return PartitionLeftsY, PartitionRightsY, PartitionLeftsX, PartitionRightsX
    
    
def save_Partitions(PartsList_LyRyLxRx, name_prefix="_"):
    """Save the Y and X partitions to train, val and test sets. """
    outfile=name_prefix+"Partition.txt"
    with open(outfile, "wb") as fp:
        pickle.dump(PartsList_LyRyLxRx, fp)
    return outfile


def load_Partitions(from_file):
    with open(from_file, "rb") as fp:  
        PartsList_LyRyLxRx= pickle.load(fp)
    return PartsList_LyRyLxRx

def Partitions2Samples(PartsFile, Xfile, Yfile, Y_name="p",
                 o_len=1, i_len=8,
                 interpolateX=True, lim_dir="both",
                 Scale01=True,Standardize=False, 
                 save_samples=False):
    """ Give: partitions file name (from save_Partitions)
              X_series, Y_series filename (after regularization),
        Get: samples for model according to i_len, o_len.
        OutputForm: Python Dictionary with keys 'train', 'validation', 'test', where:
            under 'train' there is a list of sample sets for training. sample sets are np.arrays
            under 'validation' there is a list of sample sets for validtation. sample sets are np.arrays
            under 'test' there a is list of sample sets for testing. sample sets are np.arrays """
    PartsList_LyRyLxRx= load_Partitions(PartsFile)
    PLY, PRY, PLX, PRX = PartsList_LyRyLxRx
    X_s, Y_s=read_XY_series(X_filename=Xfile,
                            Y_filename=Yfile)
    X_s[Y_name] = Y_s[Y_name]
    XsetsParts, YsetsParts = {}, {}
    for key in ["train", "validation", "test"]:
        X_cl, X_cr, Y_cl, Y_cr = PLX[key], PRX[key], PLY[key], PRY[key]
        if interpolateX:
            X_s= interpolate_by_parts( X_cl, X_cr, X_s,lim_dir=lim_dir)
        if Standardize:
            pass
        if Scale01:
            X_s, Y_s= scale_by_parts01(X_cl, X_cr, Y_cl, Y_cr,
                                       X_s, Y_s)
        XsetsParts[key], YsetsParts[key]=make_XY_samples_by_parts(X_cl, X_cr, Y_cl, Y_cr,
                                        X_s,
                                        Y_s,
                                        i_len, o_len,
                                        save=save_samples)
    return XsetsParts, YsetsParts


def Reg2Samples(Xfile, Yfile, i_len, o_len, 
                Y_name="p", 
                partition_filename_prefix=None,
                interpolateX=True, lim_dir="both",
                Scale01=True,Standardize=False):
    """Wrapper including read_XY_series -> make_XY_samples pipeline.
       Give: regularized data after make_regular_timesteps filenames, i_len and o_len.
       During usage, partition for training / validation / testing occurs. User must choose indexes from presented options.
       For each of training / validation / testing, user can supply many pairs of start and ending indexes.
       Get: samples for model according to i_len, o_len.
       OutputForm: Python Dictionary with keys 'train', 'validation', 'test', where:
            under 'train' there is a list of sample sets for training. sample sets are np.arrays
            under 'validation' there is a list of sample sets for validtation. sample sets are np.arrays
            under 'test' there a is list of sample sets for testing. sample sets are np.arrays 
    """
    X_series, Y_series = read_XY_series(X_filename=Xfile,
                                        Y_filename=Yfile)
    X_series[Y_name] = Y_series[Y_name] #add Y to X, predictions will use past Y_name timesteps
    print("Loaded X_series and Y_series")
    print("Appended "+Y_name+" to X.")
    PLY, PRY, PLX, PRX = Regularized2Partition( Y_series, i_len, o_len)
    partsfile= save_Partitions([PLY, PRY, PLX, PRX],
                               name_prefix="_" if partition_filename_prefix is None else  partition_filename_prefix )
    print("Done partitioning. Saved partitions to "+partsfile)
    XsetsParts, YsetsParts = Partitions2Samples(PartsFile= partsfile, 
                                                Xfile=Xfile, Yfile=Yfile, Y_name="p",
                                                o_len=o_len, i_len=i_len,
                                                interpolateX=interpolateX, lim_dir=lim_dir,
                                                Scale01=Scale01,Standardize=Standardize, 
                                                save_samples=False)
    return XsetsParts, YsetsParts


def Reg2SamplesAuto(Xfile, Yfile, i_len, o_len, 
                Y_name="p", 
                partsfile=None,
                interpolateX=True, lim_dir="both",
                Scale01=True,Standardize=False):
    """Like Reg2Samples, but user gives partition filename beforehand. 
    """
    X_series, Y_series = read_XY_series(X_filename=Xfile,
                                        Y_filename=Yfile)
    X_series[Y_name] = Y_series[Y_name] #add Y to X, predictions will use past Y_name timesteps
    print("Loaded X_series and Y_series")
    print("Appended "+Y_name+" to X.")
    print("Partition filename is "+partsfile)
    XsetsParts, YsetsParts = Partitions2Samples(PartsFile= partsfile, 
                                                Xfile=Xfile, Yfile=Yfile, Y_name="p",
                                                o_len=o_len, i_len=i_len,
                                                interpolateX=interpolateX, lim_dir=lim_dir,
                                                Scale01=Scale01,Standardize=Standardize, 
                                                save_samples=False)
    return XsetsParts, YsetsParts


def build_and_compile(Layers, 
                      optimizer=tf.keras.optimizers.Adam, learning_rate=0.0001,
                     loss='mse', metrics=['mean_absolute_error'],
                     summary_filename=None):
    model= tf.keras.models.Sequential()
    for lay in Layers:
        print("adding layer")
        model.add(lay)
    if summary_filename is not None:
        with open(summary_filename, 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
    opt = optimizer(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss=loss, metrics=metrics)
    return model


def fit_partwise(model, XsetsParts, YsetsParts, covarIdx=None,
                epochs=300, verbose=1, callbacks=None):
    trainPartsX, trainPartsY= XsetsParts["train"], YsetsParts["train"]
    valPartsX, valPartsY = XsetsParts["validation"], YsetsParts["validation"]
    assert len(trainPartsX)==len(trainPartsY), "Error: there are {} parts of train set in X and {} in Y".format(len(trainPartsX), len(trainPartsY))
    assert len(valPartsX)==len(valPartsY), "Error: there are {} parts of val set in X and {} in Y".format(len(valPartsX), len(valPartsY))
    validX = np.concatenate(valPartsX)
    validY= np.concatenate(valPartsY)
    trainX=np.concatenate(trainPartsX)
    trainY=np.concatenate(trainPartsY)
    print("Input shape: ", trainX[:,:,covarIdx].shape)
    print(covarIdx)
    if covarIdx is None:
        print("if")
        mdl=model.fit(trainX, trainY, validation_data= (validX, validY), 
                  epochs=epochs, verbose=verbose, callbacks=callbacks)
    else:
        print("else")
        mdl=model.fit(trainX[:,:,covarIdx], trainY, validation_data= (validX[:,:,covarIdx], validY), 
                  epochs=epochs, verbose=verbose, callbacks=callbacks)
    return model, mdl

def pred_pairwise(model, XsetsParts, covarIdx=None):
    pred_parts=[]
    testPartsX = XsetsParts["test"]
    testX = np.concatenate(testPartsX)
    print("Input shape: ", testX[:,:,covarIdx].shape)
    print(covarIdx)
    if covarIdx is None:
        Prediction=model.predict(testX)
    else:
        Prediction=model.predict(testX[:,:,covarIdx])
    return Prediction


def test_mae_pairwise(prediction, YsetsParts):
    testY= np.concatenate(YsetsParts["test"])
    return np.abs(prediction - testY).mean()

def test_mse_pairwise(prediction, YsetsParts):
    testY= np.concatenate(YsetsParts["test"])
    return np.power(prediction - testY, 2).mean()

from tensorflow.keras import backend as K
from timeit import default_timer as timer 
def RunTest(onRegXWithName, onRegYWithName, of_model_specs, 
            with_i_len, with_o_len, model_name, savedir=None,
            Y_name="p",
            partition_filename_prefix=None,
            interpolateX=True, lim_dir="both",
            Scale01=True,Standardize=False,
            optimizer=tf.keras.optimizers.Adam, learning_rate=0.0001,
            loss='mse', metrics=['mean_absolute_error'],
            summary_filename=None,
            epochs=300, verbose=1, callbacks=None,
            use_past_Y_only=True,
            use_vars_idx=None):
    if not use_past_Y_only:
        if use_vars_idx is None:
            print("not only past Y to use, but no specification of use_vars_idx")
            aaa=[]
            more=True
            while more:
                zzz=int(input("Type ONE index of var to use."))
                aaa.append(zzz)
                an=input("More?: (y/n)")
                if an!="y":
                    more=False
            use_vars_idx=aaa
    Xfile, Yfile = onRegXWithName, onRegYWithName
    i_len, o_len = with_i_len, with_o_len
    XsetsParts, YsetsParts=Reg2Samples(Xfile, Yfile, i_len, o_len, 
                                        Y_name=Y_name, 
                                        partition_filename_prefix=partition_filename_prefix,
                                        interpolateX=interpolateX, lim_dir=lim_dir,
                                        Scale01=Scale01,Standardize=Standardize)
    if verbose>0:
        print("{} has samples.".format(model_name))
    K.clear_session()
    if savedir is not None:
        os.chdir(savedir)
    compiled_model= build_and_compile(of_model_specs, 
                    optimizer=optimizer, learning_rate=learning_rate,
                    loss=loss, metrics=metrics,
                    summary_filename=summary_filename)
    if verbose>0:
        print("{} compiled, summary in file {} ".format(model_name, summary_filename))
    if use_past_Y_only:
        shapetuple= XsetsParts["train"][0].shape
        use_vars_idx= [shapetuple[ len(shapetuple) -1] -1]
    training_start=timer()
    fitted_model, hist= fit_partwise(compiled_model, XsetsParts, YsetsParts, covarIdx=use_vars_idx,
                epochs=epochs, verbose=verbose, callbacks=callbacks)
    training_end= timer()
    training_time= training_end - training_start
    history_filename= model_name + "History.csv"
    training_history=pd.DataFrame(hist.history) 
    training_history.to_csv(history_filename)
   # if verbose>0:
   #     print("{} fitted, history in file {} ".format(model_name, history_filename))
    if verbose>0:
        print("{} training time is {} ".format(model_name, training_time))
    model_prediction= pred_pairwise(fitted_model, XsetsParts, covarIdx=use_vars_idx)
    MAE =test_mae_pairwise(model_prediction, YsetsParts)
    MSE =test_mse_pairwise(model_prediction, YsetsParts)
    result_test = {"mae":MAE,
                   "mse":MSE,
                    "prediction": model_prediction,
                    "trueY": np.concatenate(YsetsParts["test"]),
                    "time": training_time}
    with open(model_name+"TestPred.txt", "wb") as fp:   #Pickling
            print("Saving...")
            pickle.dump(result_test, fp)
    if verbose>0:
        print("{} tested, prediction,trueY, test errors, time in file {} ".format(model_name, model_name+"TestPred.txt"))
        print("Done with {}".format(model_name))
    return hist


def RunTestPartitionsGiven(onRegXWithName, onRegYWithName, of_model_specs, 
            with_i_len, with_o_len, model_name, savedir=None,
            Y_name="p",
            partition_filename=None,
            interpolateX=True, lim_dir="both",
            Scale01=True,Standardize=False,
            optimizer=tf.keras.optimizers.Adam, learning_rate=0.0001,
            loss='mse', metrics=['mean_absolute_error'],
            summary_filename=None,
            epochs=300, verbose=1, callbacks=None,
            use_past_Y_only=True,
            use_vars_idx=None):
    if not use_past_Y_only:
        if use_vars_idx is None:
            print("not only past Y to use, but no specification of use_vars_idx")
            aaa=[]
            more=True
            while more:
                zzz=int(input("Type ONE index of var to use."))
                aaa.append(zzz)
                an=input("More?: (y/n)")
                if an!="y":
                    more=False
            use_vars_idx=aaa
    Xfile, Yfile = onRegXWithName, onRegYWithName
    i_len, o_len = with_i_len, with_o_len
    XsetsParts, YsetsParts=Reg2SamplesAuto(Xfile, Yfile, i_len, o_len, 
                                        Y_name=Y_name, 
                                        partsfile=partition_filename,
                                        interpolateX=interpolateX, lim_dir=lim_dir,
                                        Scale01=Scale01,Standardize=Standardize)
    if verbose>0:
        print("{} has samples.".format(model_name))
    K.clear_session()
    if savedir is not None:
        os.chdir(savedir)
    compiled_model= build_and_compile(of_model_specs, 
                    optimizer=optimizer, learning_rate=learning_rate,
                    loss=loss, metrics=metrics,
                    summary_filename=summary_filename)
    if verbose>0:
        print("{} compiled, summary in file {} ".format(model_name, summary_filename))
    if use_past_Y_only:
        shapetuple= XsetsParts["train"][0].shape
        use_vars_idx= [shapetuple[ len(shapetuple) -1] -1]
    training_start=timer()
    fitted_model, hist= fit_partwise(compiled_model, XsetsParts, YsetsParts, covarIdx=use_vars_idx,
                epochs=epochs, verbose=verbose, callbacks=callbacks)
    training_end= timer()
    training_time= training_end - training_start
    history_filename= model_name + "History.csv"
    training_history=pd.DataFrame(hist.history) 
    training_history.to_csv(history_filename)
   # if verbose>0:
   #     print("{} fitted, history in file {} ".format(model_name, history_filename))
    if verbose>0:
        print("{} training time is {} ".format(model_name, training_time))
    model_prediction= pred_pairwise(fitted_model, XsetsParts, covarIdx=use_vars_idx)
    MAE =test_mae_pairwise(model_prediction, YsetsParts)
    MSE =test_mse_pairwise(model_prediction, YsetsParts)
    result_test = {"mae":MAE,
                   "mse":MSE,
                    "prediction": model_prediction,
                    "trueY": np.concatenate(YsetsParts["test"]),
                    "time": training_time}
    with open(model_name+"TestPred.txt", "wb") as fp:   #Pickling
            print("Saving...")
            pickle.dump(result_test, fp)
    if verbose>0:
        print("{} tested, prediction,trueY, test errors, time in file {} ".format(model_name, model_name+"TestPred.txt"))
        print("Done with {}".format(model_name))
    return hist