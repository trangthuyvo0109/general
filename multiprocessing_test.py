



## With para




def multipro():
    import multiprocessing as mp
    import time
    import pandas as pd
    import linearalg_np as ln

    start_time = time.time()
    cores = mp.cpu_count()
    
    # Creating multiprocessing pool
    pool = mp.Pool(cores)
    
    # Process the df
    #out_value_modi_para = pool.starmap(groupby_outvalue, [(out_value_list,row,col,25)])
    out_value_list_para = pool.starmap(ln.cal_linear_alg_lstsq, [(df_landcover_concat,df_lst, df_emiswb, df_emis_repre, row, col, [25] , "Emis", False, (290**4,310**4),  4  )])
    
    
    
    pool.close()
    pool.join()
    
    print(time.time() - start_time)
    
    return out_value_list_para
    

def normal():
    import time
    ## Without para
    start_time = time.time()
    #out_value_modi_norm = groupby_outvalue(out_value_list,row,col,3)
    
    
    
    out_value_list_non_agg = ln.cal_linear_alg_lstsq(df_landcover_concat, df_lst, df_emiswb, df_emis_repre,
                row, col, kernel_list=[25], _type="Emis", 
                bounds=(290**4,310**4), moving_pixel = 5,  radiance=False)
    
    
    print(time.time() - start_time)
    
    return out_value_list_non_agg

#normal()

def main():
    multipro()
    multiprocessing.freeze_support()
    

if __name__ == "__main__":
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    import linearalg_np as ln
    import pandas as pd
    import multiprocessing

    file = '//uahdata/rhome/py_code/aes509/data/out_value'
    
    out_value_list = pd.read_pickle(file)
    df_landcover_concat = pd.read_pickle('//uahdata/rhome/py_code/aes509/data/df_landcover_concat')
    df_lst = pd.read_pickle('//uahdata/rhome/py_code/aes509/data/df_lst')
    df_emiswb = pd.read_pickle('//uahdata/rhome/py_code/aes509/data/df_emiswb')
    df_emis_repre = pd.read_pickle('//uahdata/rhome/py_code/aes509/data/df_emis_repre')
    
    # Staten Island
    row = 303
    col = 243
    
    main()

    