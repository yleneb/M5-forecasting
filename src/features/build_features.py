import sys
import os
import pathlib
import numpy as np
import pandas as pd
import dask.dataframe as dd
from distributed import Client
from joblib import Parallel, delayed
import multiprocessing
from tqdm import tqdm

ROOT = pathlib.Path().absolute()
RAW_DATA_PATH = ROOT / 'data' / 'raw'
INTERIM_DATA_PATH = ROOT / 'data' / 'interim'
PROCESSED_DATA_PATH = ROOT / 'data' / 'processed'
DAYS_PRED = 28

# endure this project is in the path
sys.path.insert(0, ROOT.absolute().as_posix())
from src.data.process_data import reduce_memory_usage

def _demand_features(group):

    d = {}
    demand = group['demand']
    # what was the demand x days ago
    for diff in [0, 1, 2]:
        shift = diff#DAYS_PRED + diff
        d[f"shift_t{shift}"] = demand.transform(
            lambda x: x.shift(shift))

    # what was the rolling x day std 28 days ago?
    for size in [7, 30, 60, 90, 180]:
        d[f"rolling_std_t{size}"] = demand.transform(
            lambda x: x.shift(DAYS_PRED).rolling(size).std())

    # rolling mean
    for size in [7, 30, 60, 90, 180]:
        d[f"rolling_mean_t{size}"] = demand.transform(
            lambda x: x.shift(DAYS_PRED).rolling(size).mean())

    # rolling skew and kurtosis
    d["rolling_skew_t30"] = demand.transform(
        lambda x: x.shift(DAYS_PRED).rolling(30).skew())
    d["rolling_kurt_t30"] = demand.transform(
        lambda x: x.shift(DAYS_PRED).rolling(30).kurt())

    # rolling days with sales 28 days ago
    for size in [7, 30, 60, 90, 180]:
        d[f"rolling_nonzero_sale_count_t{size}"] = demand.transform(
            lambda x: x.shift(DAYS_PRED).rolling(size).apply(np.count_nonzero))

    return d

def _price_features(group):
    d = {}
    sell_price = group['sell_price_cent']
    # change in price since yesterday
    shift_price_t1 = sell_price.transform(lambda x: x.shift(1))

    # fractional price change since yesterday
    d["price_change_t1"] = \
    (shift_price_t1 - sell_price) / (shift_price_t1)

    # max price over last year
    rolling_price_max_t365 = sell_price.transform(
        lambda x: x.shift(1).rolling(365).max())

    # fractional price diff between today and the year's max price
    d["price_change_t365"] = \
    (rolling_price_max_t365 - sell_price) / rolling_price_max_t365

    # rolling std price
    d["rolling_price_std_t7"] = sell_price.transform(
        lambda x: x.rolling(7).std())
    d["rolling_price_std_t30"] = sell_price.transform(
        lambda x: x.rolling(30).std())

    return d

def _time_features(group):
    attrs = [
        # "year",
        "quarter",
        # "month",
        # "week",
        "weekofyear",
        "day",
        # "dayofweek",
        "dayofyear",
        "is_year_end",
        "is_year_start",
        "is_quarter_end",
        "is_quarter_start",
        "is_month_end",
        "is_month_start"]

    d = {}
    date = group['date']
    # use the datetime column to extract more time features
    for attr in attrs:
        dtype = np.int16 if attr in ["year",'dayofyear'] else np.int8
        d[attr] = getattr(date.dt, attr).astype(dtype)

    return d

def _add_features(group):
    """I had an issue with some of the  variables. 
    If the item was on sale for a short period of time 
    and didn't sell often, then the rolling features
    are columns of NaN which cannot be back/forward filled.
    This only occured in ids (items in stores)
    
    The affected columns:
    rolling_std_t90, rolling_std_t180, rolling_mean_t90,
    rolling_mean_t180, rolling_nonzero_sale_count_t90,
    rolling_nonzero_sale_count_t180, price_change_t365
    
    In these cases I chose to fillna(0)
    """
    
    group = group.sort_values(by='date')

    new_features = {}
    new_features.update(_demand_features(group))
    new_features.update(_price_features(group))
    new_features.update(_time_features(group))

    new_features = \
    (pd.DataFrame
    .from_dict(new_features)
    .replace([np.inf,-np.inf], np.nan)
    .bfill()
    .ffill()
    .fillna(0)
    .pipe(reduce_memory_usage))

    # passthrough all other features
    return group.join(new_features)
   

def applyParallel(dfGrouped, func):
    retLst = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(func)(group) for name, group in tqdm(dfGrouped))
    return pd.concat(retLst)

def train_and_validation(df, dept, store):
    # dataset is too big, so add features to chunks then later join together
    dfGrouped=\
    (df
     .query("""dept_id==@dept and store_id==@store""")
     # some categorical types were loaded incorrectly
     .astype({c:'category' for c in ['wday','month','year']})
     # exclude evaluation for now
     .query("""part!='evaluation'""")
     .sort_values(by=['id','date'])
     .groupby('id', observed=True))
    
    # apply to each group in parallel
    # return \
    df = \
    (applyParallel(dfGrouped, func=_add_features)
     .reset_index(drop=True))

    (df
     # cannot save float16 to parquet
     .astype({col:np.float32 for col,dtype in zip(df.columns,df.dtypes) if dtype in [np.float16]})
     .to_parquet(PROCESSED_DATA_PATH / 'train_validation' / f'{dept}_{store}.parquet'))

if __name__=='__main__':
    depts = ['FOODS_1','FOODS_2','FOODS_3','HOBBIES_1','HOBBIES_2','HOUSEHOLD_1','HOUSEHOLD_2']
    stores = ['CA_1','CA_2','CA_3','CA_4','TX_1','TX_2','TX_3','WI_1','WI_2','WI_3']

    # on;y want to read in data once
    df = pd.read_parquet(INTERIM_DATA_PATH / 'combined_dataset.parquet')

    # feature engineering on each chunck and save as parquet
    for dept in depts:
        for store in stores:
            # print(dept, '|', store)
            train_and_validation(df, dept, store)
            
    del df
    
    # get a list of all the file paths
    files = os.listdir(PROCESSED_DATA_PATH / 'train_validation')  
    files = [PROCESSED_DATA_PATH / 'train_validation' / x for x in files]
    
    # read into pandas - do in parallel for speed
    retLst = Parallel(n_jobs=multiprocessing.cpu_count())
    retLst = retLst(delayed(pd.read_parquet)(path) for path in tqdm(files))
    
    # concatenate and save full file
    (pd.concat(retLst)
     .reset_index(drop=True)
     .to_parquet(PROCESSED_DATA_PATH / 'train_validation.parquet'))
    
# dask attempt


# def train_and_validation(verbose=False):
    
#     # df = \
#     # (dd
#     #  .read_parquet(INTERIM_DATA_PATH / 'combined_dataset.parquet', npartitions=1)
#     #  # use sample for testing
#     #  .query("""id==('HOBBIES_1_001_CA_1_validation','FOODS_3_161_WI_3_validation','FOODS_3_149_WI_3_validation')""")
#     #  # some categorical types were loaded incorrectly
#     #  .astype({c:'category' for c in ['wday','month','year']})
#     #  .astype({'id':'string'}))
#     # if verbose: print('data loaded')
    
#     df = \
#     (pd
#      .read_parquet(INTERIM_DATA_PATH / 'combined_dataset.parquet')
#      # use sample for testing
#     #  .query("""id==('HOBBIES_1_001_CA_1_validation','FOODS_3_161_WI_3_validation','FOODS_3_149_WI_3_validation')""")
#      # some categorical types were loaded incorrectly
#      .astype({c:'category' for c in ['wday','month','year']}))
#     if verbose: print('read into pandas')

        
#     # dask will partition by the index
#     df.id = df.id.cat.remove_unused_categories()
#     df.id = df.id.cat.as_ordered()
#     npartitions = len(df.id.cat.categories)-1
#     # partitions = df.id.cat.categories
#     # partitions = df['id'].unique()
#     # print(partitions)
    
#     # don't want to lose the id column so partition on a copy
#     df['use_as_index'] = df.id.copy()
#     df = df.set_index('use_as_index')
#     # df = df.set_index('id')
#     if verbose: print('df prepared for dask')
    
#     # read into dask, I have 64gb ram so persist
#     df = dd.from_pandas(df, npartitions=npartitions)
#     # df = df.repartition(npartitions=len(partitions)-1)
#     if verbose: print('read into dask')
#     # print(df.divisions)
    
#     df = df.persist()
#     if verbose: print('df persisted')

#     df = \
#     (df
#      .query("""part!='evaluation'""")
#      .map_partitions(_add_features)
#      .compute()
#      .reset_index(drop=True)
#     #  .reset_index(drop=False)
#      )
#     if verbose: print('features made')
    
#     df = \
#     (df
#      # cannot save float16 to parquet
#     .astype({col:np.float32 for col,dtype in zip(df.columns,df.dtypes) if dtype in [np.float16]})
#     .to_parquet(PROCESSED_DATA_PATH / 'train_validation.parquet'))
        
# if __name__=='__main__':
#     client = Client()
#     train_and_validation(verbose=True)