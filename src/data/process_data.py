import pandas as pd
import numpy as np
import pathlib

ROOT = pathlib.Path().absolute()
RAW_DATA_PATH = ROOT / 'data' / 'raw'
PROCESSED_DATA_PATH = ROOT / 'data' / 'processed'

DAYS_PRED = 28

def reduce_memory_usage(df, verbose=False):
    """Check if each col is numeric, then reduce it's dtype to the lowest memory
    
    pandas only supports datetime64[ns]
    """

    if verbose:
        starting_memory_usage = df.memory_usage(deep=True).sum() / 1024**2

    # try each column
    for col in df.columns:
        col_type = df[col].dtypes

        # Downsample numeric dtypes - no need to reduce columns of type int8
        if col_type in [np.int16, np.int32, np.int64, np.float16, np.float32, np.float64]:
            c_min, c_max = df[col].agg(['min','max'])

            # rare case of boolean column
            if c_min in [0,1] and c_max in [0,1]:
                if all(df[col].isin([0,1])):
                    df[col] = df[col].astype('bool')
                    continue

            # if col is float and cannot be converted to int then only try float dtypes
            if str(col_type).startswith('float') and not np.array_equal(df[col], df[col].astype(int)):
                dtypes = [np.float16, np.float32, np.float64]
            # if all values are positive we can use unsigned integer
            elif all(df[col]>=0):
                dtypes = [np.uint8, np.uint16, np.uint32, np.uint64]
            else:
                dtypes = [np.int8, np.int16, np.int32, np.int64]

            # get the info about each dtype
            dtype_info = [np.iinfo(dtype) if np.issubdtype(dtype, np.integer) else np.finfo(dtype) for dtype in dtypes]

            # try all the smaller dtypes in order of size
            for dtype, info in zip(dtypes, dtype_info):
                # if this dtype is suitable,. use it and break, else try the next dtype
                if c_min >= info.min and c_max <= info.max:
                    df[col] = df[col].astype(dtype)
                    break

    if verbose:
        ending_memory_usage = df.memory_usage(deep=True).sum() / 1024**2
        reduction = 100*(starting_memory_usage-ending_memory_usage)/starting_memory_usage
        print(f"Memory usage decreased from {np.round(starting_memory_usage,3)} Mb "
              f"to {np.round(ending_memory_usage,3)} Mb ({np.round(reduction,1)}% reduction)")

    return df

def make_processed_calendar_dataset(verbose=True):
    categorical_variables = [
        'd','weekday','wday','month', 'year',
        'event_name_1','event_type_1',
        'event_name_2','event_type_2']
    dtypes = {var: 'category' for var in categorical_variables}
    dtypes['d'] = 'string'

    # read the data
    df = pd.read_csv(RAW_DATA_PATH / 'calendar.csv', parse_dates=['date'])

    if verbose:
        starting_memory_usage = df.memory_usage(deep=True).sum() / 1024**2

    # update data types
    df = df.astype(dtypes).pipe(reduce_memory_usage)

    if verbose:
        ending_memory_usage = df.memory_usage(deep=True).sum() / 1024**2
        reduction = 100*(starting_memory_usage-ending_memory_usage)/starting_memory_usage
        print(f"Memory usage decreased from {np.round(starting_memory_usage,3)} Mb "
              f"to {np.round(ending_memory_usage,3)} Mb ({np.round(reduction,1)}% reduction)")

    # save to feather
    df.to_feather(PROCESSED_DATA_PATH / 'calendar_processed.feather')

def make_processed_sale_price_dataset(verbose=True):
    # really I want to convert sell_price to pence then make an int16
    dtypes = {'store_id': 'category', 'item_id':'category', 'sell_price_cent':'int'}

    # read the data with chosen data types
    df = pd.read_csv(RAW_DATA_PATH / 'sell_prices.csv')

    if verbose:
        starting_memory_usage = df.memory_usage(deep=True).sum() / 1024**2

    df['sell_price_cent'] = df.pop('sell_price')*100
    df = df.astype(dtypes)

    # get the state from the store_id - eg: CA_1, TX_3
    df['state_id'] = \
    (df.store_id
    .replace({x: x.split('_')[0] for x in df.store_id.cat.categories})
    .astype('category'))

    # get the department and category from the item_id
    df = \
    (df.join(
        df.item_id
        .str.extract("^(?P<dept_id>(?P<cat_id>[A-Z]+)\_\d)")
        .astype('category'))
    # reduce numeric features
    .pipe(reduce_memory_usage)
    # reorder columns
    [['state_id','store_id','cat_id','dept_id','item_id','wm_yr_wk','sell_price_cent']])

    if verbose:
        ending_memory_usage = df.memory_usage(deep=True).sum() / 1024**2
        reduction = 100*(starting_memory_usage-ending_memory_usage)/starting_memory_usage
        print(f"Memory usage decreased from {np.round(starting_memory_usage,3)} Mb "
              f"to {np.round(ending_memory_usage,3)} Mb ({np.round(reduction,1)}% reduction)")
        
    df.to_feather(PROCESSED_DATA_PATH / 'sell_prices_processed.feather')
    
def make_processed_sales_dataset(verbose=True, sample_submission=False):
    
    file_name_load = 'sales_train_validation' if not sample_submission else 'sample_submission'
    file_name_save = 'sales_train_validation_processed' if not sample_submission else 'submission'
    
    categorical_variables = ['dept_id','cat_id', 'store_id','state_id']
    string_variables = ['item_id','id']
    
    dtypes = {}
    dtypes.update({var: 'category' for var in categorical_variables})
    dtypes.update({var: 'string' for var in string_variables})
    
    # read in the data
    df = pd.read_csv(RAW_DATA_PATH / f'{file_name_load}.csv')
    
    if verbose:
        starting_memory_usage = df.memory_usage(deep=True).sum() / 1024**2
    
    # extract the product and store info from the id if not available
    if sample_submission:
        # product info
        regex = "^(?P<item_id>(?P<dept_id>(?P<cat_id>[A-Z]+)\_\d)\_\d*)"
        # store info
        regex+= "\_" + "(?P<store_id>(?P<state_id>[A-Z]+)\_\d)"
        df = df.join(df.id.str.extract(regex).astype('category'))

    df = df.astype(dtypes).pipe(reduce_memory_usage)

    if verbose:
        ending_memory_usage = df.memory_usage(deep=True).sum() / 1024**2
        reduction = 100*(starting_memory_usage-ending_memory_usage)/starting_memory_usage
        print(f"Memory usage decreased from {np.round(starting_memory_usage,3)} Mb "
              f"to {np.round(ending_memory_usage,3)} Mb ({np.round(reduction,1)}% reduction)")

    df.to_feather(PROCESSED_DATA_PATH / f'{file_name_save}.feather')

def make_melted_sales_dataset():
    """Manipulate the sales and submission data to return a df with:
    One row per day of sales at each store for each item"""
    sales_train_val = \
    (pd.read_feather(PROCESSED_DATA_PATH / 'sales_train_validation_processed.feather')
    .astype({'id':'string','item_id':'string'}))

    submission = \
    (pd.read_feather(PROCESSED_DATA_PATH / 'submission.feather')
    .astype({'id':'string', 'item_id':'string'}))

    # get product information table
    id_columns = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]

    # melt sales data
    sales_train_val = pd.melt(sales_train_val, id_vars=id_columns, var_name='d', value_name='demand')

    # separate test dataframes
    validation = submission[submission.id.str.endswith('validation')]
    evaluation = submission[submission.id.str.endswith('evaluation')]

    # change columns names
    validation.columns = ['id']+[f"d_{d}" for d in range(1914, 1914 + DAYS_PRED)]+id_columns[1:]
    evaluation.columns = ['id']+[f"d_{d}" for d in range(1942, 1942 + DAYS_PRED)]+id_columns[1:]

    # return validation
    # melt the validation and evaluation dataframes
    validation = pd.melt(validation, id_vars=id_columns, var_name="d", value_name="demand")
    evaluation = pd.melt(evaluation, id_vars=id_columns, var_name="d", value_name="demand")

    # join together all datasets and set data types
    sales_train_val = \
    (pd
    .concat([sales_train_val, validation, evaluation],  axis=0,
            keys=['train','validation','evaluation'], names=['part']))
    
    del validation, evaluation
    
    sales_train_val = \
    (sales_train_val
    .reset_index('part')
    .reset_index(drop=True)
    .astype({'d':'category','demand':'int','part':'category',
            'id':'category', 'item_id':'category'})
    .pipe(reduce_memory_usage))

    sales_train_val.to_parquet(PROCESSED_DATA_PATH / 'melted_sales.parquet')

def make_combined_dataset(verbose=False):
    verbose=True
    cal_df = pd.read_feather(PROCESSED_DATA_PATH / 'calendar_processed.feather').astype({'d':'category'})

    # create boolean columns, did this days' sales happen on an event / event type
    event_types = (
    (pd.get_dummies(cal_df.event_type_1) | pd.get_dummies(cal_df.event_type_2))
    .astype(bool)
    .rename(columns=lambda x: 'event_type_'+x.lower()))

    event_names = (
    (pd.get_dummies(cal_df.event_name_1) | pd.get_dummies(cal_df.event_name_2))
    .astype(bool)
    .rename(columns=lambda x: 'event_name_'+x.lower()))
    if verbose: print('Events one hot encoded')

    # create the calendar including booleans for is_weekend and event
    cal_df = pd.concat([cal_df, event_types, event_names], axis=1)
    cal_df['is_weekend'] = cal_df.weekday.isin(['Saturday','Sunday'])
    cal_df['event'] = event_types.any(axis=1)
    del event_types, event_names
    if verbose: print('cal_df complete')
        
    # join calendar to sales data
    df = pd.read_parquet(PROCESSED_DATA_PATH / 'melted_sales.parquet')

    df = df.join(cal_df.set_index('d'), how='left', on='d').reset_index(drop=True)
    del cal_df
    if verbose: print('sales df joined to cal_df')

    # create a boolean: did this day's sales happen on a snap day in that state
    df = \
    (df.eval("""snap = """
        """(state_id=='CA' and snap_CA==True) or """
        """(state_id=='TX' and snap_TX==True) or """
        """(state_id=='WI' and snap_WI==True)""")
    .drop(columns=['snap_CA','snap_TX','snap_WI']))
    if verbose: print('snap boolean created')

    # read in the item prices
    price_df = pd.read_feather(
        PROCESSED_DATA_PATH / 'sell_prices_processed.feather',
        columns=['store_id','item_id','wm_yr_wk','sell_price_cent'])

    # join in the item prices
    df = \
    (df
    .join(price_df.set_index(['store_id','item_id','wm_yr_wk']),
        how='left', on=['store_id','item_id','wm_yr_wk'])
    # if there was no price available then item was not on sale
    .dropna(subset=['sell_price_cent']))
    del price_df
    if verbose: print('prices joined to sales')

    # drop unnecessary columns and set data types
    df = \
    (df
    .drop(columns=['event_type_1','event_type_2','event_name_1','event_name_2','wm_yr_wk'])
    .astype({'item_id':'category','demand':'int'})
    .reset_index(drop=True)
    .pipe(reduce_memory_usage))
    if verbose: print('dataframe complete')

    # save to feather
    df.to_parquet(PROCESSED_DATA_PATH / 'combined_dataset.parquet')
    if verbose:
        print('dataset saved')
        print(df.shape)
        df.info()
    
def make_aggregated_dataset():
    # read in the combined dataset
    df = \
    (pd.read_parquet(PROCESSED_DATA_PATH / 'combined_dataset.parquet')
    .astype({c:'category' for c in ['wday','month','year']}))

    # how will the columns be aggregated
    to_agg = {'demand':'sum'}
    to_agg.update({x:'first' for x in df.columns if x.startswith('event_') or x=='snap'})

    # groupby everything except item_id and columns top aggregate
    aggregated_sales = \
    (df
    .groupby(['state_id','store_id','date','year','month','weekday','cat_id','dept_id'], observed=True)
    .agg(to_agg)
    .sort_index(axis=1))

    # create a boolean calendar which can join the aggregated data
    aggregated_calendar = \
    (aggregated_sales
    .drop(columns=['demand'])
    .groupby(['state_id','store_id','date','year','month','weekday'], observed=True)
    .first())
    
    # add more levels to multiindex for intuitive indexing
    aggregated_calendar = pd.concat(
        [aggregated_calendar[['snap']],
         aggregated_calendar[[x for x in aggregated_calendar.columns if x.startswith('event_type_')]],
         aggregated_calendar[[x for x in aggregated_calendar.columns if x.startswith('event_name_')]]
         ], keys=['snap','type','name'], axis=1)
    
    aggregated_calendar = pd.concat(
        [aggregated_calendar[('snap','snap')],
         aggregated_calendar[['type','name']]],
        keys=['snap','events'], axis=1)
    
    # rename the columns to cutr off 'event_name_' and 'event_type_'
    aggregated_calendar.rename(lambda col: col.split('_')[-1], axis='columns', inplace=True, level=2)

    # unstack the category and department indices for more intuitive indexing
    aggregated_sales = \
    (aggregated_sales
    [['demand']]
    .unstack(['cat_id','dept_id'], fill_value=0))

    # join in the boolean calendar
    aggregated_sales = \
    (aggregated_calendar
    .join(aggregated_sales)
    .sort_index(axis=1, ascending=False)
    .sort_index(axis=0)
    .pipe(reduce_memory_usage))

    aggregated_sales.to_pickle(PROCESSED_DATA_PATH / 'aggregated_dataset.pickle')
    
# make_processed_calendar_dataset(True)
# make_processed_sale_price_dataset(False)
# make_processed_sales_dataset(False)
# make_processed_sales_dataset(False, sample_submission=True)

# make_melted_sales_dataset()
# make_combined_dataset(False)

# make_aggregated_dataset()