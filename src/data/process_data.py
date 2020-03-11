import pandas as pd
import pathlib

ROOT = pathlib.Path().absolute()
RAW_DATA_PATH = ROOT / 'data' / 'raw'
PROCESSED_DATA_PATH = ROOT / 'data' / 'processed'

def make_processed_calendar_dataset():
    categorical_variables = [
        'weekday','wday','month',
        'event_name_1','event_type_1',
        'event_name_2','event_type_2']
    boolean_variables = ['snap_CA','snap_TX','snap_WI']
    dtypes = {var: 'category' for var in categorical_variables}

    # read the data with chosen data types
    df = pd.read_csv(RAW_DATA_PATH / 'calendar.csv', parse_dates=['date'], dtype=dtypes)
    # convert snap columns to boolean
    df = df.astype({var: 'boolean' for var in boolean_variables})
    df.to_feather(PROCESSED_DATA_PATH / 'calendar_processed.feather')

def make_processed_sale_price_dataset():
    # really I want to convert sell_price to pence then make an int16
    dtypes = {'store_id': 'category', 'item_id':'category', 'wm_yr_wk': 'int16'}

    # read the data with chosen data types
    df = pd.read_csv(RAW_DATA_PATH / 'sell_prices.csv', dtype=dtypes)
    df['sell_price_cent'] = (df['sell_price']*100).astype('int16')
    df = df.drop(columns=['sell_price'])
    
    # get the state from the store_id - eg: CA_1, TX_3
    df['state_id'] = \
    (df.store_id
    .replace({x: x.split('_')[0] for x in df.store_id.cat.categories})
    .astype('category'))

    # get the department and catyegory from the item_id
    df = df.join(
        df.item_id
        .str.extract("^(?P<dept_id>(?P<cat_id>[A-Z]+)\_\d)")
        .astype('category'))
    
    # reoreder columns
    df = df[['state_id','store_id','cat_id','dept_id','item_id','wm_yr_wk','sell_price_cent']]
        
    df.to_feather(PROCESSED_DATA_PATH / 'sell_prices_processed.feather')
    
    
def make_processed_sales_dataset():
    categorical_variables = ['dept_id','cat_id', 'store_id','state_id']
    int_variables = (f'd_{i}' for i in range(1, 1914))
    string_variables = ['id','item_id']

    dtypes = {}
    dtypes.update({var: 'int16' for var in int_variables})
    dtypes.update({var: 'category' for var in categorical_variables})
    dtypes.update({var: 'string' for var in string_variables})

    # exclude the id column as same as item_id since all rows are '_validation'
    df = pd.read_csv(RAW_DATA_PATH / 'sales_train_validation.csv',
                     dtype=dtypes,
                     usecols=lambda col: col not in ['id'])
    
    # # convert validation set to a binary flag
    # df['valid'] = df.pop('id').str.endswith('validation')
    df.to_feather(PROCESSED_DATA_PATH / 'sales_train_validation_processed.feather')

def make_combined_dataset():
    # get data
    cal_df   = pd.read_feather(PROCESSED_DATA_PATH / 'calendar_processed.feather')
    price_df = pd.read_feather(PROCESSED_DATA_PATH / 'sell_prices_processed.feather')
    sales_df = pd.read_feather(PROCESSED_DATA_PATH / 'sales_train_validation_processed.feather')
    
    price_df = price_df[['store_id','item_id','wm_yr_wk','sell_price_cent']]
    
    df = \
    (sales_df
    # make data a single column of prices with all else in the index (including 'd')
    .set_index(['item_id','dept_id','cat_id','store_id','state_id'])
    .rename_axis(columns=['d'])
    .stack()
    # name this new col and reset the index
    .rename('sales')
    .reset_index()
    # join in the calendar
    .merge(cal_df, left_on='d', right_on='d')
    .drop(columns=['d','wday'])
    # set a boolean, did this day's sales happen on a snap day in that state
    .eval("""snap = """
        """(state_id=='CA' and snap_CA==True) or """
        """(state_id=='TX' and snap_TX==True) or """
        """(state_id=='WI' and snap_WI==True)""")
    .drop(columns=['snap_CA','snap_TX','snap_WI']))

    # create boolean columns, did this days' sales happen on an event / event type
    event_types = (
    (pd.get_dummies(df.event_type_1) | pd.get_dummies(df.event_type_2))
    .astype(bool)
    .rename(columns=lambda x: 'event_type_'+x.lower()))

    event_names = (
    (pd.get_dummies(df.event_name_1) | pd.get_dummies(df.event_name_2))
    .astype(bool)
    .rename(columns=lambda x: 'event_name_'+x.lower()))

    # join in the booleans for events
    df = \
    (pd.concat([df, event_types, event_names], axis=1)
     # boolean flag for if there was any event on that day
    .assign(event=event_types.any(axis=1))
    # merge in the prices - what was the value of the product on that day and in that store
    .merge(price_df, on=['store_id','item_id','wm_yr_wk'])
    # drop unnecessary columns
    .drop(columns=['event_type_1','event_type_2','event_name_1','event_name_2','wm_yr_wk'])
    .astype({'item_id':'string'}))

    # save to feather
    df.to_parquet(PROCESSED_DATA_PATH / 'combined_dataset.parquet')
    
def make_aggregated_dataset():
    # read in the combined dataset
    df = pd.read_parquet(PROCESSED_DATA_PATH / 'combined_dataset.parquet')

    # how will the columns be aggregated
    to_agg = {'sales':'sum'}
    to_agg.update({x:'first' for x in df.columns if x.startswith('event_') or x=='snap'})

    # groupby everything except item_id and columns top aggregate
    aggregated_sales = \
    (df
    # .sample(frac=0.001)
    .groupby(['state_id','store_id','date','year','month','weekday','cat_id','dept_id'], observed=True)
    .agg(to_agg)
    .astype({'sales':'int16'})
    .sort_index(axis=1))

    # create a boolean calendar which can join the aggregated data
    aggregated_calendar = \
    (aggregated_sales
    .drop(columns=['sales'])
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
    [['sales']]
    .unstack(['cat_id','dept_id'], fill_value=0))

    # join in the boolean calendar
    aggregated_sales = \
    (aggregated_calendar
    .join(aggregated_sales)
    .sort_index(axis=1, ascending=False)
    .sort_index(axis=0))

    aggregated_sales.to_pickle(PROCESSED_DATA_PATH / 'aggregated_dataset.pickle')
    
# make_processed_sale_price_dataset()
# make_processed_sales_dataset()
# make_processed_calendar_dataset()
# make_combined_dataset()
# make_aggregated_dataset()