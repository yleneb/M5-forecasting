import pandas as pd
import pathlib

ROOT = pathlib.Path().absolute()
RAW_DATA_PATH = ROOT / 'data' / 'raw'
PROCESSED_DATA_PATH = ROOT / 'data' / 'processed'

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

    
# make_processed_sale_price_dataset()
# make_processed_sales_dataset()