Grouped time series data

3075 products, in 3 product categories, and 7 product departments, i.e.:
- Category Hobbies
    - Dep. Hobbies 1
        - item 1-416
    - Dep. Hobbies 2
        - item 437-565
- Category Foods
    - Dep. Foods 1
        - item ...
    - Dep. Foods 2
    - Dep. Foods 3
- Category Household
    - Dep. Household 1
    - Dep. Household 2

Products are sold across 10 stores in 3 states:
- California
- Texas
- Wisconsin

Historical date range from 2011-01-29 to 2016-06-19, so each product has a maximum selling history of 1941 days (5.4 years). The test data of h=28 days is not included.

The pdf guide has a collection of benchmarks, both statistical and ML based which I need to outperform.

# Model Checklist

- Random Forest
- AdaBoost
- XGBoost - try histogram mode
- LightGBM (see that it is much faster)
- CatBoost (should perform slightly better but be slowest)

# To-do list

- Initial exploration of the data with simple plots
- process the data with scripts in src (including normalising if necessary)
- Explore the example ML models to get a feeling for what is achievable
- Explore new models (including those above)
- Fine tune & ensemble
- Submit
- Repeat a few times
- Write up work and tidy the plots

# Work done

- Data loaded
- Processed
- Initial simple linear model
- Feature Engineering from kaggle
- Initial Lightgbm Model

# Successes

- Installed CUDA CUDnn
- Installed Tensorflow-gpu
- Installed LightGBM-gpu (with great difficulty)
- Worked in a tidy directory & environment
- Well documented progress in github
- Moving from notebooks to scripts
- Better understanding of pathlib, os, zipfile
- (kaggle) Found reduce_memory_usage function and made improvements 
- More experience with regex for extraction
- Better knowledge of data types (eg np.uint8)
- More familiarity with pandas io and file types
- Aggregated dataset for more familiar plotting
- (kaggle) Learnt some sensible demand and price features
- Applying a function to df.groupby groups in parallel
- Use of tqdm to create progress bars
- used itertools.product in place of a nested loop
- First experiments with dask
    - managed to perform feature engineering on a sample but got stuck when scaling to the full dataset
- (kaggle) found a CustomTimeSeriesSplitter and plotting function
- (kaggle) First experiment with LightGBM + working with gpu