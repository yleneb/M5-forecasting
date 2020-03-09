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