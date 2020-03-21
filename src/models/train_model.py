"""
Notebook is based on m5-baseline (harupy), which is itself based on Very fst Model (ragnar123).

https://www.kaggle.com/ragnar123/very-fst-model
https://www.kaggle.com/harupy/m5-baseline"""


import sys
import os
import pathlib
import numpy as np
import pandas as pd
import lightgbm as lgb
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder

ROOT = pathlib.Path().absolute()
RAW_DATA_PATH = ROOT / 'data' / 'raw'
INTERIM_DATA_PATH = ROOT / 'data' / 'interim'
PROCESSED_DATA_PATH = ROOT / 'data' / 'processed'
DAYS_PRED = 28

# endure this project is in the path
sys.path.insert(0, ROOT.absolute().as_posix())
from src.data.process_data import reduce_memory_usage

def encode_categorical(df, cols):
    for col in cols:
        # Leave NaN as it is.
        le = LabelEncoder()
        not_null = df[col][df[col].notnull()]
        df[col] = pd.Series(le.fit_transform(not_null), index=not_null.index)

    return df

class CustomTimeSeriesSplitter:
    def __init__(self, n_splits=5, train_days=80, test_days=20, day_col="d"):
        self.n_splits = n_splits
        self.train_days = train_days
        self.test_days = test_days
        self.day_col = day_col

    def split(self, X, y=None, groups=None):
        SEC_IN_DAY = 3600 * 24
        sec = (X[self.day_col] - X[self.day_col].iloc[0]) * SEC_IN_DAY
        duration = sec.max()

        train_sec = self.train_days * SEC_IN_DAY
        test_sec = self.test_days * SEC_IN_DAY
        total_sec = test_sec + train_sec

        if self.n_splits == 1:
            train_start = duration - total_sec
            train_end = train_start + train_sec

            train_mask = (sec >= train_start) & (sec < train_end)
            test_mask = sec >= train_end

            yield sec[train_mask].index.values, sec[test_mask].index.values

        else:
            # step = (duration - total_sec) / (self.n_splits - 1)
            step = DAYS_PRED * SEC_IN_DAY

            for idx in range(self.n_splits):
                # train_start = idx * step
                shift = (self.n_splits - (idx + 1)) * step
                train_start = duration - total_sec - shift
                train_end = train_start + train_sec
                test_end = train_end + test_sec

                train_mask = (sec > train_start) & (sec <= train_end)

                if idx == self.n_splits - 1:
                    test_mask = sec > train_end
                else:
                    test_mask = (sec > train_end) & (sec <= test_end)

                yield sec[train_mask].index.values, sec[test_mask].index.values

    def get_n_splits(self):
        return self.n_splits

def show_cv_days(cv, X, dt_col, day_col):
    for ii, (tr, tt) in enumerate(cv.split(X)):
        print(f"----- Fold: ({ii + 1} / {cv.n_splits}) -----")
        tr_start = X.iloc[tr][dt_col].min()
        tr_end = X.iloc[tr][dt_col].max()
        tr_days = X.iloc[tr][day_col].max() - X.iloc[tr][day_col].min() + 1

        tt_start = X.iloc[tt][dt_col].min()
        tt_end = X.iloc[tt][dt_col].max()
        tt_days = X.iloc[tt][day_col].max() - X.iloc[tt][day_col].min() + 1

        df = pd.DataFrame(
            {
                "start": [tr_start, tt_start],
                "end": [tr_end, tt_end],
                "days": [tr_days, tt_days],
            },
            index=["train", "test"])

        display(df)

def plot_cv_indices(cv, X, dt_col, lw=10):
    n_splits = cv.get_n_splits()
    _, ax = plt.subplots(figsize=(20, n_splits))

    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(
            X[dt_col],
            [ii + 0.5] * len(indices),
            c=indices,
            marker="_",
            lw=lw,
            cmap=plt.cm.coolwarm,
            vmin=-0.2,
            vmax=1.2)

    # Formatting
    MIDDLE = 15
    LARGE = 20
    ax.set_xlabel("Datetime", fontsize=LARGE)
    ax.set_xlim([X[dt_col].min(), X[dt_col].max()])
    ax.set_ylabel("CV iteration", fontsize=LARGE)
    ax.set_yticks(np.arange(n_splits) + 0.5)
    ax.set_yticklabels(list(range(n_splits)))
    ax.invert_yaxis()
    ax.tick_params(axis="both", which="major", labelsize=MIDDLE)
    ax.set_title("{}".format(type(cv).__name__), fontsize=LARGE)
    return ax

def train_lgb(bst_params, fit_params, X, y, cv, drop_when_train=None):
    models = []

    if drop_when_train is None:
        drop_when_train = []

    for idx_fold, (idx_trn, idx_val) in enumerate(cv.split(X, y)):
        print(f"\n---------- Fold: ({idx_fold + 1} / {cv.get_n_splits()}) ----------\n")

        X_trn, X_val = X.iloc[idx_trn], X.iloc[idx_val]
        y_trn, y_val = y.iloc[idx_trn], y.iloc[idx_val]
        train_set = lgb.Dataset(X_trn.drop(drop_when_train, axis=1)#.values.astype(np.float32) # lgb just copies everything to float64?
                                , label=y_trn)
        val_set = lgb.Dataset(X_val.drop(drop_when_train, axis=1), label=y_val)

        model = lgb.train(
            bst_params,
            train_set,
            valid_sets=[train_set, val_set],
            valid_names=["train", "valid"],
            **fit_params,)
        models.append(model)

        del idx_trn, idx_val, X_trn, X_val, y_trn, y_val

    return models

def feature_importance(
    features,importances,importance_type,limit=None,normalize=False, figname='figure.png', figpath=ROOT):
    # from harupy's ml-extend

    features = np.array(features)
    importances = np.array(importances)
    indices = np.argsort(importances)

    if limit is not None:
        indices = indices[-limit:]

    if normalize:
        importances = importances / importances.sum()

    features = features[indices]
    importances = importances[indices]
    num_features = len(features)
    bar_pos = np.arange(num_features)

    # Adjust the figure height to prevent the plot from becoming too dense.
    w, h = plt.rcParams["figure.figsize"]
    h += 0.1 * num_features if num_features > 10 else 0

    fig, ax = plt.subplots(figsize=(w, h))
    ax.barh(bar_pos, importances, align="center", height=0.5)
    ax.set_yticks(bar_pos)
    ax.set_yticklabels(features)
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.set_title("Feature Importance ({})".format(importance_type))
    fig.tight_layout()
    plt.savefig(figpath / figname)
    
def make_submission(test, submission, save_path=ROOT):
    preds = test[["id", "date", "demand"]]
    preds = preds.pivot(index="id", columns="date", values="demand").reset_index()
    preds.columns = ["id"] + ["F" + str(d + 1) for d in range(DAYS_PRED)]

    evals = submission[submission["id"].str.endswith("evaluation")]
    vals = submission[["id"]].merge(preds, how="inner", on="id")
    final = pd.concat([vals, evals])

    assert final.drop("id", axis=1).isnull().sum().sum() == 0
    assert final["id"].equals(submission["id"])

    final.to_csv(save_path / "submission.csv", index=False)
    
#####################################################################
#####################################################################
#####################################################################

def build_model(features, name, cv_params, bst_params, fit_params, exclude_event_name=True):
    
    # read in the data
    print('Loading data')
    df = \
    (pd.read_parquet(PROCESSED_DATA_PATH / 'train_validation.parquet')#/ 'FOODS_1_CA_1.parquet')
    #  .head(1000)
    .astype({c:'category' for c in ['wday','month','year']}))
    
    print('converting "d" to integer')
    df['d'] = df['d'].astype('string').str.replace('d_','').astype(np.uint16)

    # leave out some columns for now
    if exclude_event_name:
        print('Dropping event_name columns')
        df = df.loc[:,[not x.startswith('event_name') for x in df.columns]]

    # follow kaggle example
    print('Applying Label Encoder')
    df = encode_categorical(df, ['item_id','dept_id','cat_id','store_id','state_id']).pipe(reduce_memory_usage)

    print('Applying CustomTimeSeriesSplitter')
    cv = CustomTimeSeriesSplitter(**cv_params)

    # sample = df.iloc[::1000][[day_col, dt_col]].reset_index(drop=True)
    # show_cv_days(cv, sample, dt_col, day_col)
    # plot_cv_indices(cv, sample, dt_col)

    # del sample

    print('Splitting train and test sets')
    mask = df['date'] <= '2016-04-24'
    # Attach "d" to X_train for cross validation.
    X_train = df[mask][[day_col] + features].reset_index(drop=True)
    y_train = df[mask]["demand"].reset_index(drop=True)
    X_test = df[~mask][features].reset_index(drop=True)

    # keep these two columns to use later.
    id_date = df[~mask][["id", "date"]].reset_index(drop=True)

    del df

    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)

    print('Training model')
    models = train_lgb(
        bst_params, fit_params, X_train, y_train, cv, drop_when_train=[day_col])

    print('Saving and Evaluating model')
    imp_type = "gain"
    importances = np.zeros(X_test.shape[1])
    preds = np.zeros(X_test.shape[0])
    
    for i, model in enumerate(models):
        save_path = ROOT / 'models' / name
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        model.save_model((save_path / str(str(i)+'_'+name+'.txt') ).as_posix())
        preds += model.predict(X_test)
        importances += model.feature_importance(imp_type)

    preds = preds / cv.get_n_splits()
    importances = importances / cv.get_n_splits()

    features = models[0].feature_name()
    feature_importance(features, importances, imp_type, limit=30, figname=name+'.png', figpath=ROOT / 'models' / name)
    
    submission = pd.read_csv(RAW_DATA_PATH / "sample_submission.csv")
    make_submission(id_date.assign(demand=preds), submission, save_path=ROOT / 'models' / name)

    
features = [
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id",
    "event_type_cultural",
    "event_type_national",
    "event_type_religious",
    "event_type_sporting",
    'event_name_chanukah end',
    'event_name_christmas',
    'event_name_cinco de mayo',
    'event_name_columbusday',
    'event_name_easter',
    'event_name_eid al-fitr',
    'event_name_eidaladha',
    "event_name_father's day",
    'event_name_halloween',
    'event_name_independenceday',
    'event_name_laborday',
    'event_name_lentstart',
    'event_name_lentweek2',
    'event_name_martinlutherkingday',
    'event_name_memorialday',
    "event_name_mother's day",
    'event_name_nbafinalsend',
    'event_name_newyear',
    'event_name_orthodoxchristmas',
    'event_name_orthodoxeaster',
    'event_name_pesach end',
    'event_name_presidentsday',
    'event_name_purim end',
    'event_name_ramadan starts',
    'event_name_stpatricksday',
    'event_name_superbowl',
    'event_name_thanksgiving',
    'event_name_valentinesday',
    'event_name_veteransday',
    "event",
    "snap",
    "sell_price_cent",
    # demand features.
    "shift_t28",
    "shift_t29",
    "shift_t30",
    "rolling_std_t7",
    "rolling_std_t30",
    "rolling_std_t60",
    "rolling_std_t90",
    "rolling_std_t180",
    "rolling_mean_t7",
    "rolling_mean_t30",
    "rolling_mean_t60",
    "rolling_mean_t90",
    "rolling_mean_t180",
    "rolling_skew_t30",
    "rolling_kurt_t30",
    'rolling_nonzero_sale_count_t7', # how many of the last 7 days were there >0 sales
    'rolling_nonzero_sale_count_t30',
    'rolling_nonzero_sale_count_t60',
    'rolling_nonzero_sale_count_t90',
    'rolling_nonzero_sale_count_t180',
    # price features
    "price_change_t1",
    "price_change_t365",
    "rolling_price_std_t7",
    "rolling_price_std_t30",
    # time features.
    "year",
    "month",
    "weekofyear",
    "day", # day of month
    "wday",
    "is_weekend",
    'quarter',
    'dayofyear',
    'is_year_end',
    'is_year_start',
    'is_quarter_end',
    'is_quarter_start',
    'is_month_end',
    'is_month_start'
    ]

bst_params = {
    "boosting_type": "gbdt",
    "metric": "rmse",
    "objective": "regression",
    "n_jobs": -1,
    # "seed": 42,
    "learning_rate": 0.1,
    "bagging_fraction": 0.75,
    "bagging_freq": 10,
    "colsample_bytree": 0.75,
    # 'device': 'gpu',
    # 'gpu_platform_id': 0,
    # 'gpu_device_id': 0
}

fit_params = {
    "num_boost_round": 100_000,
    "early_stopping_rounds": 200,
    "verbose_eval": 100}

day_col = "d"
dt_col = "date"

cv_params = {
    "n_splits": 7,
    "train_days": 365 * 2,
    "test_days": DAYS_PRED,
    "day_col": day_col}

# name = 'lgb-event-names'
# name = 'top-30-features+extra-7fold'
name = 'all-features-7fold'
exclude_event_name = False

if __name__=='__main__':
    build_model(features, name, cv_params, bst_params, fit_params, exclude_event_name)