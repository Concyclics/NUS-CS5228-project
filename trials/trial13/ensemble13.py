# %%
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats import zscore
import ray
from flaml import AutoML


# %%
#setconfig

ray.init(num_cpus=128, ignore_reinit_error=True)
#np.setbufsize(1e6)
#np.getbufsize()

config = {
    "time_limits": 80000, # in seconds
    "early_stop": False,
    "max_threads": 4, # threads for each learner
    "parallel_trail": 32, # trails in parallel
    "log_postfix": 'trial13.log',
    "eval_method": 'cv',
    "use_model": ['lgbm', 'xgboost', 'catboost'],
    "model_weights": [1, 1, 1],
}

# %%
path_prefix = '~/CS5228/cs5228-2310-final-project/'
df_train = pd.read_csv(path_prefix + 'train.csv')
df_test = pd.read_csv(path_prefix + 'test.csv')

# %%
def category_map(df_origin, df_train_):
    df = df_origin.copy()
    df_train_ = df_train_.copy()
    colums = ['town', 'flat_type', 'flat_type2', 'cluster', 'street_name', 'block',
              'lease_commence_date', 'flat_model', 'subzone', 'planning_area','region']
    #locations = ['town', 'block', 'street_name', 'subzone', 'planning_area','region', 'cluster']
    
    for col in colums:
        df_train_[col] = df_train_[col].astype(str).str.lower()
        df[col] = df[col].astype(str).str.lower()

    df['block'] = df['street_name'] + ' ' + df['block']
    df_train_['block'] = df_train_['street_name'] + ' ' + df_train_['block']

    for col in colums:
        group_mean = df_train_.groupby(col)['monthly_rent'].mean()
        group_std = df_train_.groupby(col)['monthly_rent'].std()
        group_median = df_train_.groupby(col)['monthly_rent'].median()
        group_cnt = df_train_[col].value_counts()
        total_mean = df_train_['monthly_rent'].mean()
        total_median = df_train_['monthly_rent'].median()
        total_std = df_train_['monthly_rent'].std()
        cat_map = group_cnt[group_cnt.values >= 8].index.tolist()
        std_map = group_std.to_dict()
        mean_map = group_mean.to_dict()
        median_map = group_median.to_dict()

        #df[col] = df[col].apply(lambda x: x if x in cat_map else 'other')
        df[col+'_std'] = df[col].apply(lambda x: std_map[x] if x in cat_map else 0)
        #df[col+'_mean'] = df[col].apply(lambda x: mean_map[x] if x in cat_map else -10)
        #df[col+'_median'] = df[col].apply(lambda x: median_map[x] if x in cat_map else -10)
        #df.drop(columns=[col], inplace=True)
        
    return df


# %%
#add KNN feature
def add_KNN_feature(df_origin, df_pos, K: int):
    KNN_X = df_pos[['latitude', 'longitude']]
    KNN_X['latitude'] *= 2
    KNN_y = df_pos['monthly_rent']

    KNN_model = KNeighborsRegressor(n_neighbors=K)
    KNN_model.fit(KNN_X, KNN_y)

    KNN_y2 = KNN_y ** 2
    KNN_model2 = KNeighborsRegressor(n_neighbors=K)
    KNN_model2.fit(KNN_X, KNN_y2)

    predict_X = df_origin[['latitude', 'longitude']]
    predict_X['latitude'] *= 2
    predict_y = KNN_model.predict(predict_X)
    predict_y2 = KNN_model2.predict(predict_X)
    df = df_origin.copy()
    #use std to be the feature
    df['K=' + str(K) + ' KNN_std'] = np.sqrt(predict_y2 - predict_y ** 2)
    #df['K=' + str(K) + ' KNN_mean'] = predict_y

    return df

# %%
def data_preprocess(df, df_train_, category_mapping=True):
    df = df.copy()
    df_train_ = df_train_.copy()

    df['flat_type2'] = df['flat_type'].str.replace('-', ' ')
    df_train_['flat_type2'] = df_train_['flat_type'].str.replace('-', ' ')
    #df.drop(['block'], axis=1, inplace=True)
    
    #normalize by date
    #df['monthly_rent'] = np.log(df['monthly_rent'])
    #df_train_['monthly_rent'] = np.log(df_train_['monthly_rent'])
    means = df_train_.groupby('rent_approval_date')['monthly_rent'].mean()
    stds = df_train_.groupby('rent_approval_date')['monthly_rent'].std()
    median = df_train_.groupby('rent_approval_date')['monthly_rent'].median()

    df['monthly_rent'] = df.apply(lambda x: (x['monthly_rent'] - means[x['rent_approval_date']]) / stds[x['rent_approval_date']], axis=1)
    df_train_['monthly_rent'] = df_train_.apply(lambda x: (x['monthly_rent'] - means[x['rent_approval_date']]) / stds[x['rent_approval_date']], axis=1)
    #normalize monthly rent by date
    
    #add coe price
    df_coe = pd.read_csv(path_prefix + 'auxiliary-data/auxiliary-data/sg-coe-prices.csv')
    month_to_numeric = {
        'january': 1,
        'february': 2,
        'march': 3,
        'april': 4,
        'may': 5,
        'june': 6,
        'july': 7,
        'august': 8,
        'september': 9,
        'october': 10,
        'november': 11,
        'december': 12
    }

    df_coe['month'] = df_coe['month'].apply(lambda x: month_to_numeric[x.lower()])
    df_coe['month'] = df_coe['month'].apply(lambda x: f'{x:02d}')

    df_coe['date'] = df_coe['year'].astype(str) + '-' + df_coe['month']
    df_coe['date'] = pd.to_datetime(df_coe['date'])
    df_coe = df_coe[['date', 'price']]
    avg_price_bids_quota = df_coe.groupby('date').mean().reset_index()
    avg_price_bids_quota['date'] = pd.to_datetime(avg_price_bids_quota['date'])

    df['coe_price'] = df['rent_approval_date'].apply(lambda x: avg_price_bids_quota[avg_price_bids_quota['date'] == x]['price'].values[0])

    
    if category_mapping:
        df = category_map(df, df_train_)
    df.drop(['elevation'], axis=1, inplace=True)
    df.drop(['furnished'], axis=1, inplace=True)


    
    for K in [8, 16, 24, 32, 64, 80, 96, 128, 192, 256]:
        df = add_KNN_feature(df, df_train_, K)
    
    #df['date_mean'] = df['rent_approval_date'].apply(lambda x: means[x])
    df['date_std'] = df['rent_approval_date'].apply(lambda x: stds[x])
    #df['date_median'] = df['rent_approval_date'].apply(lambda x: median[x])
 
    #df.drop(['latitude'], axis=1, inplace=True)
    #df.drop(['longitude'], axis=1, inplace=True)
    #df['rent_approval_date'] = pd.to_datetime(df['rent_approval_date']).astype('int64')
    df.drop(['rent_approval_date'], axis=1, inplace=True)
    #df.drop(['block'], axis=1, inplace=True)

    return df

# %%
def handle_outliers(group, attribute):
    z_scores = zscore(group[attribute])
    threshold = 3 
    outlier_indices = np.where(np.abs(z_scores) > threshold)[0]
    print(len(outlier_indices))
    return group[(z_scores < threshold)]  

# %%
df_train = pd.read_csv(path_prefix + 'train.csv')
df_test = pd.read_csv(path_prefix + 'test.csv')
df_test['monthly_rent'] = -1
#df_train['rent_approval_date'] = pd.to_datetime(df_train['rent_approval_date']).astype('int64')
#df_test['rent_approval_date'] = pd.to_datetime(df_test['rent_approval_date']).astype('int64')

# %%


# %%
df_train = df_train.groupby(df_train.columns.tolist()).mean().reset_index()
df_train

# %%
# handle outlier
grouped = df_train.groupby('rent_approval_date', group_keys=False)
df_train = grouped.apply(handle_outliers, attribute='monthly_rent')
df_train.reset_index(drop=True, inplace=True)

# %%
#add cluster feature
df_lat_long = df_train[['latitude', 'longitude']]
df_lat_long_test = df_test[['latitude', 'longitude']]
df_total = pd.concat([df_lat_long, df_lat_long_test], axis=0)
from sklearn.cluster import DBSCAN
DBSCAN_model = DBSCAN(eps=0.005, min_samples=1)
DBSCAN_model.fit(df_total)
#plot
sns.scatterplot(x='latitude', y='longitude', hue=DBSCAN_model.labels_, data=df_total)


# %%
df_train['cluster'] = DBSCAN_model.labels_[:len(df_train)]
df_test['cluster'] = DBSCAN_model.labels_[len(df_train):]

# %%
ds_train_processed = data_preprocess(df_train, df_train)
#ds_train_unmapped = data_preprocess(df_train, df_train, category_mapping=False)
#ds_test_processed = data_preprocess(df_test, df_train)
#ds_test_unmapped = data_preprocess(df_test, df_train, category_mapping=False)
ds_train_processed

# %%
df_norm = df_train.copy()
#df_norm['monthly_rent'] = np.log(df_norm['monthly_rent'])
means = df_norm.groupby('rent_approval_date')['monthly_rent'].mean()
stds = df_norm.groupby('rent_approval_date')['monthly_rent'].std()

# %%
X = ds_train_processed.drop(['monthly_rent'], axis=1)
cat_cols = ['town', 'flat_type', 'flat_type2', 'cluster', 'street_name', 'block',
              'lease_commence_date', 'flat_model', 'subzone', 'planning_area','region']
X_cat = X
X = X.drop(cat_cols, axis=1)
#X.drop(['rent_approval_date'], axis=1, inplace=True)
y = df_train['monthly_rent'] - means[df_train['rent_approval_date']].reset_index(drop=True)
#X = X[best_features]
#X_cat = ds_train_unmapped.drop(['monthly_rent'], axis=1)



#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#real_X_train, real_X_test, real_y_train, real_y_test = train_test_split(df_train, real_y, test_size=0.3, random_state=42)





# %%
for col in X.columns:
    if len(X[col].unique()) == 1:
        print(X[col].value_counts())

# %%
lgbm_best_params = {"n_estimators": 984, "num_leaves": 14, "min_child_samples": 20, "learning_rate": 0.00955415404855802, "log_max_bin": 9, "colsample_bytree": 0.1715233882278872, "reg_alpha": 0.014850297149652449, "reg_lambda": 133.998563158583}

xgb_best_params = {"n_estimators": 1916, "max_leaves": 41, "min_child_weight": 2.801525277874866, "learning_rate": 0.008298317608896174, "subsample": 0.6353412977381421, "colsample_bylevel": 0.41285744272632785, "colsample_bytree": 0.28604232010408714, "reg_alpha": 0.0011682284135139422, "reg_lambda": 5.118367390261997}

cat_best_params = {"early_stopping_rounds": 10, "learning_rate": 0.0357338948551828, "n_estimators": 230}

# %%
automl_settings_lightgbm = {
    "time_budget": config['time_limits'],
    "early_stop": config['early_stop'],
    "metric": 'rmse',
    "task": 'regression',
    "estimator_list": ['lgbm'],
    "eval_method": config['eval_method'],
    "log_file_name": 'lgbm_' + config['log_postfix'],
    "n_jobs": config['max_threads'],
    "n_concurrent_trials": config['parallel_trail'],

}

automl_settings_xgboost = {
    "time_budget": config['time_limits'],
    "early_stop": config['early_stop'],
    "metric": 'rmse',
    "task": 'regression',
    "estimator_list": ['xgb_limitdepth'],
    "eval_method": config['eval_method'],
    "log_file_name": 'xgboost_' + config['log_postfix'],
    "n_jobs": config['max_threads'],
    "n_concurrent_trials": config['parallel_trail'],

}

automl_settings_catboost = {
    "time_budget": config['time_limits'],
    "early_stop": config['early_stop'],
    "metric": 'rmse',
    "task": 'regression',
    "estimator_list": ['catboost'],
    "eval_method": config['eval_method'],
    "log_file_name": 'catboost_' + config['log_postfix'],
    "n_jobs": config['max_threads'],
    "n_concurrent_trials": config['parallel_trail'],

}

lightgbm_automl = AutoML()

catboost_automl = AutoML()

xgboost_automl = AutoML()

# %%
if 'lgbm' in config['use_model']:
    lightgbm_automl.fit(X_train=X, y_train=y, **automl_settings_lightgbm)#, starting_points=lgbm_best_params)

if 'catboost' in config['use_model']:
    catboost_automl.fit(X_train=X, y_train=y, **automl_settings_catboost)#, starting_points=cat_best_params)

if 'xgboost' in config['use_model']:
    xgboost_automl.fit(X_train=X, y_train=y, **automl_settings_xgboost)#, starting_points=xgb_best_params)

# %%
#plot importance of xgboost
feature_importance_xgb = xgboost_automl.model.estimator.feature_importances_ / xgboost_automl.model.estimator.feature_importances_.sum()
feature_importance_lgbm = lightgbm_automl.model.estimator.feature_importances_ / lightgbm_automl.model.estimator.feature_importances_.sum()
feature_names = X.columns
feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance_xgb': feature_importance_xgb, 'importance_lgbm': feature_importance_lgbm, 'importance_total': feature_importance_xgb + feature_importance_lgbm})
feature_importance_df.sort_values(by=['importance_lgbm'], ascending=False, inplace=True)
feature_importance_df.to_csv('feature_importance.csv', index=False)
feature_importance_df

# %%

df_test['monthly_rent'] = 0

ds_test_processed = data_preprocess(df_test, df_train)
ds_test_processed.drop(['monthly_rent'], axis=1, inplace=True)
X_test = ds_test_processed
X_test_cat = X_test
X_test = X_test.drop(cat_cols, axis=1)
#ds_test_processed.drop(['rent_approval_date'], axis=1, inplace=True)
if 'lgbm' in config['use_model']:
    y_pred_lightgbm = lightgbm_automl.predict(X_test)
    y_pred_lightgbm += means[df_test['rent_approval_date']].reset_index(drop=True)
if 'catboost' in config['use_model']:
    y_pred_catboost = catboost_automl.predict(X_test)
    y_pred_catboost += means[df_test['rent_approval_date']].reset_index(drop=True)
if 'xgboost' in config['use_model']:
    y_pred_xgboost = xgboost_automl.predict(X_test)
    y_pred_xgboost += means[df_test['rent_approval_date']].reset_index(drop=True)



# %%
submission = pd.read_csv(path_prefix + 'example-submission.csv')
if 'lgbm' in config['use_model']:
    submission['Predicted'] = y_pred_lightgbm
    submission.to_csv('submission_lightgbm.csv', index=False)
if 'catboost' in config['use_model']:
    submission['Predicted'] = y_pred_catboost
    submission.to_csv('submission_catboost.csv', index=False)
if 'xgboost' in config['use_model']:
    submission['Predicted'] = y_pred_xgboost
    submission.to_csv('submission_xgboost.csv', index=False)

y_mean = 0
y_square_mean = 0
y_harmonic_mean = 0
y_geometric_mean = 0

if 'lgbm' in config['use_model']:
    y_mean += y_pred_lightgbm * config['model_weights'][config['use_model'].index('lgbm')]
    y_square_mean += y_pred_lightgbm ** 2 * config['model_weights'][config['use_model'].index('lgbm')]
    y_harmonic_mean += 1 / y_pred_lightgbm * config['model_weights'][config['use_model'].index('lgbm')]
    y_geometric_mean *= y_pred_lightgbm ** config['model_weights'][config['use_model'].index('lgbm')]
if 'catboost' in config['use_model']:
    y_mean += y_pred_catboost * config['model_weights'][config['use_model'].index('catboost')]
    y_square_mean += y_pred_catboost ** 2 * config['model_weights'][config['use_model'].index('catboost')]
    y_harmonic_mean += 1 / y_pred_catboost * config['model_weights'][config['use_model'].index('catboost')]
    y_geometric_mean *= y_pred_catboost ** config['model_weights'][config['use_model'].index('catboost')]
if 'xgboost' in config['use_model']:
    y_mean += y_pred_xgboost * config['model_weights'][config['use_model'].index('xgboost')]
    y_square_mean += y_pred_xgboost ** 2 * config['model_weights'][config['use_model'].index('xgboost')]
    y_harmonic_mean += 1 / y_pred_xgboost * config['model_weights'][config['use_model'].index('xgboost')]
    y_geometric_mean *= y_pred_xgboost ** config['model_weights'][config['use_model'].index('xgboost')]

y_mean /= sum(config['model_weights'])
y_square_mean /= sum(config['model_weights'])
y_square_mean = np.sqrt(y_square_mean)
y_harmonic_mean = sum(config['model_weights']) / y_harmonic_mean
y_geometric_mean = y_geometric_mean ** (1 / sum(config['model_weights']))

submission['Predicted'] = y_mean
submission.to_csv('submission_mean.csv', index=False)
submission['Predicted'] = y_square_mean
submission.to_csv('submission_square_mean.csv', index=False)
submission['Predicted'] = y_harmonic_mean
submission.to_csv('submission_harmonic_mean.csv', index=False)
submission['Predicted'] = y_geometric_mean
submission.to_csv('submission_geometric_mean.csv', index=False)



