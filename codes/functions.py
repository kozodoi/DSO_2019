##########
##########
########## SUBMIT TO QSCORE


import io, math, requests
def submit_prediction(df, sep=',', comment='', **kwargs):
    TOKEN='9442bece1dab6c68fbe2ba3c3e798ceb6b26d920b1bc17655690073413aa85d47d2a46fe393e013a86a5a866c34d495f5df86c690f24db12a83b17f229df6121'
    URL='https://qscore.datascience-olympics.com/api/submissions'
    buffer = io.StringIO()
    df.to_csv(buffer, sep=sep, **kwargs)
    buffer.seek(0)
    r = requests.post(URL, headers={'Authorization': 'Bearer {}'.format(TOKEN)},files={'datafile': buffer},data={'comment':comment})
    if r.status_code == 429:
        raise Exception('Submissions are too close. Next submission is only allowed in {} seconds.'.format(int(math.ceil(int(r.headers['x-rate-limit-remaining']) / 1000.0))))
    if r.status_code != 200:
        raise Exception(r.text)



##########
##########
########## ENCODING FACTORS

# performs dummy or label encoding

def encode_factors(df, method = "label"):
    
    # label encoding
    if method == "label":
        factors = [f for f in df.columns if df[f].dtype == "object"]
        for var in factors:
            df[var], _ = pd.factorize(df[var])
        
    # dummy encoding
    if method == "dummy":
        df = pd.get_dummies(df, drop_first = True)
    
    # dataset
    return df


##########
##########
########## AGGREGATING DATA

# aggregates numeric data using specified stats
# aggregates factors using mode and nunique
# returns df with generated features

def aggregate_data(df, group_var, num_stats = ['mean', 'sum'], 
                   label = None, sd_zeros = False):
    
    
    ### SEPARATE FEATURES
  
    # display info
    print("- Preparing the dataset...")

    # find factors
    df_factors = [f for f in df.columns if df[f].dtype == "object"]
    df_factors = ['fullVisitorId', 'device_operatingSystem', 'geoNetwork_country', 'channelGrouping']
        
    # partition subsets
    if type(group_var) == str:
        num_df = df[[group_var] + list(set(df.columns) - set(df_factors))]
        fac_df = df[df_factors]
    else:
        num_df = df[group_var + list(set(df.columns) - set(df_factors))]
        fac_df = df[df_factors]      
    
    # display info
    num_facs = fac_df.shape[1] - 1
    num_nums = num_df.shape[1] - 1
    print("- Extracted %.0f factors and %.0f numerics..." % (num_facs, num_nums))


    ##### AGGREGATION
 
    # aggregate numerics
    if (num_nums > 0):
        print("- Aggregating numeric features...")
        if type(group_var) == str:
            num_df = num_df.groupby([group_var]).agg(num_stats)
            num_df.columns = ["_".join(col).strip() for col in num_df.columns.values]
            num_df = num_df.sort_index()
        else:
            num_df = num_df.groupby(group_var).agg(num_stats)
            num_df.columns = ["_".join(col).strip() for col in num_df.columns.values]
            num_df = num_df.sort_index()

    # aggregate factors
    if (num_facs > 0):
        print("- Aggregating factor features...")
        if type(group_var) == str:
            fac_df = fac_df.groupby([group_var]).agg([("mode", lambda x: scipy.stats.mode(x)[0][0])])
            fac_df.columns = ["_".join(col).strip() for col in fac_df.columns.values]
            fac_df = fac_df.sort_index()
        else:
            fac_df = fac_df.groupby(group_var).agg([("mode", lambda x: scipy.stats.mode(x)[0][0])])
            fac_df.columns = ["_".join(col).strip() for col in fac_df.columns.values]
            fac_df = fac_df.sort_index()


    ##### MERGER

    # merge numerics and factors
    if ((num_facs > 0) & (num_nums > 0)):
        agg_df = pd.concat([num_df, fac_df], axis = 1)
    
    # use factors only
    if ((num_facs > 0) & (num_nums == 0)):
        agg_df = fac_df
        
    # use numerics only
    if ((num_facs == 0) & (num_nums > 0)):
        agg_df = num_df
        

    ##### LAST STEPS

    # update labels
    if (label != None):
        agg_df.columns = [label + "_" + str(col) for col in agg_df.columns]
    
    # impute zeros for SD
    if (sd_zeros == True):
        stdevs = agg_df.filter(like = "_std").columns
        for var in stdevs:
            agg_df[var].fillna(0, inplace = True)
            
    # dataset
    agg_df = agg_df.reset_index()
    print("- Final dimensions:", agg_df.shape)
    return agg_df



##########
##########
########## CREATING FEATURES FROM DATES

# creates a set of date-related features
# outputs df with generated features

def add_datepart(df, fldname, drop=True, time=False):
    "Helper function that adds columns relevant to a date."
    
    fld = df[fldname]
    fld_dtype = fld.dtype
    
    if isinstance(fld_dtype, pd.core.dtypes.dtypes.DatetimeTZDtype):
        fld_dtype = np.datetime64

    if not np.issubdtype(fld_dtype, np.datetime64):
        df[fldname] = fld = pd.to_datetime(fld, infer_datetime_format=True)
        
    targ_pre = re.sub('[Dd]ate$', '', fldname)
    attr = ['Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
            'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start', 'Is_year_end', 'Is_year_start']
    
    if time: attr = attr + ['Hour', 'Minute', 'Second']
        
    for n in attr: df[targ_pre + n] = getattr(fld.dt, n.lower())
    df[targ_pre + 'Elapsed'] = fld.astype(np.int64) // 10 ** 9
    
    if drop: df.drop(fldname, axis=1, inplace=True)
        
        
        
##########
##########
########## COUNTING MISSINGS

# computes missings per variable (count, %)
# displays variables with most missings

import pandas as pd
def count_missings(data):
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum() / data.isnull().count() * 100).sort_values(ascending = False)
    table = pd.concat([total, percent], axis = 1, keys = ["Total", "Percent"])
    table = table[table["Total"] > 0]
    return table



##########
##########
########## TEST TIME AUGMENTATION

# creates multiple versionf of test data (with noise)
# averages predictions over the created samples

def predict_proba_with_tta(X_test, model, num_iteration, alpha = 0.01, n = 4, seed = 0):

    # set random seed
    np.random.seed(seed = seed)

    # original prediction
    preds = model.predict_proba(X_test, num_iteration = num_iteration)[:, 1] / (n + 1)

    # select numeric features
    num_vars = [var for var in X_test.columns if X_test[var].dtype != "object"]

    # synthetic predictions
    for i in range(n):

        # copy data
        X_new = X_test.copy()

        # introduce noise
        for var in num_vars:
            X_new[var] = X_new[var] + alpha * np.random.normal(0, 1, size = len(X_new)) * X_new[var].std()

        # predict probss
        preds_new = model.predict_proba(X_new, num_iteration = num_iteration)[:, 1]
        preds += preds_new / (n + 1)

    # return probs
    return preds



##########
##########
########## MEAN TARGET ENCODING

# replaces factors with mean target values per value
# training data: encoding using internal CV
# validation and test data: encoding using  training data

from sklearn.model_selection import StratifiedKFold
def mean_target_encoding(train, valid, test, features, target, folds = 5):
    from sklearn.model_selection import StratifiedKFold
    ##### TRAINING

    # cross-validation
    skf = StratifiedKFold(n_splits = folds, random_state = 777, shuffle = True)
    for n_fold, (trn_idx, val_idx) in enumerate(skf.split(train, train[target])):

        # partition folds
        trn_x, trn_y = train.iloc[trn_idx], y.iloc[trn_idx]
        val_x, val_y = train.iloc[val_idx], y.iloc[val_idx]

        # loop for facrtors
        for var in features:

            # feature name
            name = "_".join(["mean_target_per", str(var)])

            # compute means
            means = val_x[var].map(trn_x.groupby(var)[target].mean())
            val_x[name] = means

            # impute means
            if n_fold == 0:
                train[name] = np.nan
                train.iloc[val_idx] = val_x
            else:
                train.iloc[val_idx] = val_x


    ##### VALIDATION

    # loop for factors
    for var in features:
        means = valid[var].map(train.groupby(var)[target].mean())
        valid[name] = means


    ##### TEST

    # copy data
    tmp_test = test.copy()

    # loop for factors
    for var in features:
        means = tmp_test[var].map(train.groupby(var)[target].mean())
        tmp_test[name] = means


    ##### CORRECTIONS

    # remove target
    del train[target], valid[target]

    # remove factors
    for var in features:
        del train[var], valid[var], tmp_test[var]

    # return data
    return train, valid, tmp_test