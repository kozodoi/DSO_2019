###############################
#                             #
#         1. MODELING         #
#                             #
###############################

###############################
#                             #
#      SUBMIT TO QSCORE       #
#                             #
###############################

import io, math, requests
def submit_prediction(df, sep=',', comment='', compression='gzip', **kwargs):
    TOKEN='e14ac5c20102ef4dc10712ec7838b44f48e227604d6a931ab92804ce785527ac43465fdfbc5bd8269a92cf32fffd71fed365d3dfd95e029ac43c14e25cc5c788'
    URL='https://qscore.datascience-olympics.com/api/submissions'
    df.to_csv('temporary.dat', sep=sep, compression=compression, **kwargs)
    r = requests.post(URL, headers={'Authorization': 'Bearer {}'.format(TOKEN)},files={'datafile': open('temporary.dat', 'rb')},data={'comment':comment, 'compression': compression})
    if r.status_code == 429:
        raise Exception('Submissions are too close. Next submission is only allowed in {} seconds.'.format(int(math.ceil(int(r.headers['x-rate-limit-remaining']) / 1000.0))))
    if r.status_code != 200:
        raise Exception(r.text)
        
        
        
################################
#                              #
#    TEST-TIME AUGMENTATION    #
#                              #
################################

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


      
    
    
################################
#                              #
#      2. DATA PREPARATION     #
#                              #
################################
    
###############################
#                             #
#        ENCODE FACTORS       #
#                             #
###############################

# performs dummy or label encoding

def encode_factors(df, method = "label", skip = None):
    
    # select columns 
    factors = [f for f in df.columns if df[f].dtype == "object"]
    factors = [f for f in factors if f not in skip]
    
    # label encoding
    if method == "label":
        for var in factors:
            df[var], _ = pd.factorize(df[var])
        
    # dummy encoding
    if method == "dummy":
        df = pd.get_dummies(df, drop_first = True, columns = factors)
    
    # dataset
    return df



###############################
#                             #
#        COUNT MISSINGS       #
#                             #
###############################

# computes missings per variable (count, %)
# displays variables with most missings

import pandas as pd
def count_missings(data):
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum() / data.isnull().count() * 100).sort_values(ascending = False)
    table = pd.concat([total, percent], axis = 1, keys = ["Total", "Percent"])
    table = table[table["Total"] > 0]
    return table



###############################
#                             #
#        AGGRGEATE DATA       #
#                             #
###############################

# aggregates numeric data using specified stats
# aggregates factors using mode and nunique
# returns df with generated features

import scipy.stats
def aggregate_data(df, group_var, num_stats = ['mean', 'sum'], factors = None, var_label = None, sd_zeros = False):
    
    
    ### SEPARATE FEATURES
  
    # display info
    print("- Preparing the dataset...")

    # find factors
    if factors == None:
        df_factors = [f for f in df.columns if df[f].dtype == "object"]
    else:
        df_factors = factors
        df_factors.append(group_var)
        
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
        num_df = num_df.groupby([group_var]).agg(num_stats)
        num_df.columns = ["_".join(col).strip() for col in num_df.columns.values]
        num_df = num_df.sort_index()

    # aggregate factors
    if (num_facs > 0):
        print("- Aggregating factor features...")
        fac_int_df = fac_df.copy()
        for var in factors:
            fac_int_df[var], _ = pd.factorize(fac_int_df[var])
        fac_int_df[group_var] = fac_df[group_var]
        fac_df = fac_int_df.groupby([group_var]).agg([('count'), ('mode', lambda x: pd.Series.mode(x)[0])])
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
    if (var_label != None):
        agg_df.columns = [var_label + "_" + str(col) for col in agg_df.columns]
    
    # impute zeros for SD
    if (sd_zeros == True):
        stdevs = agg_df.filter(like = "_std").columns
        for var in stdevs:
            agg_df[var].fillna(0, inplace = True)
            
    # dataset
    agg_df = agg_df.reset_index()
    print("- Final dimensions:", agg_df.shape)
    return agg_df



###############################
#                             #
#      ADD DATE FEATURES      #
#                             #
###############################

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
        


###############################
#                             #
#      ADD TEXT FEATURES      #
#                             #
###############################

# extract basic features from strings
# appends new features to the data frame

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
def add_text_features(data, strings, k = 5, keep = True):

    ##### PROCESSING LOOP
    for var in strings:

        ### TEXT PREPROCESSING

        # replace NaN with empty string
        data[var][pd.isnull(data[var])] = ''

        # remove common words
        freq = pd.Series(' '.join(data[var]).split()).value_counts()[:10]
        #freq = list(freq.index)
        #data[var] = data[var].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
        #data[var].head()

        # remove rare words
        freq = pd.Series(' '.join(data[var]).split()).value_counts()[-10:]
        #freq = list(freq.index)
        #data[var] = data[var].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
        #data[var].head()

        # convert to lowercase 
        data[var] = data[var].apply(lambda x: " ".join(x.lower() for x in x.split())) 

        # remove punctuation
        data[var] = data[var].str.replace('[^\w\s]','')         


        ### COMPUTE BASIC FEATURES

        # word count
        data[var + '_word_count'] = data[var].apply(lambda x: len(str(x).split(" ")))
        data[var + '_word_count'][data[var] == ''] = 0

        # character count
        data[var + '_char_count'] = data[var].str.len().fillna(0).astype('int64')


        ##### COMPUTE TF-IDF FEATURES

        # import vectorizer
        tfidf  = TfidfVectorizer(max_features = k, 
                                 lowercase    = True, 
                                 norm         = 'l2', 
                                 analyzer     = 'word', 
                                 stop_words   = 'english', 
                                 ngram_range  = (1, 1))

        # compute TF-IDF
        vals = tfidf.fit_transform(data[var])
        vals = pd.SparseDataFrame(vals)
        vals.columns = [var + '_tfidf_' + str(p) for p in vals.columns]
        data = pd.concat([data, vals], axis = 1)


        ### CORRECTIONS

        # remove raw text
        if keep == False:
            del data[var]

        # print dimensions
        #print(data.shape)
        
    # return data
    return data