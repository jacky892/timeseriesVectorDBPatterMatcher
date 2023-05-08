#%%writefile opkatsDataUtil.py
from sklearn.preprocessing import MinMaxScaler
from kats.consts import TimeSeriesData
from kats.tsfeatures.tsfeatures import TsFeatures
from decimal import Decimal
import pickle, gzip
import os
import pandas as pd
import numpy as np
import itertools
import configparser

def get_token_input(section='pinecone', value='api_token', cfg_fname=r"configfile.ini"):
    token=input('please input pinecone api token from pinecone.io')
    ### regex to check pinecone token
    import re

    pattern = r'^[a-f\d]{8}-[a-f\d]{4}-[a-f\d]{4}-[a-f\d]{4}-[a-f\d]{12}$'
    if not re.match(pattern, token):
        print('error in token')
        return None

    config = configparser.ConfigParser()
    # Add the structure to the file we will create
    config.add_section('pinecone')
    config.set('pinecone', 'api_token', token)

    pattern = r'^[a-f\d]{2}-[a-f\d]{*}-[\d]{1}-[a-f]{*}$'
    env=input('please input pinecone env from pinecone.io (e.g. us-east-1-aws)')

    config = configparser.ConfigParser()
    # Add the structure to the file we will create
    config.add_section('pinecone')
    config.set('pinecone', 'api_token', token)
    config.set('pinecone', 'env', env)
    # Write the new structure to the new file
    with open(cfg_fname, 'w') as configfile:
        config.write(configfile)
    return env, token

def get_pinecone_prop(section='pinecone', value='api_token', cfg_fname="configfile.ini"):
    config = configparser.ConfigParser()
    config.read(cfg_fname)
    try:
        tok=config['pinecone']['api_token']
        env=config['pinecone']['env']
    except:
        env,tok=get_token_input(section, value, cfg_fname)
    return env,tok

def windows(data, window_size, step):
    r = np.arange(len(data))
    s = r[::step]
    z = list(zip(s, s + window_size))
    f = '{0[0]}:{0[1]}'.format
    g = lambda t: data.iloc[t[0]:t[1]]
    return pd.concat(map(g, z), keys=map(f, z))


def stock_df_oversampler(df, ticker, up_threshold=0.07, down_threshold=0.05, forward_days=10,cnt=5, winsize=64, min_step=10):
    '''
The stock_df_oversampler function takes a Pandas DataFrame df representing stock prices and performs oversampling on the data by adding future return information and resampling the data for periods with returns exceeding specified thresholds.

Parameters:

df: A Pandas DataFrame containing stock price data.
ticker: A string representing the stock ticker symbol for which the oversampling is being performed.
up_threshold: A float representing the upper threshold for a period's future return to be considered a "big rise". Default is 0.07.
down_threshold: A float representing the lower threshold for a period's future return to be considered a "big drop". Default is 0.05.
forward_days: An integer representing the number of days in the future for which return information is added to the data. Default is 10.
cnt: An integer representing the number of periods to oversample for each "big rise" or "big drop". Default is 5.
winsize: An integer representing the window size for resampling the data. Default is 64.
min_step: An integer representing the minimum step size between windows when resampling the data. Default is 10.
Returns:

A dictionary containing three sub-dictionaries: 'bigrise', 'bigdrop', and 'avg'. Each sub-dictionary contains keys representing the oversampled periods and values representing the corresponding resampled data.
The 'bigrise' sub-dictionary contains periods where the future return exceeds the up_threshold and the 'bigdrop' sub-dictionary contains periods where the future return is less than the negative of down_threshold. The 'avg' sub-dictionary contains periods where the future return is within the specified threshold values.

The function first adds a new column to the DataFrame representing the future return information for each period. It then resamples the data into windows of winsize size with a minimum step size of min_step. For each window, the function determines if the future return information meets the specified thresholds and adds the corresponding data to the appropriate sub-dictionary. If the future return information is within the threshold values, the data is added to the 'avg' sub-dictionary only if the number of samples in the 'bigrise' and 'bigdrop' sub-dictionaries is less than cnt.

The function returns the oversampled data as a dictionary with the oversampled periods as keys and the resampled data as values.
    '''
    from collections import defaultdict
    #ret_dict=defaultdict(default_factory=dict)
    ret_dict={}
    ret_dict['bigrise']={}
    ret_dict['bigdrop']={}    
    ret_dict['avg']={}    
    df['_fw_ret']=df.Close.pct_change(forward_days).shift(-forward_days)
    df['_date']=df.index
    wdf = windows(df, winsize, min_step)
    
    for window, new_df in wdf.groupby(level=0):
        #print(window, new_df)
        new_df.dropna(subset=['Open', 'Close', 'Volume' ], inplace=True)
        if new_df.shape[0] == winsize:                
            
            k=f'{ticker}_{new_df._date[0].strftime("%Y%m%d")}_{new_df._date[-1].strftime("%Y%m%d")}'
            #print('key is ',k)
            if new_df.iloc[-1]['_fw_ret']>up_threshold:
                #print('add plus', new_df.tail())
                ret_dict['bigrise'][f'dr_{k}']=new_df
            elif new_df.iloc[-1]['_fw_ret']<-down_threshold:
                #print('add minus', new_df.tail())
                ret_dict['bigdrop'][f'ri_{k}']=new_df                
            else:
                #print('add avg')
                good_sample_size=len(ret_dict['bigdrop'])+len(ret_dict['bigdrop'])
                if len(ret_dict['avg'])<good_sample_size:
                    ret_dict['avg'][f'av_{k}']=new_df                                
        else:
            #print('skip', ticker, new_df.shape) 
            _=None
    return ret_dict

def get_stock_sample_pattern(ticker='MSFT', normalizer=None):
    if normalizer is None:
        normalizer=ma_normalized
    import yfinance as yf
    stock = yf.Ticker(ticker)
    try:
        df=stock.history(period='5y')
        df=ma_normalized(df)
        ret_dict=stock_df_oversampler(df, ticker=ticker)
    except Exception as e:
        import traceback
        print('exception:', e)
        traceback.print_exc()
        print('error sampling stock period', e)
        ret_dict={}
    return ret_dict

def init_kat_index():
    import pinecone
    env,pk=get_pinecone_prop()
    pinecone.init(api_key=pk, environment=env)
    #
    kats_index_name = 'stocks-trends-with-features'

    # Check whether the index with the same name already exists
    #if kats_index_name in pinecone.list_indexes():
    #    pinecone.delete_index(kats_index_name)
    #pinecone.create_index(name=kats_index_name, dimension=40)
    kats_index = pinecone.Index(index_name=kats_index_name)
    return kats_index

def ma_normalized(df, madays=250):
    df['vol_longma']=df.Volume.rolling(madays).mean()
    df['close_longma']=df.Close.rolling(madays).mean()
    df['Open']=df.eval('Open/close_longma')
    df['Close']=df.eval('Close/close_longma')
    df['Volume']=df.eval('Volume/vol_longma')
    df.dropna(inplace=True)
    df.ffill(inplace=True)
    return df

def min_max_normalized(df):
    from sklearn.preprocessing import MinMaxScaler
    scaler=MinMaxScaler()
    df[['Open', 'Close', 'Volume']] = scaler.fit_transform(df[['Open', 'Close', 'Volume']])
    return df

def preprocess_time_series_data(df, input_key):
    '''
    Preprocess the time series data.

    Args:
        df (pandas.DataFrame): A pandas DataFrame containing the input data.
        input_key (str): The name of the time series.

    Returns:
        pandas.DataFrame: A processed pandas DataFrame.

    '''
    ts_name = input_key
    prices = df[['Open', 'Close']].values.tolist()
    if isinstance(df.index, pd.MultiIndex):
        df['_date']=df.index.get_level_values(1)
    else:
        df['_date']=df.index
    flat_values = [item for sublist in prices for item in sublist]
    df = df.rename(columns={"_date":"time"})
    ts_df = pd.DataFrame({'time':df.time.repeat(2),
                          'price':flat_values})
    ts_df.drop_duplicates(keep='first', inplace=True)
    return ts_name, ts_df


def get_feature_embedding_for_window(df, input_key):
    '''
    Extract features for the time window.

    Args:
        df (pandas.DataFrame): A pandas DataFrame containing the input data.
        input_key (str): The name of the time series.

    Returns:
        tuple: A tuple containing the name of the time series and its feature list, or None if the feature list contains infinity.

    '''
    from decimal import Decimal
    from IPython.display import clear_output

    ts_name, ts_df = preprocess_time_series_data(df, input_key)

    # Use Kats to extract features for the time window
    try:
        if not (len(np.unique(ts_df.price.tolist())) == 1 \
            or len(np.unique(ts_df.price.tolist())) == 0):
            timeseries = TimeSeriesData(ts_df)
            features = TsFeatures().transform(timeseries)
            feature_list = [float(v) if not pd.isnull(v) else float(0) for _, v in features.items()]
            if Decimal('Infinity') in feature_list or Decimal('-Infinity') in feature_list:
                print('feature list has infinity')
                return None
            return (ts_name, feature_list)
    except np.linalg.LinAlgError as e:
        print(f"Can't process {ts_name}:{e}")
    return None

def chunks(iterable, batch_size=100):
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))
    
def gen_all_data_pack(stock_list=None, output_datapack_fname='period_stock_pack.pkl.gz'):
    if stock_list is None:
        mdf=pd.read_csv('symbols_valid_meta.csv')
        stock_list=mdf.query('ETF=="N"').Symbol
    ret_dict={'bigrise':{}, 'bigdrop':{}, 'avg':{}}
    for ticker in stock_list:
        print(ticker)    
        _=get_stock_sample_pattern(ticker)
        for k in ret_dict:
            if k in _.keys():
                ret_dict[k].update(_[k])
    with gzip.open(output_datapack_fname, 'wb') as f:
        pickle.dump(ret_dict, f)
    return ret_dict

def gen_kats_embedding_from_datapack(period_pack_dict=None, in_pack_fname='period_stock_pack.pkl.gz', out_vector_fname='pinecone_kats_upload_list.pkl.gz'):
    import sys
    import pandas as pd
    save_stdout=sys.stdout
    save_stderr=sys.stderr
    if period_pack_dict is None:
        with gzip.open(in_pack_fname, 'rb') as f:
            period_pack_dict=pickle.load(f)
    items_to_upload=[]
    from kats.consts import TimeSeriesData
    from kats.tsfeatures.tsfeatures import TsFeatures
    import warnings
    warnings.filterwarnings("ignore")
    cnt=0
    for label_type in period_pack_dict.keys():
        for ik1 in iter(period_pack_dict[label_type]):
            cnt=cnt+1
            print(ik1)
            d=period_pack_dict[label_type][ik1]
            d.ffill(inplace=True)

            pair=get_feature_embedding_for_window(d, ik1)
            items_to_upload.append(pair)
            
    with gzip.open(out_vector_fname, 'wb') as f:
        pickle.dump(items_to_upload, f)
    return items_to_upload


def init_kat_index(reset_index_db=False):
    import pinecone
    pk,env='688f90ee-f916-4a3d-a6ae-f19d66ada9e2','us-east-1-aws'
    pinecone.init(api_key=pk, environment=env)
    #
    kats_index_name = 'stocks-trends-with-features'

    if reset_index_db:
        # Check whether the index with the same name already exists
        if kats_index_name in pinecone.list_indexes():
            pinecone.delete_index(kats_index_name)
        pinecone.create_index(name=kats_index_name, dimension=40)
    kats_index = pinecone.Index(index_name=kats_index_name)
    return kats_index

def chunks(iterable, batch_size=100):
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))
        
def upload_kats_vector_to_pinecone(kats_index, items_to_upload):
    for batch in chunks(items_to_upload, 1000):
        k,df=batch[0]
        kats_index.upsert(vectors=batch)
        print('now upload batch is ',k)
    return kats_index


def test_gen_data():
    period_dict_pack=gen_all_data_pack(stock_list=['A', 'MSFT', 'QQQ', 'GOOG', 'AMZN', 'ARKK'])
    items_to_uploadx=gen_kats_embedding_from_datapack(period_dict_pack)

def gen_all_data(force_update=False):

    datapack_fname='_period_stock_pack.pkl.gz'
    tlist=['A', 'MSFT', 'QQQ', 'GOOG', 'AMZN', 'ARKK']
#    tlist=None
    if not os.path.exists(datapack_fname) or force_update:
        period_dict_pack=gen_all_data_pack(stock_list=tlist, output_datapack_fname=datapack_fname)
    else:
        with gzip.open(datapack_fname, 'rb') as f:
            period_dict_pack=pickle.load(f)

    upload_list_fname='_pinecone_kats_upload_list.pkl.gz'
    if not os.path.exists(upload_list_fname) or force_update:
        items_to_upload=gen_kats_embedding_from_datapack(period_dict_pack, out_vector_fname=upload_list_fname)
    else:
        with gzip.open(upload_list_fname, 'rb') as f:
            items_to_upload=pickle.load(f)
    return items_to_upload

def upload_data(items_to_upload=None, upload_list_fname='_pinecone_kats_upload_list.pkl.gz'):
    kats_index=init_kat_index(reset_index_db=True)
    kats_index.describe_index_stats()

    if items_to_upload is None:
        items_to_uploadx=gen_kats_embedding_from_datapack(period_dict_pack, out_vector_fname=upload_list_fname)
    else:
        upload_kats_vector_to_pinecone(kats_index, items_to_upload)


if __name__=='__main__':
    items_to_upload=gen_all_data()
    kats_index=init_kat_index(reset_index_db=False)
    print(kats_index.describe_index_stats())
    ret=upload_kats_vector_to_pinecone(kats_index, items_to_upload)
