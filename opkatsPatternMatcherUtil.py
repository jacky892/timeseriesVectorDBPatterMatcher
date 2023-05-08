import pandas as pd
import sys
import numpy as np
import os,gzip
from matplotlib import pyplot as plt
import yfinance as yf
from typing import Optional, Tuple, Callable, Dict
from katslib.opkatsDataUtil import ma_normalized, get_feature_embedding_for_window, init_kat_index
SLIDING_WINDOW_SIZE=64

def get_period_block_by_end_date(ticker: str, end_date: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Get a pair of embeddings (numpy arrays) representing a historical period of a stock.

    Args:
        ticker (str): The stock ticker symbol.
        end_date (str, optional): The end date of the period to retrieve.
            If not specified, retrieves the most recent period. Default is None.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple of two numpy arrays representing the embeddings for the period.
            The first array is the embedding for the period, and the second array is the embedding for the following days.
    """
    stk = yf.Ticker(ticker)
    df=stk.history(period='7y')
    if end_date is None:
        df=ma_normalized(df, 250)
    else:
        df=ma_normalized(df.loc[:end_date], 250)
    new_df=df.iloc[-64:]
    new_df['_date']=new_df.index
    ik1=f'qa_{ticker}_{new_df._date[0].strftime("%Y%m%d")}_{new_df._date[-1].strftime("%Y%m%d")}'
    pair=get_feature_embedding_for_window(new_df, ik1)
    return ik1, new_df, pair[0], pair[1]


def get_perf_by_ticker_key(query_key: str, normalizer: Optional[Callable] = None) -> Optional[Dict[str, pd.DataFrame]]:
    """
    Retrieves historical stock data for a given ticker and returns a dictionary containing two dataframes:
    1) the performance of the stock during the specified period, normalized using the given `normalizer` function
    2) the next 10 days of stock performance

    Args:
        query_key: A string containing the ticker symbol and end date in the format 'TICKER_STARTDATE_ENDDATE'.
        normalizer: A function that takes a pandas DataFrame of stock data and returns a normalized DataFrame.
            If None, the default normalizer `ma_normalized` is used.

    Returns:
        A dictionary containing two pandas DataFrames:
        - 'period_df': a DataFrame containing the performance of the stock during the specified period
        - 'following_df': a DataFrame containing the next 10 days of stock performance

        If the requested stock data cannot be retrieved, returns None.
    """
    if normalizer is None:
        normalizer=ma_normalized
    datestr=query_key.split('_')[-1]
    ticker=query_key.split('_')[0]
    if ticker in ['dr', 'av', 'ri', 'qa']:        
        ticker=query_key.split('_')[1]
    print('ticker:',ticker)        
    stk = yf.Ticker(ticker)
    df=stk.history(period='7y')
    df=normalizer(df)
    ret_dict={}
    if df is None:
        return None
    ret_df=df.loc[:datestr].iloc[-64:]
    ret_dict['period_df']=ret_df    
    ret_dict['following_df']=ret_df.loc[datestr:].iloc[:10]
    return ret_dict
 
def prepare_items_for_graph(data):
    """
    Prepare items for graphing.

    Args:
        data (pandas.DataFrame): A pandas DataFrame containing the input data.

    Returns:
        list: A list of tuples containing the prepared data.

    """
    
    result_list = []    
    for _, row in data.iterrows():
        id = row['id']
        vec = row['values']
        ret_dict=get_perf_by_ticker_key(id)
        period_df=ret_dict['period_df']
        following_df=ret_dict['period_df']
        prices = period_df[['Open', 'Close']].values.tolist()
        flat_values = [item for sublist in prices for item in sublist]
        result_list.append((id, (vec, flat_values, following_df)))
    return result_list

def prepare_graph(data):
    """
    Prepare data and create a graph index.

    Args:
        data (pandas.DataFrame): A pandas DataFrame containing the input data.

    Returns:
        tuple: A tuple containing the prepared data and the graph index.

    """
    data_prepared = prepare_items_for_graph(data)
    graph_index = pd.Float64Index(np.arange(start=0, stop=SLIDING_WINDOW_SIZE, step=0.5))
    return data_prepared, graph_index

def plot_graphs(data_prepared, graph_index, query_item, outdir='plotjpg'):
    """
    Plots stock patterns and their normalized market values for a given query item and its top 10 similar items.

    Args:
    - data_prepared (list): a list of tuples containing the stock ID and its corresponding feature vectors.
    - graph_index (list): a list of integers representing the days in the time window.
    - query_item (str): the ID of the query item.
    - outdir (str): the output directory for saving the generated plot (default: 'plotjpg').

    Returns:
    - None

    Raises:
    - None
    """
    fig = plt.figure(figsize=(28,13))
    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2)
    ax3 = plt.subplot(1, 3, 3)

    _id_0, vectors_0 = data_prepared[0]
    graph = ax1.plot(graph_index, vectors_0[1], label=_id_0, marker='o' if _id_0 == query_item else None)

    for item in data_prepared[0:10]:
        _id, vectors = item
        vectors=list(vectors)
        
        if not  len(vectors[1])==len(graph_index):
            continue
        graph1 = ax2.plot(graph_index, vectors[1], label=_id, marker='o' if _id == query_item else None)
        graph2 = ax3.plot(graph_index[:10], vectors[2]['Close'][:10], label=_id, marker='o' if _id == query_item else None)

    ax1.set_xlabel("Days in time window")
    ax2.set_xlabel("Days in time window")
    ax1.set_ylabel("Stock values")
    ax2.set_ylabel("Normalized Stock Values")
    ax1.title.set_text(f'source stock patterns and their normalized market values {_id_0}')
    ax2.title.set_text(f'Similar stock patterns and their normalized market values')
    ax2.title.set_text(f'')
#    plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
    plt.legend(loc='lower center', ncol=4, fontsize='xx-large', bbox_to_anchor=[-0.6, -0.15])
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    ofname=f'{outdir}/{query_item}.jpg'
    plt.savefig(ofname)
    if not sys.stdout.isatty():
        plt.show()
    return ofname

def show_query_results2( query_item, data, outdir='plotjpg'):
    data_prepared, graph_index = prepare_graph(data)
    ofname=plot_graphs( data_prepared, graph_index, query_item, outdir)
    return ofname
    
def filter_results(query_item, data, historical_only=False):
    '''
    Filters the input DataFrame by removing rows that have already been included based on a given query item, and optionally includes only data prior to the query interval.

    Args:
        query_item (str): The query item to filter the data by.
        data (pandas.DataFrame): The DataFrame to filter.
        historical_only (bool, optional): Whether to only include data prior to the query interval. Defaults to False.

    Returns:
        pandas.DataFrame: The filtered DataFrame, with duplicates removed and optionally with only historical data included.

    Raises:
        None
    '''
    already_present = []
    
    # Remove symbol that is already included
    for i, row in data.iterrows():
        final_date_idx=-1
        if len(row.id.split('_'))==3:
            check_name = row.id.split('_')[0]            
        else:
            check_name = row.id.split('_')[1]
            
        if check_name not in already_present:
            already_present.append(check_name)
        else:
            data.drop(i,axis=0,inplace=True)
            
    # Include only data prior to query interval
    if historical_only:
        import re
        _type, _, start_dt, end_dt = query_item.split('_')
        start_dt = pd.to_datetime(start_dt).date()
        try:
            data['final_date'] = data.id.apply(lambda x: re.sub('av_|br_|ri_', '', x).split('_')[final_date_idx])
        except:
            print('except data id is ',data.id)
        data['final_date'] =  data.final_date.apply(lambda x: pd.to_datetime(x).date())
        data = data[data.final_date <= start_dt]
        del data['final_date']
       
    return data

def fetch_vector_by_tag(items_to_query):
    # Fetch vectors from the index
    fetch_res = kats_index.fetch(ids=items_to_query)
    #print(fetch_res)
    # Create a list of ids and vectors for the fetched items
    query_ids = [res.id for res in fetch_res.vectors.values()]
    query_vectors = [res.values for res in fetch_res.vectors.values()]
    return query_vectors
        
class opkatsPatternMatcherUtil:
    @staticmethod 
    def match_period_with_vectordb(ticker, end_date=None, kat_index=None):

        ik1, src_df, query_item, query_vectors=get_period_block_by_end_date(ticker, end_date)
        if kat_index is None:
            kats_index=init_kat_index()
#        print('id:',ik1,  src_df.shape, src_df.tail())
        # actually search from index
        query_vectors=[query_vectors]
        query_results = []
        for xq in query_vectors:
            q_res = kats_index.query(xq, top_k=20, include_values=True)
            print('query_results len is ',len(q_res['matches']))    
            prefilter_res_df = pd.DataFrame(
                {
                    'id': [res.id for res in q_res.matches], 
                    'score': [res.score for res in q_res.matches],
                    'values': [res.values for res in q_res.matches]
                 }
            )
            prices =src_df[['Open', 'Close']].values.tolist()
            flat_values = [item for sublist in prices for item in sublist]
            #prefilter_res_df.append({'id':ik1, 'score':1, 'values':flat_values}, ignore_index=True)


            res_df = filter_results(query_item, prefilter_res_df, historical_only=True).copy()
            
#            res_df.append({'id':ik1, 'score':1, 'values':flat_values}, ignore_index=True)
            res_df.iloc[-1]=pd.Series({'id':ik1, 'score':1, 'values':query_vectors})
            res_df.sort_values(by='score', ascending=False, inplace=True)
            ofname=show_query_results2(query_item, res_df.iloc[0:10])        
        return ofname

if __name__=='__main__':
    opkatsPatternMatcherUtil.match_period_with_vectordb('C')
