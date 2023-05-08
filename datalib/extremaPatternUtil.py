
#from debuglib.debugUtil import _debugl
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
#from datalib.taxonomyUtil import read_tlist
from datalib.extremaPatternLooper import extremaPatternLooper
from collections import defaultdict
from datalib.commonUtil import commonUtil as cu
from scipy.signal import argrelextrema
#from sig_matrix.matrixTraderUtil import tradeDateBackTester, signalTradeReviewer
from vcplib.vcpUtil import resample_df
def get_max_min_prices_dateidx(prices, smoothing=3, window_range=10, order=3):

    local_max = argrelextrema(prices.High.values, np.greater, order=order)[0]
#    local_max = argrelextrema(prices.High.values, np.greater_equal, order=order, mode='clip')[0]    
#    less_equal
    local_min = argrelextrema(prices.Low.values, np.less, order=order)[0]
#    local_min = argrelextrema(prices.Low.values,  np.less_equal, order=order, mode='clip')[0]    
    price_local_max_dt = []
    for i in local_max:
        if (i>window_range) and (i<len(prices)-window_range):
            price_local_max_dt.append(prices.iloc[i-window_range:i+window_range]['High'].idxmax())
    price_local_min_dt = []
    for i in local_min:
        if (i>window_range) and (i<len(prices)-window_range):
            price_local_min_dt.append(prices.iloc[i-window_range:i+window_range]['Low'].idxmin())  
    
    return min_max_from_local_extrema(price_local_max_dt, price_local_min_dt, prices)
    
def get_max_min_dateidx(prices, colname='Close', smoothing=3, window_range=10, order=3):
    smooth_prices = prices[colname].rolling(window=smoothing).mean().dropna()
#    local_max = argrelextrema(smooth_prices.values, np.greater, order=order)[0]
    local_max = argrelextrema(smooth_prices.values, np.greater_equal, order=order, mode='clip')[0]    
#    less_equal
#    local_min = argrelextrema(smooth_prices.values, np.less, order=order)[0]
    local_min = argrelextrema(smooth_prices.values, np.less_equal, order=order, mode='clip')[0]    
    price_local_max_dt = []
    for i in local_max:
        if (i>window_range) and (i<len(prices)-window_range):
            price_local_max_dt.append(prices.iloc[i-window_range:i+window_range][colname].idxmax())
    price_local_min_dt = []
    for i in local_min:
        if (i>window_range) and (i<len(prices)-window_range):
            price_local_min_dt.append(prices.iloc[i-window_range:i+window_range][colname].idxmin())  

    return min_max_from_local_extrema(price_local_max_dt, price_local_min_dt, prices)
    
def min_max_from_local_extrema(price_local_max_dt, price_local_min_dt, prices):
    #price_local_max_dt
#    price_local_min_dt

    maxima = pd.DataFrame(prices.loc[price_local_max_dt])
    minima = pd.DataFrame(prices.loc[price_local_min_dt])
        
    max_min = pd.concat([maxima, minima]).sort_index()
    max_min = max_min[~max_min.index.duplicated()]    
    p = prices
    p[p.index.isin(max_min.index)].index.values
#    max_min['day_num'] = max_min.index
    
    max_min['minmax_type'] = 'na'
    max_min.loc[maxima.index,'minmax_type'] ='max'
    max_min.loc[minima.index,'minmax_type'] ='min'    
#    max_only=maxima.sort_index()
#    min_only=minima.sort_index()
    return max_min


def plot_minmax_patterns(prices, max_min, patterns, ticker, signame, window=10, ema=3, colname='Close'):
    td=(prices.index[1] - prices.index[0])
    incr = str(td.seconds/60)
    max_min['_date']=max_min.index
#    print('incr:',incr, prices.index[1], td)
    if len(patterns) == 0:
        pass
    else:
        num_pat = len([x for x in patterns.items()][0][1])
        f, axes = plt.subplots(1, 2, figsize=(20, 10))
        axes = axes.flatten()
#        prices_ = prices.reset_index()['Close']
        prices_=prices[colname]
#        print('shape:',prices_.shape)
        axes[0].plot(prices_)
#        print(max_min.index, max_min.shape, max_min.Close[-3:])
        axes[0].scatter(max_min.index, max_min[colname], s=100, alpha=.3, color='orange')
        axes[1].plot(prices_)
        

        for name, end_day_nums in patterns.items():
#            print('name: %s  end_day_nums: %s , patterns:%s' % (name, end_day_nums, patterns))
            for i, tup in enumerate(end_day_nums):
                sd = tup[0]
                ed = tup[1]
#                print(sd, ed,)
#                print(max_min.loc[sd:ed].index, max_min.loc[sd:ed])
                axes[1].scatter(max_min.loc[sd:ed].index,
                              max_min.loc[sd:ed][colname],
                              s=200, alpha=.3)
        
                x=max_min.loc[sd:ed, '_date'].apply(lambda date: date.toordinal())
                y=max_min.loc[sd:ed][colname]
    #                   m, b = np.polyfit(x, y, 1)
                axes[1].plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
                axes[1].axvline(sd, color='green')
                axes[1].axvline(ed, color='blue')                
    
                plt.yticks([])
        plt.tight_layout()
        plt.title('{}: {}: EMA {}, Window {} ({} {} patterns)'.format(ticker, incr, ema, window, num_pat, signame))


#plot_minmax_patterns(subdf, sub_max_min, patterns, stock='ticker', window=10, ema=3)
def get_results(prices, max_min, pat, ticker, sig_name='IHS', ema_=3, window_=10, local_order=1):
    if len(prices)<30:
        return pd.DataFrame()
    incr=(prices.index[1] - prices.index[0])
    bullbear=get_bullbear_from_signame(sig_name)

    fw_list = [1, 5, 12, 24, 36] 
    #fw_list = [1, 2, 3]
    results = []
    if len(pat.items()) > 0:
        end_dates = [v for k, v in pat.items()][0]      
        for date in end_dates:  
            duration=date[1]-date[0]
            param_res = {'ticker': ticker,
                         'signame': sig_name,
                         'increment': incr,
                         'ema': ema_,
                         'window': window_, 
                         'bullbear':bullbear,
                         'date': date,
                         'signal_date':date[1],                      
                         'entry_date':date[1],                                             
                         'local_order':local_order,                         
                         'duraction':duration}

            for x in fw_list:
                returns = (prices['Close'].pct_change(x-1).shift(-x))
                entry_date=pd.to_datetime('19970701')
                try:
                    _=prices.loc[date[1]:]
                    if len(_)>0:
                        entry_date=_.index[1]
#                        print('entry_date is ',entry_date)
                    param_res['fw_ret_{}'.format(x)] = returns.loc[entry_date]   
                    param_res['entry_date']=entry_date
                except Exception as e:
                    param_res['fw_ret_{}'.format(x)] = e
                    
            results.append(param_res)  
    else:
        param_res = {'ticker': ticker,
                     'signame': sig_name,                     
                     'increment': incr,
                     'ema': ema_,
                     'window': window_,
                     'date': None,
                     'signal_date':None,
                     'entry_date':None,
                     'bullbear':bullbear,
                     'local_order':local_order,
                     'duraction':None}        
        for x in fw_list:
            param_res['fw_ret_{}'.format(x)] = None   
        results.append(param_res)
    return pd.DataFrame(results)

def extrema_screener(tlist=['AMZN', 'GOOG'], ema_list=[ 3, 5,8], window_list=[10], local_order=10, pset=['ihs', 'hs', 'hl', 'hhhl', 'lh', 'll'], plot=True, 
                        results=True, startdate='20070101', enddate=None, debug=False, remove_overlap=False, resample_rule=None):
    '''
    resample_rule = '2D', '3D', '5D' etc
    '''
    
    all_results = pd.DataFrame()
    pat_dicts={}
    for t in tlist:
        ticker=t
        pdf = cu.read_quote(t)
        if pdf is None:
            print('missing data for ',t)
            continue
        
        if not resample_rule is None:
            pdf=resample_df(pdf, rule=resample_rule)
            pdf.ffill(inplace=True)
#        print('b4 resample shape:',pdf.shape)
#        pdf=resample_df(pdf, '5D')
#        print('after resample shape:',pdf.shape)        
        pdf=pdf.loc[startdate:]        
        for ema_ in ema_list:
            for window_ in window_list:
#                max_min = get_max_min_dateidx(pdf, smoothing=ema_, window_range=window_, order=local_order)
                max_min=get_max_min_prices_dateidx(pdf, smoothing=ema_, window_range=window_, order=local_order)                                
                min_subset=max_min.query('minmax_type=="min"')                
                max_subset=max_min.query('minmax_type=="max"')                
#                print(max_min)
                if 'ihs' in pset:
                    signame='ihs'
                    pat = extremaPatternLooper.find_ihs_patterns(max_min)
                    if remove_overlap:
                        pat=remove_overlap_patterns(pat)                    
                    pat_dicts['%s %s' % (ticker, signame)]=pat                                        
    #                print('pat:',pat)
                    if plot == True:
                        plot_minmax_patterns(pdf, max_min, pat, t, signame, window_, ema_)

                    if results == True:
                        all_results = pd.concat([all_results, get_results(pdf, max_min, pat, t, signame, ema_, window_, local_order)], axis=0)

                        
                if 'hs' in pset:
                    signame='hs'
                    pat = extremaPatternLooper.find_hs_patterns(max_min)
                    if remove_overlap:
                        pat=remove_overlap_patterns(pat)                    
                    pat_dicts['%s %s' % (ticker, signame)]=pat                                            
    #                print('pat:',pat)
                    if plot == True:
                        plot_minmax_patterns(pdf, max_min, pat, t, signame, window_, ema_)

                    if results == True:
                        all_results = pd.concat([all_results, get_results(pdf, max_min, pat, t, signame, ema_, window_, local_order)], axis=0)

                        
                if 'vcp_up' in pset:
                    signame='vcp_up'
                    pat = extremaPatternLooper.find_vcp_up_patterns(max_min, ticker=ticker)
                    if remove_overlap:
                        pat=remove_overlap_patterns(pat)                    
                    pat_dicts['%s %s' % (ticker, signame)]=pat                                            
    #                print('pat:',pat)
                    if plot == True:
                        plot_minmax_patterns(pdf, max_min, pat, t, signame, window_, ema_)

                    if results == True:
                        all_results = pd.concat([all_results, get_results(pdf, max_min, pat, t, signame, ema_, window_, local_order)], axis=0)
                        
                        
                if 'hhhl' in pset:
                    signame='hhhl'
                    pat = extremaPatternLooper.find_higher_high_low_patterns(max_min)
                    if remove_overlap:
                        pat=remove_overlap_patterns(pat)                    
                    pat_dicts['%s %s' % (ticker, signame)]=pat                                            
    #                print('pat:',pat)
                    if plot == True:
                        plot_minmax_patterns(pdf, max_min, pat, t, signame, window_, ema_)

                    if results == True:
                        all_results = pd.concat([all_results, get_results(pdf, max_min, pat, t, signame, ema_, window_, local_order)], axis=0)

                if 'hl' in pset:
                    signame='hl'

#                    print('subset size:', subset.shape)
                    pat = extremaPatternLooper.find_higher_low_patterns(min_subset)
                    if remove_overlap:
                        pat=remove_overlap_patterns(pat)                    
                    pat_dicts['%s %s' % (ticker, signame)]=pat                            
    #                print('pat:',pat)
                    if plot == True:
                        plot_minmax_patterns(pdf, min_subset, pat, t, signame, window_, ema_)

                    if results == True:
                        all_results = pd.concat([all_results, get_results(pdf, min_subset, pat, t, signame, ema_, window_, local_order)], axis=0)

                if 'hh' in pset:
                    signame='hh'

#                    print('subset size:', subset.shape)
                    pat = extremaPatternLooper.find_higher_high_patterns(max_subset)
                    if remove_overlap:
                        pat=remove_overlap_patterns(pat)                    
                    pat_dicts['%s %s' % (ticker, signame)]=pat                            
    #                print('pat:',pat)
                    if plot == True:
                        plot_minmax_patterns(pdf, max_subset, pat, t, signame, window_, ema_)

                    if results == True:
                        all_results = pd.concat([all_results, get_results(pdf, max_subset, pat, t, signame, ema_, window_, local_order)], axis=0)
                        
                if 'lh' in pset:
                    signame='lh'
#                    print('subset size:', subset.shape)
                    pat = extremaPatternLooper.find_lower_high_patterns(max_subset)
                    if remove_overlap:
                        pat=remove_overlap_patterns(pat)                    
                    pat_dicts['%s %s' % (ticker, signame)]=pat                            
    #                print('pat:',pat)
                    if plot == True:
                        plot_minmax_patterns(pdf, max_subset, pat, t, signame, window_, ema_)

                    if results == True:
                        all_results = pd.concat([all_results, get_results(pdf, max_subset, pat, t, signame, ema_, window_, local_order)], axis=0)

                if 'll' in pset:
                    signame='ll'
#                    print('subset size:', subset.shape)
                    pat = extremaPatternLooper.find_lower_low_patterns(min_subset)
                    if remove_overlap:
                        pat=remove_overlap_patterns(pat)                    
                    pat_dicts['%s %s' % (ticker, signame)]=pat                        
    #                print('pat:',pat)
                    if plot == True:
                        plot_minmax_patterns(pdf, min_subset, pat, t, signame, window_, ema_)
                    if results == True:
                        all_results = pd.concat([all_results, get_results(pdf, min_subset, pat, t, signame, ema_, window_, local_order)], axis=0)
                        
    if debug:
        ret_dict={}
        ret_dict['pdf']=pdf    
        ret_dict['max_min']=max_min
        ret_dict['min_subset']=min_subset
        ret_dict['max_subset']=max_subset
        ret_dict['all_results']=all_results
        ret_dict['pat_dicts']= pat_dicts
        return ret_dict
    else:
        return all_results
    
def get_bullbear_from_signame(signame):
    signame_bullbear_dict={}
    signame_bullbear_dict['ihs']=1
    signame_bullbear_dict['hs']=-1
    signame_bullbear_dict['vcp_up']=1
    signame_bullbear_dict['hhhl']=1
    signame_bullbear_dict['hl']=1
    signame_bullbear_dict['hh']=1
    signame_bullbear_dict['lh']=-1
    signame_bullbear_dict['ll']=-1
    signame=signame.lower() 
    if signame in signame_bullbear_dict.keys():
        return signame_bullbear_dict[signame]
    return 0


def remove_overlap_patterns(patterns):
    ret_patterns = defaultdict(list)
    prev_ed=None
#    print('pattern:', patterns, type(patterns[1]))
    ret_list=[]
    removed_list=[]
    for name, end_day_nums in patterns.items():
#        print('nname:',name, )
        for i, tup in enumerate(end_day_nums):
            sd = tup[0]
            ed = tup[1]
            if prev_ed is None:                        
                #ret_list.append(prev_ed)
                ret_patterns[name].append((sd,ed))
                prev_ed=ed
                continue
            if sd < prev_ed:
#                print('over lap!, remove one patter:', sd, ed ,' vs ', prev_ed)                        
                removed_list.append({'prev_ed':prev_ed, 'sd':sd, 'ed':ed})

            else:
                #ret_list.append(prev_ed)
                ret_patterns[name].append((sd,ed))                        
                prev_ed=ed
                continue
#    print(ret_list, 'removed:', removed_list)
    return ret_patterns
        
        
def divergence_screener(tlist=['AMZN', 'GOOG'], ema_list=[ 3], window_list=[10], local_order=10, pset=['obv_bull_div', 'macd_bull_div'], ex_cond='up,down',
                          plot=True, results=True, startdate='20070101', enddate=None, debug=False, remove_overlap=True, main_size=6, resample_rule=None):
    '''
    resample_rule = '2D', '3D', '5D' etc    
    available TA indicator for divergence: ['obv', 'macd', 'rsi', 'ad', 'adsoc', 'mfi'], only macd and rsi does not require volume
    '''
    from datalib.extremaPatternLooper import extremaPatternLooper    
    all_results = pd.DataFrame()
    pat_dicts={}
    startdate_with_buf=pd.to_datetime(startdate)-pd.to_timedelta('100 days')
    extra_info=''
    for t in tlist:

        pdf = cu.read_quote(t)
        if pdf is None:
#            print('missing data for ',t)
            continue
        
#        print('b4 resample shape:',pdf.shape)
        if not resample_rule is None:
            pdf=resample_df(pdf, resample_rule).ffill()
            extra_info='%s resample %s' % (t, resample_rule)
#        print('after resample shape:',pdf.shape)        

        if enddate is None:
            pdf=pdf.loc[startdate_with_buf:]
        else:
            pdf=pdf.loc[startdate_with_buf:enddate]
        if len(pdf.dropna())==0:
            print('zero len data for ',t)
            continue
        
        
#        macd, signal, hist=talib.MACD(pdf.Close)
        import pandas_ta
        _=pdf.ta.macd()
        macd=_['MACD_12_26_9']
        hist=_['MACDh_12_26_9']
        signal=_['MACDs_12_26_9']
#        obv=talib.OBV(pdf.Close, pdf.Volume)
        obv=pdf.ta.obv()
#        rsi=talib.RSI(pdf.Close, 14)        
        rsi=pdf.ta.rsi(14)
#        ad=talib.AD(pdf.High, pdf.Low, pdf.Close, pdf.Volume) 
        ad=pdf.ta.ad()
#        adosc=talib.ADOSC(pdf.High, pdf.Low, pdf.Close, pdf.Volume)         
        adosc=pdf.ta.adosc()
#        mfi=talib.MFI(pdf.High, pdf.Low, pdf.Close, pdf.Volume)                 
        mfi=pdf.ta.mfi()
        pdf['obv']=obv
        pdf['macd']=macd
        pdf['rsi']=rsi  
        pdf['ad']=ad   
        pdf['adosc']=adosc
        pdf['mfi']=mfi 
        pdf_with_buf=pdf.copy()
        pdf=pdf.loc[startdate:]
        ta_min_max_dict={}
#        print('debug div scren:', debug, remove_overlap)

        for ema_ in ema_list:
            for window_ in window_list: 
                ticker=t
                close_max_min=get_max_min_prices_dateidx(pdf, smoothing=ema_, window_range=window_, order=local_order)                
                max_subset=close_max_min.query('minmax_type=="max"')
                min_subset=close_max_min.query('minmax_type=="min"')
                
                for col in ['obv', 'macd', 'rsi', 'ad', 'adosc', 'mfi']:
                    ta_min_max_dict[col]=get_max_min_dateidx(pdf.dropna(), colname=col, smoothing=ema_, window_range=window_, order=local_order)
                
                
                if debug and False:
                    print('last close points', close_max_min.index[-5:])                                    
                    for k in ta_min_max_dict.keys():
                        print('last %s points' % k, ta_min_max_dict[k].index[-5:])


#                print(max_subset)
                if 'mfi_bull_div' in pset:
                    col1='Close'
                    col2='mfi'
                    signame='%s_bull_div.%s' % (col2, ex_cond)
                    ta_minmax=ta_min_max_dict[col2].query('minmax_type=="min"')
#                    pat=find_bullish_divergence_patterns(min_subset, ta_minmax,  trend_down_col=col1, trend_up_col=col2, ex_cond=ex_cond) #ok
                    pat=extremaPatternLooper.find_general_divergence_patterns(max_subset, ta_minmax, main_col=col1, aux_col=col2, ex_cond=ex_cond, main_size=main_size)                                        

                    if remove_overlap:
                        pat=remove_overlap_patterns(pat)
                    pat_dicts['%s %s' % (ticker, signame)]=pat                        
#                    print('pat:',pat)
                    if plot == True:
                        #plot_minmax_patterns(pdf, min_subset, pat, t, signame, window_, ema_)
                        plot_dual_minmax_patterns(pdf, col1, col2, min_subset, ta_minmax, pat, ticker, signame, window=window_, ema=ema_, extra_info=extra_info)

                    if results == True:
                        all_results = pd.concat([all_results, get_results(pdf, min_subset, pat, t, signame, ema_, window_, local_order)], axis=0)
                        
                if 'mfi_bear_div' in pset:
                    col1='Close'
                    col2='mfi'
                    signame='%s_bear_div.%s' % (col2, ex_cond)
                    ta_minmax=ta_min_max_dict[col2].query('minmax_type=="max"')                    
#                    pat=find_bullish_divergence_patterns(max_subset, ta_minmax,  trend_down_col=col1, trend_up_col=col2, ex_cond=ex_cond) #ok                    
                    pat=extremaPatternLooper.find_general_divergence_patterns(max_subset, ta_minmax, main_col=col1, aux_col=col2, ex_cond=ex_cond, main_size=main_size)                                        

                    if remove_overlap:
                        pat=remove_overlap_patterns(pat)                    
                    pat_dicts['%s %s' % (ticker, signame)]=pat                    
#                    pat=find_bullish_divergence_patterns(max_subset, obv_max_min.query('minmax_type=="max"'),  trend_down_col='Close', trend_up_col='obv', ex_cond=ex_cond) #ok
#                    print('pat:',pat)
                    if plot == True:
#                        plot_minmax_patterns(pdf, max_subset, pat, t, signame, window_, ema_)
                        plot_dual_minmax_patterns(pdf, col1, col2, max_subset, ta_minmax, pat, ticker, signame, window=window_, ema=ema_, extra_info=extra_info)  
                    if results == True:
                        all_results = pd.concat([all_results, get_results(pdf, max_subset, pat, t, signame, ema_, window_, local_order)], axis=0)    

                if 'adsoc_bull_div' in pset:
                    col1='Close'
                    col2='adosc'
                    signame='%s_bear_div.%s' % (col2, ex_cond)
                    ta_minmax=ta_min_max_dict[col2].query('minmax_type=="min"')                    
#                    pat=find_bullish_divergence_patterns(min_subset, ta_minmax,  trend_down_col=col1, trend_up_col=col2, ex_cond=ex_cond) #ok
                    pat=extremaPatternLooper.find_general_divergence_patterns(min_subset, ta_minmax, main_col=col1, aux_col=col2, ex_cond=ex_cond, main_size=main_size)                                        
                    if remove_overlap:
                        pat=remove_overlap_patterns(pat)
                    pat_dicts['%s %s' % (ticker, signame)]=pat                                        
#                    print('pat:',pat)
                    if plot == True:
                        #plot_minmax_patterns(pdf, min_subset, pat, t, signame, window_, ema_)
                        plot_dual_minmax_patterns(pdf, col1, col2, min_subset, ta_minmax, pat, ticker, signame, window=window_, ema=ema_, extra_info=extra_info)  

                    if results == True:
                        all_results = pd.concat([all_results, get_results(pdf, min_subset, pat, t, signame, ema_, window_, local_order)], axis=0)
                        
                if 'adsoc_bear_div' in pset:
                    signame='ad_bear_div.%s' % ex_cond
                    col1='Close'
                    col2='adosc'
                    ta_minmax=obv_max_min.query('minmax_type=="max"')
#                    pat=find_bullish_divergence_patterns(max_subset, ta_minmax,  trend_down_col=col1, trend_up_col=col2, ex_cond=ex_cond) #ok                    
                    pat=extremaPatternLooper.find_general_divergence_patterns(max_subset, ta_minmax, main_col=col1, aux_col=col2, ex_cond=ex_cond, main_size=main_size)                                        
                    if remove_overlap:
                        pat=remove_overlap_patterns(pat)                    
                    pat_dicts['%s %s' % (ticker, signame)]=pat                                        
#                    pat=find_bullish_divergence_patterns(max_subset, obv_max_min.query('minmax_type=="max"'),  trend_down_col='Close', trend_up_col='obv', ex_cond=ex_cond) #ok
#                    print('pat:',pat)
                    if plot == True:
#                        plot_minmax_patterns(pdf, max_subset, pat, t, signame, window_, ema_)
                        plot_dual_minmax_patterns(pdf, col1, col2, max_subset, ta_minmax, pat, ticker, signame, window=window_, ema=ema_, extra_info=extra_info)             
                    if results == True:
                        all_results = pd.concat([all_results, get_results(pdf, max_subset, pat, t, signame, ema_, window_, local_order)], axis=0)    

                if 'ad_bull_div' in pset:
                    col1='Close'
                    col2='ad'
                    signame='%s_bull_div.%s' % (col2, ex_cond)
                    ta_minmax=ta_min_max_dict[col2].query('minmax_type=="min"')                    
#                    pat=find_bullish_divergence_patterns(min_subset, ta_minmax,  trend_down_col=col1, trend_up_col=col2, ex_cond=ex_cond) #ok
                    pat=extremaPatternLooper.find_general_divergence_patterns(min_subset, ta_minmax, main_col=col1, aux_col=col2, ex_cond=ex_cond, main_size=main_size)                                        
                    if remove_overlap:
                        pat=remove_overlap_patterns(pat)
                    pat_dicts['%s %s' % (ticker, signame)]=pat                                        
#                    print('pat:',pat)
                    if plot == True:
                        #plot_minmax_patterns(pdf, min_subset, pat, t, signame, window_, ema_)
                        plot_dual_minmax_patterns(pdf, col1, col2, min_subset, ta_minmax, pat, ticker, signame, window=window_, ema=ema_, extra_info=extra_info)  

                    if results == True:
                        all_results = pd.concat([all_results, get_results(pdf, min_subset, pat, t, signame, ema_, window_, local_order)], axis=0)
                        
                if 'ad_bear_div' in pset:
                    col1='Close'
                    col2='ad'
                    signame='%s_bear_div.%s' % (col2, ex_cond)
                    ta_minmax=ta_min_max_dict[col2].query('minmax_type=="max"')                    
#                    pat=find_bullish_divergence_patterns(max_subset, ta_minmax,  trend_down_col=col1, trend_up_col=col2, ex_cond=ex_cond) #ok                    
                    pat=extremaPatternLooper.find_general_divergence_patterns(max_subset, ta_minmax, main_col=col1, aux_col=col2, ex_cond=ex_cond, main_size=main_size)                                        
                    if remove_overlap:
                        pat=remove_overlap_patterns(pat)                    
                    pat_dicts['%s %s' % (ticker, signame)]=pat                                        
#                    pat=find_bullish_divergence_patterns(max_subset, obv_max_min.query('minmax_type=="max"'),  trend_down_col='Close', trend_up_col='obv', ex_cond=ex_cond) #ok
#                    print('pat:',pat)
                    if plot == True:
#                        plot_minmax_patterns(pdf, max_subset, pat, t, signame, window_, ema_)
                        plot_dual_minmax_patterns(pdf, col1, col2, max_subset, ta_minmax, pat, ticker, signame, window=window_, ema=ema_, extra_info=extra_info)              
                    if results == True:
                        all_results = pd.concat([all_results, get_results(pdf, max_subset, pat, t, signame, ema_, window_, local_order)], axis=0)    

                if 'rsi_bull_div' in pset:
                    col1='Close'
                    col2='rsi'
                    signame='%s_bull_div.%s' % (col2, ex_cond)
                    ta_minmax=ta_min_max_dict[col2].query('minmax_type=="min"')                    
#                    pat=find_bullish_divergence_patterns(min_subset, ta_minmax,  trend_down_col=col1, trend_up_col=col2, ex_cond=ex_cond) #ok
                    pat=extremaPatternLooper.find_general_divergence_patterns(min_subset, ta_minmax, main_col=col1, aux_col=col2, ex_cond=ex_cond, main_size=main_size)                                        
                    if remove_overlap:
                        pat=remove_overlap_patterns(pat)
                    pat_dicts['%s %s' % (ticker, signame)]=pat                                        
#                    print('pat:',pat)
                    if plot == True:
                        #plot_minmax_patterns(pdf, min_subset, pat, t, signame, window_, ema_)
                        plot_dual_minmax_patterns(pdf, col1, col2, min_subset, ta_minmax, pat, ticker, signame, window=window_, ema=ema_, extra_info=extra_info)  

                    if results == True:
                        all_results = pd.concat([all_results, get_results(pdf, min_subset, pat, t, signame, ema_, window_, local_order)], axis=0)
                         
                                        
                if 'rsi_bear_div' in pset:
                    col1='Close'
                    col2='rsi'
                    signame='%s_bear_div.%s' % (col2, ex_cond)
                    ta_minmax=ta_min_max_dict[col2].query('minmax_type=="max"')                    
#                    pat=find_bullish_divergence_patterns(max_subset, ta_minmax,  trend_down_col=col1, trend_up_col=col2, ex_cond=ex_cond) #ok                    
                    pat=extremaPatternLooper.find_general_divergence_patterns(max_subset, ta_minmax, main_col=col1, aux_col=col2, ex_cond=ex_cond, main_size=main_size)                                        
                    if remove_overlap:
                        pat=remove_overlap_patterns(pat)                    
                    pat_dicts['%s %s' % (ticker, signame)]=pat                                        
#                    pat=find_bullish_divergence_patterns(max_subset, obv_max_min.query('minmax_type=="max"'),  trend_down_col='Close', trend_up_col='obv', ex_cond=ex_cond) #ok
#                    print('pat:',pat)
                    if plot == True:
#                        plot_minmax_patterns(pdf, max_subset, pat, t, signame, window_, ema_)
                        plot_dual_minmax_patterns(pdf, col1, col2, max_subset, ta_minmax, pat, ticker, signame, window=window_, ema=ema_, extra_info=extra_info)              
                    if results == True:
                        all_results = pd.concat([all_results, get_results(pdf, max_subset, pat, t, signame, ema_, window_, local_order)], axis=0)                            
                        
                if 'obv_bull_div' in pset:
                    col1='Close'
                    col2='obv'
                    signame='%s_bull_div.%s' % (col2, ex_cond)
                    ta_minmax=ta_min_max_dict[col2].query('minmax_type=="min"')                    
#                    pat=find_bullish_divergence_patterns(min_subset, ta_minmax,  trend_down_col=col1, trend_up_col=col2, ex_cond=ex_cond) #ok
                    pat=extremaPatternLooper.find_general_divergence_patterns(max_subset, ta_minmax, main_col=col1, aux_col=col2, ex_cond=ex_cond, main_size=main_size)                                        
                    if remove_overlap:
                        pat=remove_overlap_patterns(pat)
#                    print('pat:',pat)
                    pat_dicts['%s %s' % (ticker, signame)]=pat                        
                    if plot == True:
                        #plot_minmax_patterns(pdf, min_subset, pat, t, signame, window_, ema_)
                        plot_dual_minmax_patterns(pdf, col1, col2, min_subset, ta_minmax, pat, ticker, signame, window=window_, ema=ema_, extra_info=extra_info)  

                    if results == True:
                        all_results = pd.concat([all_results, get_results(pdf, min_subset, pat, t, signame, ema_, window_, local_order)], axis=0)

                if 'obv_bear_div' in pset:
                    col1='Close'
                    col2='obv'
                    signame='%s_bear_div.%s' % (col2, ex_cond)
                    ta_minmax=ta_min_max_dict[col2].query('minmax_type=="max"')                    
#                    pat=find_bullish_divergence_patterns(max_subset, ta_minmax,  trend_down_col=col1, trend_up_col=col2, ex_cond=ex_cond) #ok                    
                    pat=extremaPatternLooper.find_general_divergence_patterns(max_subset, ta_minmax, main_col=col1, aux_col=col2, ex_cond=ex_cond, main_size=main_size)                                        
                    if remove_overlap:
                        pat=remove_overlap_patterns(pat)                    
                    pat_dicts['%s %s' % (ticker, signame)]=pat                                        
#                    pat=find_bullish_divergence_patterns(max_subset, obv_max_min.query('minmax_type=="max"'),  trend_down_col='Close', trend_up_col='obv', ex_cond=ex_cond) #ok
#                    print('pat:',pat)
                    if plot == True:
#                        plot_minmax_patterns(pdf, max_subset, pat, t, signame, window_, ema_)
                        plot_dual_minmax_patterns(pdf, col1, col2, max_subset, ta_minmax, pat, ticker, signame, window=window_, ema=ema_, extra_info=extra_info)             
                    if results == True:
                        all_results = pd.concat([all_results, get_results(pdf, max_subset, pat, t, signame, ema_, window_, local_order)], axis=0)                        

                if 'macd_bull_div' in pset:
                    col1='Close'
                    col2='macd'
                    signame='%s_bull_div.%s' % (col2, ex_cond)
                    ta_minmax=ta_min_max_dict[col2].query('minmax_type=="min"')                    
#                    pat=find_bullish_divergence_patterns(min_subset, ta_minmax,  trend_down_col=col1, trend_up_col=col2, ex_cond=ex_cond) #ok
                    pat=extremaPatternLooper.find_general_divergence_patterns(min_subset, ta_minmax, main_col=col1, aux_col=col2, ex_cond=ex_cond, main_size=main_size)                                        
                    if remove_overlap:
                        pat=remove_overlap_patterns(pat)  
                    pat_dicts['%s %s' % (ticker, signame)]=pat                                        
#                    print('pat:',pat)
                    if plot == True:
#                        plot_minmax_patterns(pdf, min_subset, pat, t, signame, window_, ema_)
                        plot_dual_minmax_patterns(pdf, col1, col2, min_subset, ta_minmax, pat, ticker, signame, window=window_, ema=ema_, extra_info=extra_info)              

                    if results == True:
                        all_results = pd.concat([all_results, get_results(pdf, min_subset, pat, t, signame, ema_, window_, local_order)], axis=0)   
                        


                if 'macd_bear_div' in pset:
                    col1='Close'
                    col2='macd'
                    signame='%s_bear_div.%s' % (col2, ex_cond)
                    ta_minmax=ta_min_max_dict[col2].query('minmax_type=="max"')                    
#                    pat=find_bullish_divergence_patterns(max_subset, ta_minmax,  trend_down_col=col2, trend_up_col=col1, ex_cond=ex_cond) #ok   
                    pat=extremaPatternLooper.find_general_divergence_patterns(max_subset, ta_minmax, main_col=col1, aux_col=col2, ex_cond=ex_cond, main_size=main_size)                    
                    if remove_overlap:
                        pat=remove_overlap_patterns(pat)                    
#                    pat=find_bullish_divergence_patterns(min_subset, obv_max_min.query('minmax_type=="min"'),  trend_down_col=col1, trend_up_col=col2, ex_cond=ex_cond) #ok                    
#                    print('pat:',pat)
                    pat_dicts['%s %s' % (ticker, signame)]=pat                        
                    if plot == True:
#                        plot_minmax_patterns(pdf, max_subset, pat, t, signame, window_, ema_)
                        plot_dual_minmax_patterns(pdf, col1, col2, max_subset, ta_minmax, pat, ticker, signame, window=window_, ema=ema_, extra_info=extra_info)            
                    if results == True:
                        all_results = pd.concat([all_results, get_results(pdf, max_subset, pat, t, signame, ema_, window_, local_order)], axis=0)    
                
                if 'macd_bear_div2' in pset:
                    col1='Close'
                    col2='macd'
                    signame='%s_bear_div.%s' % (col2, ex_cond)
                    ta_minmax=ta_min_max_dict[col2].query('minmax_type=="max"')                    
#                    pat=find_bullish_divergence_patterns(max_subset, ta_minmax,  trend_down_col=col1, trend_up_col=col2, ex_cond=ex_cond) #ok                    
                    pat=extremaPatternLooper.find_general_divergence_patterns(max_subset, ta_minmax, main_col='Close', aux_col='macd', ex_cond=ex_cond, main_size=main_size)
                    if remove_overlap:
                        pat=remove_overlap_patterns(pat)                    
                    
#                    pat=find_bullish_divergence_patterns(min_subset, obv_max_min.query('minmax_type=="min"'),  trend_down_col=col1, trend_up_col=col2, ex_cond=ex_cond) #ok                    
#                    print('pat:',pat)
                    pat_dicts['%s %s' % (ticker, signame)]=pat                        
                    if plot == True:
#                        plot_minmax_patterns(pdf, max_subset, pat, t, signame, window_, ema_)
                        plot_dual_minmax_patterns(pdf, col1, col2, max_subset, ta_minmax, pat, ticker, signame, window=window_, ema=ema_, extra_info=extra_info)             
                    if results == True:
                        all_results = pd.concat([all_results, get_results(pdf, max_subset, pat, t, signame, ema_, window_, local_order)], axis=0)                        

    if not 'entry_date' in all_results.columns:
        print(all_results)
        return None

    bidx=all_results.entry_date>pd.to_datetime(startdate)                    
    all_results=all_results[bidx]
    all_results['ticker_']=all_results['ticker']
    all_results['entry_date_']=all_results['entry_date']    
    all_results.set_index(['ticker_', 'entry_date_'], inplace=True)

    if debug:
        ret_dict={}
        ret_dict['pdf']=pdf_with_buf    
        ret_dict['close_max_min']=close_max_min
        ret_dict['ta_min_max_dict']=ta_min_max_dict
        ret_dict['all_results']=all_results
        ret_dict['pat_dicts']=pat_dicts
        ret_dict['pdf']=pdf        

        return ret_dict
    else:
        return all_results

def batch_price_extrema_screener(tlist=['SOXX', 'SMH'], pset=['hh', 'hl', 'll', 'lh', 'hhhl', 'hs', 'ihs', 'vcp_up'], local_order=1, startdate='20070101',
                                  enddate=None, remove_overlap=True, include_ch_ex_trade=True, resample_rule=None):
    bullbear='bull'
    all_res_dict={}

#    print('ex_cond:%s for tlist %s' % (ex_cond, tlist))
    res_df=extrema_screener(tlist=tlist, ema_list=[3], window_list=[10], local_order=local_order, startdate=startdate, enddate=enddate,
                               pset=pset, plot=False, remove_overlap=remove_overlap, resample_rule=resample_rule) 
    #big_res=pd.concat([res_df, res_df2, res_df3,res_df4])
    from datalib.chandelierExitWrapper import chandelierExitWrapper    
    big_res=res_df
    if len(res_df)==0:
        return res_df, pd.DataFrame()
    signame_list=res_df.signame.unique()
    all_perf_df_list=[]    
    for signame in signame_list:
        perf_df=get_performance_by_year(big_res, signame)
        if include_ch_ex_trade:
            ret_dict=chandelierExitWrapper.batch_add_ch_ex_columns(big_res)        
            big_res=ret_dict['new_res_df']
        
        all_perf_df_list=[]
        for signame in signame_list:
            all_perf_df_list.append(get_performance_by_year(big_res, signame=signame))
    all_perf_df=pd.concat(all_perf_df_list)
    all_res_dict['%s' % (signame)]=big_res
    all_perf_df=pd.concat(all_perf_df_list)
    all_res_df=pd.concat(all_res_dict.values())
    return all_res_df, all_perf_df


def batch_divergence_screener(tlist=['AAPL', 'AMZN'], bullbear='bull', pset_prefix=['obv', 'rsi', 'macd', 'ad', 'adosc', 'mfi'], local_order=1, 
                                startdate='20170101', enddate=None, remove_overlap=True, include_ch_ex_trade=True, resample_rule=None):
    from datalib.chandelierExitWrapper import chandelierExitWrapper        
    all_res_dict={}
    pset=['%s_%s_div' % (p, bullbear) for p in pset_prefix]
    print('pset:', pset)
    all_perf_df_list=[]
    for ex_cond in ['down,down', 'up,up', 'up,down', 'down,up']:
        print('ex_cond:%s for tlist %s' % (ex_cond, tlist))
        res_df=divergence_screener(tlist=tlist, ema_list=[3], window_list=[10], local_order=local_order, startdate=startdate, enddate=enddate,
                                   pset=pset, ex_cond=ex_cond,plot=False, debug=False, remove_overlap=remove_overlap, resample_rule=resample_rule) 
        #big_res=pd.concat([res_df, res_df2, res_df3,res_df4])
        if len(res_df)==0:
            continue
        
        big_res=res_df
        signame_list=res_df.signame.unique()
        
        for signame in signame_list:
            perf_df=get_performance_by_year(big_res, signame)
            all_perf_df_list.append(perf_df)
        all_res_dict['%s|%s' % (signame, ex_cond)]=big_res
        all_res_df=pd.concat(all_res_dict.values())
    if len(all_perf_df_list)>0:
        all_perf_df=pd.concat(all_perf_df_list)        


    if include_ch_ex_trade:
        ret_dict=chandelierExitWrapper.batch_add_ch_ex_columns(all_res_df)        
        new_res_df=ret_dict['new_res_df']
#        display(new_res_df.head())

        new_res_df=ret_dict['new_res_df']
        signame_list=new_res_df.signame.unique()    
        all_perf_df_list=[]
        for signame in signame_list:
            all_perf_df_list.append(get_performance_by_year(new_res_df, signame=signame))
        all_perf_df=pd.concat(all_perf_df_list)
        return new_res_df, all_perf_df
    else:
        return all_res_df, all_perf_df



def test_divergence_pattern():
    tlist=['FB', 'TSLA', 'SQ']
    all_res_df, all_perf_df=batch_divergence_screener(tlist=tlist, local_order=2)
    ret_dict=batch_add_ch_ex_columns(all_res_df)
    new_res_df=ret_dict['new_res_df']
    signame_list=list(new_res_df.signame.unique())
    signame_perf_dict={}
    all_perf_df_dict={}
    for signame in signame_list:        
        all_perf_df_dict[signame]=get_performance_by_year(new_res_df, signame)
             
    all_trades_df=ret_dict['all_trades_df']
    srt=signalTradeReviewer()
    ret=srt.review_all_ticker(all_trades_df)
    ret_dict['signame_perf_dict']=signame_perf_dict
    ret_dict['all_tickers_review']=ret
    ret_dict['all_perf_df']=pd.concat(all_perf_df_dict.values())
    return ret_dict

def test2():

    res_df=test_divergence_pattern()    
    display(res_df)
    #test_ihs()
    #res_df=test_screener()
    signame_list=res_df.signame.unique()
#    print(signame_list)
    display(res_df)
    all_res_list=[]
    for signame in signame_list:
        bar_list=[5,12,24,36]
        for n in bar_list:
            subres_df=res_df.query('signame=="%s"' % signame).dropna()        
    #        display('subres_df:',subres_df.head())
            r=subres_df['fw_ret_%s' %n ]

            all_res_list.append(subres_df)
            print(signame, 'f%s:' % n, r.mean(), len(r), 'exposure adjusted:', r.mean()/n*250, 'total_return:', r.sum())    

    all_res_df=pd.concat(all_res_list)
    
def get_year_range_dict(big_res, signame):
    _=big_res.dropna().query('signame=="%s"' % signame)
#    print(_.dtypes)
#    print(_.entry_date)
    if len(_)==0:
        return []
    sd=_.entry_date.min()
    ed=_.entry_date.max()
    print('sd:', sd, _, ed)
    y1=sd.year
    y2=ed.year
#    print(y1, y2)
    range_list=[]
    range_list.append({'startdate':sd, 'enddate':ed, 'year':'all'})
    for y in range(y1, y2+1):
#        print(y)
        range_list.append({'startdate':pd.to_datetime('%s0101' % y), 'enddate':pd.to_datetime('%s1231' % y), 'year':y})
#    print(range_list)
    return range_list

def get_performance_by_year(big_res, signame):
    bar_list=[5,12,24,36]
    
#    print(big_res.signame.unique())
#    print(big_res.signame.unique())    
    range_list=get_year_range_dict(big_res, signame)
#    print('range_list:', range_list)
    all_perf_dict={}    
    for range_dict in range_list:
#        print('xxx:', range_dict)
        sd=range_dict['startdate']
        ed=range_dict['enddate']
        year=range_dict['year']
        bidx=(big_res.entry_date>=sd )& ( big_res.entry_date<=ed)
#        print(bidx)
        sub_res=big_res[bidx].query('signame=="%s"' % signame)            

        for n in bar_list:
            r=sub_res.dropna()['fw_ret_%s' %n ]
            winner=len(r[(r>0)])
            loser=len(r[(r<=0)])            
            winrate=0
            if len(r)>0:
                winrate=winner/len(r)
            exit_type='bar_%s' % n
#            print(sd, signame, 'f%s:' % n, r.mean(), len(r), 'exposure adjusted:', r.mean()/n*250, 'total_return:', r.sum(), '1st trade:', 'winrate:',winrate, sub_res.entry_date.min())    
            key='%s.%s.%s' % (sd, signame, exit_type)

            
            perf_dict={'ret':r.mean(), 'tcnt':len(r), 'exp_adj_ret':r.mean()/n*250, 'total_ret':r.sum(), 'winrate':winrate, 'signame':signame, 'exit_type':exit_type, 'year':year}
            all_perf_dict[key]=perf_dict
        if 'ch_exit_pct_profit' in sub_res.columns:
            r=sub_res['ch_exit_pct_profit']                                                                                                                     
            exit_type='cha_ex'
            key='%s.%s.%s' % (sd, signame, exit_type)
            perf_dict={'ret':r.mean(), 'tcnt':len(r), 'exp_adj_ret':r.mean()/n*250, 'total_ret':r.sum(), 'winrate':winrate, 'signame':signame, 'exit_type':exit_type, 'year':year}
            all_perf_dict[key]=perf_dict
    return pd.DataFrame.from_dict(all_perf_dict).T

def plot_dual_minmax_patterns(pdf, col1, col2, max_min1, max_min2, patterns, ticker, signame, window=10, ema=3, debug=True, extra_info=None):
    td=(pdf.index[1] - pdf.index[0])
    incr = str(td.days)
    if extra_info is None:    
        extra_info='incr %sD' % incr
    else:
        extra_info='%s incr %sD' % (extra_info, incr)
#    print('incr:',incr, prices.index[1], td)
#    patterns=[{},{}]
    max_min1['_date']=max_min1.index
    max_min2['_date']=max_min2.index
#    sd=min(max_min1.index[0], max_min2.index[0])
#    ed=min(max_min1.index[1], max_min2.index[-1])    
    dfta=pdf[[col1,col2]].dropna().copy()
    col1ma50='%s_ma50' % col1
    col2ma20='%s_ma20' % col2
    col1ma100='%s_ma100' % col1
    col2ma30='%s_ma30' % col2
    
    dfta[col1ma50]=dfta[col1].rolling(50).mean()
    dfta[col2ma20]=dfta[col2].rolling(20).mean()    
    dfta[col1ma100]=dfta[col1].rolling(100).mean()
    dfta[col2ma30]=dfta[col2].rolling(30).mean()    
    
#    dfta=dfta.loc[sd:ed]
    dfta['_date']=dfta.index
    num_pat=0
    if len(patterns) == 0 and not debug:
        pass
    else:

        f, axes = plt.subplots(2, 2, figsize=(20, 10))
        #axes 0, 1, ts1 plot, axes[2,3] ts2 plot
        axes = axes.flatten()
        #        prices_ = prices.reset_index()['Close']
        dfta[col1]
        #        print('shape:',prices_.shape)
        axes[0].plot(dfta[col1].dropna())
#        axes[0].plot(dfta[col1])
        axes[0].plot(dfta[col1ma50])    
        axes[0].legend([col1, col1ma50])
#        axes[2].plot(dfta[col2].dropna())    
        axes[2].plot(dfta[col2].dropna())         
        axes[2].plot(dfta[col2ma20])        
        axes[2].legend([col2, col2ma20])    
#        axes[2].plot(dfta[col2].rolling(50).mean())  
        #        print(max_min.index, max_min.shape, max_min.Close[-3:])
        axes[0].scatter(max_min1.index, max_min1[col1], s=100, alpha=.3, color='orange')

        axes[2].scatter(max_min2.index, max_min2[col2], s=100, alpha=.3, color='orange')    
        axes[1].plot(dfta[col1].dropna())
        axes[1].plot(dfta[col1ma100])    
        axes[1].legend([col1, col1ma100])
        
#        axes[3].plot(dfta[col2].dropna())        
        axes[3].plot(dfta[col2])                
        axes[3].plot(dfta[col2ma30])    
        axes[3].legend([col2, col2ma30])
    
        if len(patterns)>0:
            num_pat = len([x for x in patterns.items()][0][1])
            for name, end_day_nums in patterns.items():
#                print('name: %s  end_day_nums: %s , patterns:%s' % (name, end_day_nums, patterns))
                for i, tup in enumerate(end_day_nums):
                    sd = tup[0]
                    ed = tup[1]
        #                print(sd, ed,)
        #                print(max_min.loc[sd:ed].index, max_min.loc[sd:ed])
                    between_idx=pdf.loc[sd:ed].index
                    bullbear=get_bullbear_from_signame(signame)
                    for idx in between_idx:
                        if bullbear>0:
                            axes[1].axvline(sd, color='lightgreen')
                        if bullbear<0:                            
                            axes[1].axvline(sd, color='pink')

                            
                    axes[1].scatter(max_min1.loc[sd:ed].index,
                                  max_min1.loc[sd:ed][col1],
                                  s=200, alpha=.3)

                    x=max_min1.loc[sd:ed, '_date'].apply(lambda date: date.toordinal())
                    y=max_min1.loc[sd:ed][col1]
 #                   m, b = np.polyfit(x, y, 1)
                    axes[1].plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
                    axes[1].axvline(sd, color='green')

                    axes[1].axvline(ed, color='blue')                
                    axes[3].scatter(max_min2.loc[sd:ed].index,
                                  max_min2.loc[sd:ed][col2],
                                  s=200, alpha=.3)
                    x=max_min2.loc[sd:ed, '_date'].apply(lambda date: date.toordinal())
                    y=max_min2.loc[sd:ed][col2]
                    axes[3].plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))                    
                    axes[3].axvline(sd, color='green')
                    axes[3].axvline(ed,  color='blue') 

                    plt.yticks([])
        
            axes[0].text(ed, max_min1[col1][0], extra_info)
                    
        plt.tight_layout()
        plt.title('{}: {}: EMA {}, Window {} ({} {} patterns {} {} )'.format(ticker, incr, ema, window, num_pat, signame, col1, col2))
        
def test():
    #_all_res_df=divergence_screener(tlist=['AMZN'], ema_list=[ 3], window_list=[10], local_order=1, pset=['macd_bear_div'], ex_cond='up,down',
    ticker='QQQ'
    tlist=[ticker]
    tlist=['QQQ', 'AMZN', 'FB', 'AAPL', 'GOOG', 'NVDA', 'SPY']
    _all_res_df=divergence_screener(tlist=tlist, ema_list=[3], window_list=[10], local_order=2, pset=['obv_bear_div'], ex_cond='up,down',
                              plot=True, results=True, startdate='20170101', enddate='20210127', debug=True, remove_overlap=True)

    all_results=_all_res_df['all_results']
    pdf=_all_res_df['pdf']
    close_max_min=_all_res_df['close_max_min']
    startdate='20201101'        
    enddate='20210214'
    subpdf=pdf.loc[startdate:enddate]
    sub_max_min1=close_max_min.loc[startdate:enddate]
    #sub_max_min2=macd_max_min.loc[startdate:enddate]
    _=_all_res_df['ta_min_max_dict']
    print(_.keys())
    rsi_max_min=_['rsi']
    sub_max_min2=rsi_max_min.loc[startdate:enddate]
    #plot_dual_minmax_patterns(pdf, col1, col2, max_min1, max_min2, patterns, ticker, signame, window=10, ema=3, debug=True)
    ticker='QQQ'
    col1='Close'
    col2='macd'
    col2='rsi'

    res_df, perf_df=batch_price_extrema_screener()
    #res_df, perf_df=batch__extrema_screener()
    res_df, perf_df=batch_divergence_screener()
#res_df, perf_df=batch_divergence_screener()    
#perf_df
