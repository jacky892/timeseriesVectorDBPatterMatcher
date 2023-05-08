# %load datalib/extremaPlotter.py
from vcplib.vcpUtil import get_cup_features
import numpy as np
import pandas as pd

from datalib.extremaPatternUtil import get_max_min_prices_dateidx, get_max_min_dateidx, remove_overlap_patterns, get_results
from datalib.commonUtil import commonUtil as cu

    
def find_box_on_plot_by_date(pdf, dateidx='20201103',enddate=None, barcnt=250, need_3pt=True, is_forecast=False, rticker='IWV'):

    if not enddate is None:
        pdf=pdf.loc[:enddate]
    subdf=pdf.iloc[-barcnt:]
    min_max=get_max_min_prices_dateidx(subdf,  smoothing=1, window_range=10, order=2)

    delta_th=0.03
    xmin=dateidx
    subdf=pdf.iloc[-barcnt:]    
    min_max['extreme']=min_max.High
    __=min_max.apply(lambda x:x.High if x.minmax_type=='max' else x.Low, axis=1)
    min_max['extreme']=__  
    idx_list=min_max.index
    pdf['extreme_se']=np.nan
    #pdf['extreme'].plot()
    pdf.loc[idx_list, 'extreme_se']=min_max['extreme']
    x=pdf.loc[dateidx]    
    xmax=pd.to_datetime(dateidx)

#    print('xmin:',xmin)
#    ax=subdf.Close.plot(figsize=(18,8))

    lb_xmin=0
    ub_xmin=0
    ub_near_cnt=0
    box_score=0
    #for n2 in [10, 15, 20, 25]:
    for n2 in [10, 15 ]:
        field_n=f'ub{n2}'
        ub_near_cnt=0        
        #for n1 in [n for n in [10, 15, 20, 25 ,30,35,40,45,50] if n>n2]:
        for n1 in [n for n in [15, 20, 25 ,30] if n>n2]:

            field=f'u1b{n1}'
#            print('max:', x[field], x[field_n], n1, n2)
            if x[field_n] >= x[field]:
#                print('got max point at ',field, field_n)        
                delta=abs(x[field_n]-x[field])/x[field]
                if delta<delta_th:
                    ub_near_cnt=ub_near_cnt+1
                    box_score=box_score+0.1001
                recent_max=x[field_n]
                maxn=n1

                ub_xmin=pd.to_datetime(pdf.loc[:dateidx].iloc[-maxn:].index[0]        )
                
    lb_near_cnt=0
    #for n2 in [10, 15, 20, 25 ]:
    for n2 in [10, 15]:
        field_n=f'lb{n2}'
        lb_near_cnt=0        
        #for n1 in [n for n in [20,25,30,35,40,45,50] if n>n2]:
        for n1 in [n for n in [10, 15, 20,25,30] if n>n2]:
            field=f'l1b{n1}'
#            print('max:', x[field], x[field_n], n1, n2)
            if x[field_n] <= x[field]:
                delta=abs(x[field_n]-x[field])/x[field]
                if delta<delta_th:
#                    print('ub delta', delta, x[field_n] ,  x[field], n2, n1)
                    lb_near_cnt=lb_near_cnt+1
                    box_score=box_score+0.1000001
#                print('got min point at ',field, field_n)        
                recent_min=x[field_n]
                minn=n1

                lb_xmin=pd.to_datetime(pdf.loc[:dateidx].iloc[-minn:].index[0]        )
    ret_dict={}
    def_win_size=10
    _boxpdf=pdf.loc[:dateidx].iloc[-def_win_size:]
    ret_dict['pdf']=_boxpdf
    idx_price=pdf.Close.loc[:dateidx][-1]
    signame='box_breakout'
    if ub_xmin==0 or lb_xmin==0:
        print(f'nobox for {dateidx}, use default {def_win_size} trading day windows')
        box_score=box_score-3
        if is_forecast:
            signame='forecast'
            ub_xmin=pd.to_datetime(pdf.loc[:dateidx].iloc[-def_win_size:].index[0] )
            lb_xmin=pd.to_datetime(pdf.loc[:dateidx].iloc[-def_win_size:].index[0] )
            box_subdf=pdf.loc[:dateidx].iloc[-def_win_size:]
            recent_min=box_subdf.Low.min()
            recent_max=box_subdf.High.max()
#            ub_xmin=box_subdf.index[box_subdf.High.argmax()]
#            lb_xmin=box_subdf.index[box_subdf.Low.argmin()]
            print('xxxxxxxxxxxxxxxzzzzzzzzzzzx', ub_xmin, lb_xmin, recent_min, recent_max)
        else:
            print('returning box pdf 111')
            box_band=_boxpdf.High.max()-_boxpdf.Low.min()
            if (box_band/idx_price)>0.1:
                return None
            return ret_dict
    if (lb_near_cnt==0 or ub_near_cnt==0) and not is_forecast:
        print('not 3pt box for ',dateidx)
        box_score=box_score-1
        if  need_3pt:
            box_score=box_score-2
            return ret_dict
    band_width=recent_max-recent_min
    atr=x['atr']
    
    print('lb ub near pt %s %s  bw:%s, atr:%s idx price:%s' % (lb_near_cnt, ub_near_cnt, band_width, atr, idx_price))
    if (atr>band_width or band_width/idx_price>0.1) and not is_forecast:
        print('not tight enough')
        return None
    xmin=max(lb_xmin, ub_xmin)
    ret_dict={}
    ret_dict['signame']=signame
    ret_dict['start_date']=xmin
    ret_dict['end_date']=xmax
    ret_dict['ub']=recent_max
    ret_dict['lb']=recent_min
    ret_dict['clb_xmin']=lb_xmin
    ret_dict['ub_xmin']=ub_xmin    
#    ret_dict['pdf']=pdf
#    ret_dict['subdf']=subdf
    from datalib.patternTraderUtil import gen_minervini_filter_criteria
    pdf=gen_minervini_filter_criteria(pdf)
    pdf['f_mmfilter']=pdf.mmfilter_flag*1.0
    pdf['date']=pdf.index
#    print('xx with date and mmfilter',pdf.tail())
    macd_df=pdf.ta.macd()
    macd=macd_df['MACD_12_26_9']
    hist=macd_df['MACDh_12_26_9']
    signal=macd_df['MACDs_12_26_9']
    pdf['macd']=macd
    pdf['macd_signal']=signal
    pdf['macd_hist']=hist
    subdf=pdf.iloc[-barcnt:]
    xmin_with_buf=xmin-pd.to_timedelta('10 days')    
    i=0
    while xmin_with_buf not in subdf.index and i<5:
        xmin_with_buf=xmin_with_buf-pd.to_timedelta('1 day')
#        print('adjusting xmin_with_buf', xmin_with_buf)
        i=i+1
    xmin_with_buf=max(subdf.index[0], xmin_with_buf)
#    print('all index:',subdf.index)
    _=subdf.loc[xmin_with_buf:xmax]
#    print('sub index:',_.index, _)
#    macd, signal, hist=talib.MACD(_.Close)
    import pandas_ta
    _cols=['Open', 'High', 'Low' ,'Close', 'Volume']
##    print('submacd is ',_[_cols].tail())
#    macd_df=_[_cols].dropna().ta.macd()
#    print('macd_df:',macd_df.tail())
#    _['obv']=talib.OBV(_.Close, _.Volume)
    _['obv']=_.ta.obv()
#    _['rsi']=talib.RSI(_.Close, 14)        
    _['rsi']=_.ta.rsi(14)
#    _['ad']=talib.AD(_.High, _.Low, _.Close, _.Volume) 
    _['ad']=_.ta.ad()
#    _['adosc']=talib.ADOSC(_.High, _.Low, _.Close, _.Volume)         
    _['adosc']=_.ta.adosc()
#    _['mfi']=talib.MFI(_.High, _.Low, _.Close, _.Volume)                 
    _['mfi']=_.ta.mfi()
    _['rs']=get_box_rs(_)
    box_subdf=_.loc[xmin:xmax].copy()
    ret_dict['box_subdf']=box_subdf
    ret_dict['box_score']=box_score
    ret_dict['datapoint']=x
    volu_bias_dict=get_box_volu_recency_bias(box_subdf, rwf=0.1)
    ret_dict['volu_bias']=volu_bias_dict['volu_bias']
    ret_dict['box_stop']=volu_bias_dict['flat_stop']
    ret_dict['box_stop_idx']=volu_bias_dict['flat_stop_idx']    
    ret_dict['box_stop_adj']=volu_bias_dict['recency_bias_stop']
    ret_dict['box_stop_adj_idx']=volu_bias_dict['recency_bias_stop_idx']    
    ret_dict['c2ub_ratio'], ret_dict['c2lb_ratio']=get_c2ublb_ratio(x)
    return ret_dict


def get_box_ta_matrix(box_subdf, cdl_score_dict={}):
    ret_dict={}
    ret_dict['obv_plus']=1.0*box_subdf['obv'].iloc[-3:].sum()>0
    ret_dict['ad_plus']=1.0*box_subdf['ad'].iloc[-3:].mean()>0
    ret_dict['mfi_plus']=1.0*box_subdf['mfi'].iloc[-3:].mean()>0
    ret_dict['adosc_plus']=1.0*box_subdf['adosc'].iloc[-3:].mean()>0
    ret_dict['macd_plus']=1.0*box_subdf['macd_signal'].iloc[-3:].mean()>0
    ret_dict['rs_plus']=1.0*box_subdf['rs'].iloc[-3:].mean()>1
    ret_dict['snr_mid_plus']=1.0*box_subdf.eval('Close>=snr_mid').iloc[-3:].mean()>0    
    ret_dict['candle_net_score']=cdl_score_dict['net_cdl_score']
    ret_dict['score']=sum(ret_dict.values())
    ###### all score based value must be before the sum(ret_dict.values())
    cols=['Open', 'High', 'Low', 'Close', 'Volume']
    ret_dict['snr_mid']=box_subdf.snr_mid.iloc[-3:].mean()
    ret_dict['tm_snr_mid']=box_subdf.tm_snr_mid.iloc[-3:].mean()
#    ret_dict['tm_snr_mid_plus']=1.0*box_subdf.eval('Close>=tm_snr_mid').iloc[-3:].mean()>0    
#    ret_dict['tm_snr_mid']=box_subdf.tm_snr_mid    
    v=['obv', 'rs', 'volume_ma', 'f_breakout_long', 'f_mmfilter', 'mfi',  'rClose', 'low_52w', 'macd_hist', 'macd', 
        'f_breakout_comb', 'ma_long', 'high_52w',  'macd_signal', 'rsi', 'adosc', 'ad', 'date',
        'f_vcomb_flag', 'min_max_trend_up', 'f_w50_cup_signal', 'f_vola_contr', 'f_volu_contr', 'f_v2_contr'] 
    keep_cols_=[*v, *cols]
#    for col in cols:

    missing_cols=[col for col in keep_cols_ if not col in box_subdf.columns]
    keep_cols=[col for col in keep_cols_ if col in box_subdf.columns]
    print('missing cols are ', missing_cols)
#    for col in keep_cols_:
    for col in keep_cols:
        ret_dict[col]=box_subdf.iloc[-1][col]
    ret_dict['available_cols']=list(box_subdf.columns)

    return ret_dict

def get_box_rs(box_subdf, rticker='IWV'):
    rpdf=cu.read_quote(rticker)
    box_subdf['rClose']=rpdf.Close
    box_subdf['rs']=(1+box_subdf['Close'].pct_change(10))/(1+box_subdf['rClose'].pct_change(10))
    return box_subdf['rs']

def filter_duplicated_box(good_dict, ticker, keep_list=[]):
    if len(good_dict)==0:
        return []
    filtered_list=[good_dict[0]]
    prev_end=good_dict[0]['end_date']
    prev_start=good_dict[0]['start_date']    
    for ret_dict in good_dict[1:]:
#        print(ret_dict.keys())
        box_start=ret_dict['start_date']
        box_end=ret_dict['end_date']        
        td=pd.to_timedelta('5 days')
#        if prev_end>(box_start+td):
#        print('box_end vs keep list ', box_end, keep_list)
        if prev_end>(box_end-td*2) and not (box_end in keep_list):            
#            print('skip for end vs start %s %s ' % (prev_end, box_start))
            continue
        prev_start=box_start
        prev_end=box_end
        filtered_list.append(ret_dict)
    filtered_dict={}
    for keep_dict in filtered_list:
        box_start=keep_dict['start_date']
        box_end=keep_dict['end_date']
        sd_str=box_start.strftime('%Y%m%d')
        ed_str=box_end.strftime('%Y%m%d') 
        key=f'{ticker}_{sd_str}_{ed_str}'
        filtered_dict[key]=keep_dict
#    for n in filtered_list:
#        print(n['start_date'], n['end_date'])
#    return filtered_list
    return filtered_dict

def get_box_obv_stop(subdf, rwf=0, bin_cnt=8, wtype='geo'):
    from  vcplib.vcpUtil import vcpUtil as vcu 
#    print('rwf is ',rwf)
    pbv_df=vcu.get_price_by_volume_df(subdf, bin_cnt=8, rwf=rwf, wtype=wtype)
#    print(pbv_df)
    stop_idx=(pbv_df.volume_by_price).argmax()
    stop_level=pbv_df.volume_by_price.index[stop_idx]
    ret_dict={}
    ret_dict['stop_idx']=(stop_idx)/bin_cnt
    ret_dict['stop_level']=float(stop_level)
    return ret_dict

def get_box_volu_recency_bias(box_subdf, rwf=0.1):
    stop1_dict=get_box_obv_stop(box_subdf.copy(), rwf=0)    
    rwf=2/len(box_subdf)
    stop2_dict=get_box_obv_stop(box_subdf.copy(), rwf=rwf, wtype='ari')     
#    stop2_dict=get_box_obv_stop(box_subdf.copy(), rwf=rwf, wtype='geo') 
    stop1=stop1_dict['stop_level']
    stop1_idx=stop1_dict['stop_idx']    
    stop2=stop2_dict['stop_level']    
    stop2_idx=stop2_dict['stop_idx']        
    print(stop1, stop2)
    bias_dict={}
    volu_bias=0
    if stop2>stop1:
        print('recent vol bias up', box_subdf.shape)
        volu_bias=1
    if stop2<stop1:
        print('recent vol bias down', box_subdf.shape)
        volu_bias=-1        
    bias_dict['flat_stop']=stop1
    bias_dict['recency_bias_stop']=stop2    
    bias_dict['flat_stop_idx']=stop1_idx
    bias_dict['recency_bias_stop_idx']=stop2_idx

    bias_dict['volu_bias']=volu_bias
    return bias_dict

def get_box_pbv(ret_dict):
    from vcplib.vcpUtil import vcpUtil as vcu 
    box_start=ret_dict['start_date']
    box_end=ret_dict['end_date']    
    ub=ret_dict['ub']
    lb=ret_dict['lb']    
    pdf=ret_dict['subdf']
    box_subdf=pdf.loc[box_start:box_end]
    box_subdf
    x=vcu.get_price_by_volume_obv_df(box_subdf, bin_cnt=6)    
    max_v=x.volume_by_price.max()



def get_c2ublb_ratio(datapoint):
    close=datapoint.Close
    ratio_list=[]
    for col in [col for col in datapoint.keys() if 'ub' in col]:
        ratio=close/datapoint[col]
        ratio_list.append(ratio)
#        print(col, ratio)
    c2ub_ratio=sum(ratio_list)/len(ratio_list)
    ratio_list=[]
    for col in [col for col in datapoint.keys() if 'lb' in col]:
        ratio=close/datapoint[col]
#        print(col, ratio)    
        ratio_list.append(ratio)
    c2lb_ratio=sum(ratio_list)/len(ratio_list)
#    print(c2ub_ratio, c2lb_ratio)
    return c2ub_ratio, c2lb_ratio

def draw_all_box(new_box_list, subdf, ax):
    if ax is None:
        ax=subdf.Close.plot()
    stop_list=[]
    stop_msg_list=[]
    box_signal_dict={}
    #for ret_dict in new_box_list:
    for k in new_box_list.keys():
        ret_dict=new_box_list[k]
        box_details_dict=add_box_pbv(ret_dict, ax)
        stop_level=box_details_dict['stop_level']
        ub=ret_dict['ub']
        ed=ret_dict['end_date']    
        risk=float(ub)-float(stop_level)
        risk_pct=risk/ub
        stop_msg='stop level for long @%s on %s is %s, risk is %s pct %s' % (ub, ed, stop_level, risk, risk_pct)
        box_details_dict['stop_msg']=stop_msg
        box_details_dict['risk']=risk
        box_details_dict['ub']=ub
        box_details_dict['signal_date']=ed
        box_details_dict['stop_level']=float(stop_level)    
        box_signal_dict[ed]=box_details_dict
    return box_signal_dict

def check_breakout_box(pdf, dateidx=None, barcnt=300, rticker='IWV',):
    '''
    pdf need ub/lb field returned from get_box_tapdf(ticker)
    '''
    if dateidx is None:
        dateidx=pdf.index[-1]
    ret_dict=find_box_on_plot_by_date(pdf, dateidx=dateidx, enddate=dateidx, barcnt=barcnt,  need_3pt=False, rticker=rticker)
    return ret_dict

def suggest_buy(box_details):
    f1=1.0*box_details['recent_vol_bias_plus']>=0
    f2=1.0*box_details['score']>2 
    f3=1.0*box_details['rs_plus']==True
    f4=1.0*box_details['snr_mid_plus']==True    
    f5=1.0*box_details['duration']>25
    sig_date=box_details['signal_date']
    overall_score=f1+f2+f3
    if overall_score>2 and f4>0 and f5>0:
        print('buy sig date:',sig_date)
        return True
    return False

def strong_buy(box_details):
    f1=box_details['recent_vol_bias_plus']>=0
    f2=box_details['score']>4  
    f3=box_details['rs_plus']==True
    f4=1.0*box_details['snr_mid_plus']==True        
    f5=box_details['duration']>39
    sig_date=box_details['signal_date']
    if f1 and f2 and f3 and f4:
        print('buy sig date:',sig_date)
        return True
    return False

def suggest_sell(box_details):
    f1=1.0*box_details['recent_vol_bias_plus']<0
    f2=1.0*box_details['score']<3
    f3=1.0*(not box_details['rs_plus'])
    f4=not box_details['snr_mid_plus']==True        
    f5=box_details['duration']>15
    sig_date=box_details['signal_date']
    overall_score=f1+f2+f3
    if overall_score>1 and f4 and f5:
        print('sell sig date:',sig_date)
        return True
    return False

def strong_sell(box_details):
    f1=box_details['recent_vol_bias_plus']<0
    f2=box_details['score']<3
    f3=not box_details['rs_plus']
    f4=not box_details['snr_mid_plus']==True        
    f5=box_details['duration']>28
    sig_date=box_details['signal_date']
    print(f1,f2,f3,f4)
    if f1 and f2 and f3 and f4:
        print('sell sig date:',sig_date)
        return True
    return False


def plot_all_with_breakout(ticker='ARKG', pdf=None, barcnt=300, ax=None, estimate_long_stop=True, enddate=None, short_trade=False, rticker='IWV'):

    if pdf is None:
        pdf=cu.read_quote(ticker)


    if not 'f_cup_signal' in pdf.columns:
        get_cup_features(pdf, win_size=50, get_flag=False)
    else:
        pdf['f_w50_cup_signal']=pdf['f_cup_signal']
    if not 'snr_mid' in pdf.columns:
        from datalib.patternTraderUtil import gen_vcp_signal
        gen_vcp_signal(pdf)

    pdf=get_box_tapdf(ticker, pdf=pdf)
    subdf=pdf.iloc[-barcnt:]
    if not enddate is None:
        pdf=pdf.loc[:enddate]
    
    if ax is None:
        ax=subdf.Close.plot()
    sig_df=get_box_entry_signal_df(subdf, nbars=20,ub_col='ub20', lb_col='lb20', barcnt=barcnt)
    if len(sig_df)==0:
        return None
    if short_trade:
        sig=sig_df.query('Sell>0')
    else:
        sig=sig_df.query('Buy>0')
#    print(buy_sig)
    good_dict=[]
    sig_dates=list(sig.loc[subdf.index[0]:].index)
    if estimate_long_stop:
        sig_dates.append(pdf.index[-1])
    for _ in sig_dates:
#    for _ in []:
        is_forecast=False
        if _==pdf.index[-1]:
            is_forecast=True
        ret_dict=find_box_on_plot_by_date(pdf, dateidx=_, enddate=enddate, barcnt=barcnt,  need_3pt=False, is_forecast=is_forecast, rticker=rticker)
        if ret_dict is None:
            continue
#        print(ret_dict.keys())
        if 'start_date' in ret_dict.keys():
            good_ret_dict=ret_dict
            good_dict.append(ret_dict)

    new_box_list=filter_duplicated_box(good_dict, ticker=ticker, keep_list=list(pdf.index[-2:]))
    if len(new_box_list)>0:
        box_signal_dict=draw_all_box(new_box_list, subdf, ax)
    else:
        return None
    boxout_dict={}
#    boxout_dict['stop_msg_list']=stop_msg_list
    boxout_dict['box_signal_dict']=box_signal_dict

    box_signal_df=pd.DataFrame.from_dict(box_signal_dict, orient='index')
    box_signal_df['ticker']=ticker
    box_signal_df['signame']='box_breakout'
    last_date=pdf.index[-1]
    if box_signal_df.index[-1]==last_date:
        box_signal_df.loc[last_date, 'signame']='forecast'
    boxout_dict['box_signal_df']=box_signal_df
    if len(box_signal_df)>0:
        trades_summary_df, trades_df=gen_trade_results_from_box_signal(box_signal_df, ticker, short_trade=short_trade, rticker=rticker )
        boxout_dict['trades_summary_df']=trades_summary_df
        boxout_dict['trades_df']=trades_df    
        boxout_dict['new_box_list']=new_box_list
    else:
        boxout_dict['trades_summary_df']=[]
        boxout_dict['trades_df']=[]
        boxout_dict['new_box_list']=[]
    return boxout_dict


def gen_trade_results_from_box_signal(box_signal_df, ticker, short_trade=False, rticker='IWV'):
    pdf=cu.read_quote(ticker)
    if pdf is None:
            return None
    box_signal_df['suggest_buy']=False
    box_signal_df['strong_buy']=False
    box_signal_df['suggest_sell']=False
    box_signal_df['strong_sell']=False
    box_signal_df['suggest_buy']=box_signal_df.apply(lambda x:suggest_buy(dict(x)), axis=1)
    box_signal_df['suggest_sell']=box_signal_df.apply(lambda x:suggest_sell(dict(x)), axis=1)
    box_signal_df['strong_buy']=box_signal_df.apply(lambda x:strong_buy(dict(x)), axis=1)
    box_signal_df['strong_sell']=box_signal_df.apply(lambda x:strong_sell(dict(x)), axis=1)

    box_signal_df
    from backtest.chandelierExitBacktester import chandelierExitBacktester as backtester
    if short_trade:
        trade_type='short'        
#        tradedates_df=backtester.get_ch_ex_trade_exit_date(box_signal_df.query('suggest_sell>0'), pdf, ticker=ticker)        
        #ret_dict=backtester.add_backtest_from_pred_df(ticker, box_signal_df.query('suggest_sell>0'), pred_col='suggest_sell', rticker=rticker, signame='vcp',
        ret_dict=backtester.add_backtest_from_pred_df(ticker, box_signal_df, pred_col='suggest_sell', rticker=rticker, signame='vcp',
            trade_type=trade_type, retrace_atr_multiple=3, ex_atr_bars=20, def_pct_stop=0.1, plot=False)
    else:
        trade_type='long'        
        #tradedates_df=backtester.get_ch_ex_trade_exit_date(box_signal_df.query('suggest_buy>0'), pdf, ticker=ticker)
        #ret_dict=backtester.add_backtest_from_pred_df(ticker, box_signal_df.query('suggest_buy>0'), pred_col='suggest_buy', rticker=rticker, signame='vcp',
        ret_dict=backtester.add_backtest_from_pred_df(ticker, box_signal_df, pred_col='suggest_buy', rticker=rticker, signame='vcp',
            trade_type=trade_type, retrace_atr_multiple=3, ex_atr_bars=20, def_pct_stop=0.1, plot=False)
    if not ret_dict is None:
        summary_df=ret_dict['all_tradesummary']
        trades_df=ret_dict['trades_df']
    else:
        return None, None
    if len(trades_df)==0:
        return None, None
    
#    display('xxxyyy', tradedates_df)
#    summary_df, trades_df=chandelierExitWrapper.gen_ch_ex_trades_from_tradedates(tradedates_df, pdf, ticker=ticker, trade_type=trade_type)
#    summary_df
    
    return summary_df, trades_df

def remove_duplicate_signal(idx, subdf):
    
    ret_idx=[]
    p_idx=subdf.index[0]
    for _ in idx:
        td=_-p_idx
        days=td.days
#        print('td:',days)
        if days<4:
            continue
        p_idx=_
        ret_idx.append(_)
    return ret_idx

def get_signal_table(subdf, signame='signame', colname='Close', long_idx=[], short_idx=[]):
    import numpy as np    
#    signal_df=pd.DataFrame()
#    signal_df[colname]=subdf[colname]
#    print('colname is ',colname)
    signal_df=subdf[[colname]].copy()
    signal_df['Buy']=np.nan
    signal_df['Sell']=np.nan
#    signal_df['signame']=signame
    short_idx=remove_duplicate_signal(short_idx, subdf)
    long_idx=remove_duplicate_signal(long_idx, subdf)    
    for _idx in long_idx:        
        _df=subdf.loc[_idx:]
        max_len=max(4,len(_df))
        up_idx=(_df.iloc[:max_len]).index
        signal_df.loc[up_idx, 'up']=1
    for _idx in short_idx:        
        _df=subdf.loc[_idx:]
        max_len=max(4,len(_df))
        dn_idx=(_df.iloc[:max_len]).index
        signal_df.loc[dn_idx, 'dn']=1
    signal_df.loc[short_idx, 'Sell']=1
    signal_df.loc[long_idx, 'Buy']=1
    return signal_df

#def get_box_tapdf(ticker, pdf = None, day_list=[10,15, 20,25,30,35,40,45,50]):    
def get_box_tapdf(ticker, pdf = None, day_list=[10,15, 20,25,30]):    
    #import talib
    import pandas_ta
    if pdf is None:
        pdf=cu.read_quote(ticker)
    if pdf is None:
        return None
    #pdf['atr']=talib.ATR(pdf['High'], pdf['Low'], pdf['Close'], timeperiod=25)*5
    pdf['atr']=pdf.ta.atr(25)
    for n in day_list:
        pdf[f'ub{n}']=pdf.Low.rolling(n).max()
        pdf[f'lb{n}']=pdf.Low.rolling(n).min()
        pdf[f'u1b{n}']=pdf.Low.rolling(n).max().shift(1)
        pdf[f'l1b{n}']=pdf.Low.rolling(n).min().shift(1)
    return pdf

def get_box_entry_signal_df(pdf, nbars=30, ub_col='ub', lb_col='lb', enddate=None, barcnt=280):
    s_nbars=nbars
    if not enddate is None:
        pdf=pdf.loc[:enddate]
    pdf[ub_col]=pdf.High.rolling(nbars).max()
    pdf[lb_col]=pdf.Low.rolling(s_nbars).min()
    cols=[ub_col, lb_col, 'Close']
    subdf=pdf.iloc[-barcnt:]
    #ax=subdf[cols].plot(figsize=(18,3))
    bidx=subdf.Close<subdf[lb_col].shift(1)
    short_idx=subdf[bidx].index
    bidx=subdf.Close>subdf[ub_col].shift(1)
    long_idx=subdf[bidx].index    
    if len(short_idx)==0:
        if len(long_idx)==0:
            return []
        last_signal=long_idx[-1]
    elif len(long_idx)==0:
        last_signal=short_idx[-1]
    else:
        last_signal=max(long_idx[-1], short_idx[-1])
    td_to_latest=pdf.index[-1]-last_signal
    latest_date=pdf.index[-1]
    days_th=5
    if td_to_latest.days>days_th:
        long_idx=list(long_idx)
        long_idx.append(latest_date)
    signal_df=get_signal_table(subdf, signame='box_signal', colname='Close', long_idx=long_idx, short_idx=short_idx)    
    return signal_df

def plot_all_extreama_pattern(ticker, ofname=None,  enddate=None, barcnt=280, short_trade=False, rticker='IWV'):
    import numpy as np
    from datalib.extremaPatternLooper import extremaPatternLooper
    from datalib.extremaPatternUtil import get_max_min_prices_dateidx, get_max_min_dateidx, remove_overlap_patterns, get_results
    if ofname is None:
        ofname=f'tmp/extrema_{ticker}.jpg'
    pdf=cu.read_quote(ticker)
    if pdf is None:
        return None
    if not enddate is None:
        pdf=pdf.loc[:enddate]
    plot_df_dict={}

    pdf=pdf.ffill()
    subdf=pdf.iloc[-barcnt:]
    min_max=get_max_min_prices_dateidx(subdf,  smoothing=1, window_range=10, order=2)

    pdf['r_low']=min_max.Low.rolling(3).min()
    pdf['r_high']=min_max.High.rolling(3).max()
    
    if len(subdf)<100:
        return None
    min_max['extreme']=min_max.apply(lambda x:x.High if x.minmax_type=='max' else x.Low, axis=1)    
    idx_list=min_max.index
    pdf['extreme']=np.nan
    #pdf['extreme'].plot()
    from datalib.patternTraderUtil import gen_vcp_signal, add_pbv, add_volatility_contract_by_quantile, add_volume_contraction_flag, add_vcp_features_by_pctile
    pbv2_days=120
    
    print('shape1 pdf', pdf.shape, idx_list)
    pdf.loc[idx_list, 'extreme']=min_max['extreme']
    subdf=pdf.iloc[-barcnt:]    
    subdf_with_buf=pdf.iloc[-barcnt-200:]
    print('shape1 subdf', subdf_with_buf.shape, idx_list)
    print('shape2 subdf', subdf_with_buf.shape)
    pdf=gen_vcp_signal(subdf_with_buf.iloc[-(pbv2_days+barcnt):], ticker=ticker, snr_day=pbv2_days)        
    if pdf is None:
        return 0 
#    print('shape3 with buf', pdf.shape)
    from vcplib.vcpUtil import get_cup_features
    get_cup_features(pdf, win_size=50, correct_th=0.95, get_flag=False)
    pdf['ma_10']=pdf.Close.rolling(10).mean()
    pdf['ma_25']=pdf.Close.rolling(25).mean()
    pdf['ma_50']=pdf.Close.rolling(50).mean()
    subdf=pdf.iloc[-barcnt:]        
    print(subdf.columns)
    bidx=subdf.Close>subdf.r_high.shift(1)
    long_idx=subdf[bidx].index

    bidx=subdf.Close<subdf.r_low.shift(1)
    short_idx=subdf[bidx].index

    signal_df=get_signal_table(subdf, long_idx=long_idx, short_idx=short_idx)
#    plot_cols=['extreme', 'r_low', 'r_high', 'ub', 'lb']
    plot_cols=['tm_snr_mid', 'Open', 'Close', 'Volume', 'High', 'Low', 'snr_mid', 'ma_10', 'ma_25', 'ma_50']
    signal_df=get_box_entry_signal_df(pdf, nbars=30, barcnt=barcnt)    
    subdf=pdf.iloc[-barcnt:]            
#    plot_df_dict['main']=[subdf[plot_cols], subdf[['Close', 'extreme']], signal_df]
    plot_df_dict['main']=[subdf[plot_cols], signal_df]
    #vcp_cols=['f_vola_contr', 'f_volu_contr', 'f_v2_contr', 'f_breakout_long', 'f_mmfilter', 'f_w50_cup_signal']
    vcp_cols=['f_vola_contr', 'f_volu_contr', 'f_v2_contr', 'f_w50_cup_signal']
    plot_df_dict['vcp']=subdf[vcp_cols]
#    plot_df_dict['main']=[subdf[plot_cols], subdf[['Close', 'Volume', 'High', 'Low']], signal_df]
    import pandas_ta
#    macd, signal, hist=talib.MACD(pdf.Close)
    _=pdf.ta.macd()
    macd=_['MACD_12_26_9']
    hist=_['MACDh_12_26_9']
    signal=_['MACDs_12_26_9']
    pdf['macd']=macd
    pdf['macd_signal']=signal
    pdf['macd_hist']=hist
#    pdf['obv']=talib.OBV(pdf.Close, pdf.Volume)
    pdf['obv']=pdf.ta.obv()
#    pdf['rsi']=talib.RSI(pdf.Close, 14)        
    pdf['rsi']=pdf.ta.rsi(14)
#    pdf['ad']=talib.AD(pdf.High, pdf.Low, pdf.Close, pdf.Volume) 
    pdf['ad']=pdf.ta.ad()
#    pdf['adosc']=talib.ADOSC(pdf.High, pdf.Low, pdf.Close, pdf.Volume)         
    pdf['adosc']=pdf.ta.adosc()
#    pdf['mfi']=talib.MFI(pdf.High, pdf.Low, pdf.Close, pdf.Volume)                 
    pdf['mfi']=pdf.ta.mfi()
    subdf=pdf.iloc[-barcnt:]
    plot_df_dict['macd']=subdf[['macd', 'macd_signal', 'macd_hist']]
    plot_df_dict['mfi']=subdf[['mfi']]
    plot_df_dict['adosc']=subdf[['adosc']]
    ta_min_max_dict={}
    #        print('debug div scren:', debug, remove_overlap)
    ema_list=[2]
    window_list=[4]
    local_order=1
    ex_cond='up,down'
    debug=True
    main_size=3

    def get_pat_tag_list():
        import random
        pat_tag_list=[]
        pat_tag='mfi_max_div' # mfi_bear_div
        pat_tag_list.append(pat_tag)
        pat_tag='mfi_min_div'
        pat_tag_list.append(pat_tag)    
        pat_tag='adosc_min_div' #adosc_bull_div
        pat_tag_list.append(pat_tag)    
        pat_tag='adosc_max_div' #adosc_bear_div        
        pat_tag_list.append(pat_tag)    
        pat_tag='ad_min_div' #ad_bull_div
        pat_tag_list.append(pat_tag)    
        pat_tag='ad_max_div' #ad_bear_div        
        pat_tag_list.append(pat_tag)    
        pat_tag='rsi_min_div' #rsi_bull_div
        pat_tag_list.append(pat_tag)    
        pat_tag='rsi_max_div' #rsi_bear_div        
        pat_tag_list.append(pat_tag)    
        pat_tag='obv_min_div' #obv_bull_div
        pat_tag_list.append(pat_tag)    
        pat_tag='obv_max_div' #obv_bear_div        
        pat_tag_list.append(pat_tag)    
        pat_tag='macd_min_div' #macd_bull_div
        pat_tag_list.append(pat_tag)    
        pat_tag='macd_max_div' #macd_bear_div        
        pat_tag_list.append(pat_tag)    
        random.shuffle(pat_tag_list)
        return pat_tag_list
    results_dict={}
    for ema_ in ema_list:
        for window_ in window_list: 
            close_max_min=get_max_min_prices_dateidx(subdf, smoothing=ema_, window_range=window_, order=local_order)                
            max_subset=close_max_min.query('minmax_type=="max"')
            min_subset=close_max_min.query('minmax_type=="min"')


            for col in ['obv', 'macd', 'rsi', 'ad', 'adosc', 'mfi']:
                ta_min_max_dict[col]=get_max_min_dateidx(subdf.dropna(), colname=col, smoothing=ema_, window_range=window_, order=local_order)

            pat_tag_list=get_pat_tag_list()
            if debug and False:
                print('last close points', close_max_min.index[-5:])                                    
                for k in ta_min_max_dict.keys():
                    print('last %s points' % k, ta_min_max_dict[k].index[-5:])


            for pat_tag in pat_tag_list:
                col1='Close'
                col2=pat_tag.split('_')[0]
                minmax_type=pat_tag.split('_')[1]
                signame='%s_bull_div.%s' % (col2, ex_cond)

                ta_minmax=ta_min_max_dict[col2].query(f'minmax_type=="{minmax_type}"')
                close_subset=close_max_min.query(f'minmax_type!="{minmax_type}"')
                pat=extremaPatternLooper.find_general_divergence_patterns(close_subset, ta_minmax, main_col=col1, aux_col=col2, ex_cond=ex_cond, main_size=main_size)                 
                plot_df_dict_item=[subdf[[col2]], ta_minmax[[col2]]]

                print(signame, pat)
                if len(pat)>0:
#                    print('have pat', pat)
                    results=get_results(subdf, min_subset, pat, ticker, signame, ema_, window_, local_order)
                    results_dict[pat_tag]=results
                    print(subdf.tail())
                    if minmax_type=='max':
                        sig_df=get_signal_table(subdf, signame, colname=col2, short_idx=results['entry_date']).query('Sell>0')
                    if minmax_type=='min':
                        sig_df=get_signal_table(subdf, signame,  colname=col2, long_idx=results['entry_date']).query('Buy>0')
#                    display('signdf:',sig_df)
                    plot_df_dict_item.append(sig_df)
                    plot_df_dict[signame]=plot_df_dict_item          


    pattern_func_dict={}

    from datalib.extremaPatternLooper import extremaPatternLooper
    pattern_func_dict['ihs']=extremaPatternLooper.find_ihs_patterns                                                                                                                      
    pattern_func_dict['hs']=extremaPatternLooper.find_hs_patterns
    pattern_func_dict['vcp_up']=extremaPatternLooper.find_vcp_up_patterns
    pattern_func_dict['hhl']=extremaPatternLooper.find_higher_high_low_patterns

    pattern_func_dict_h={}
    pattern_func_dict_l={}
    pattern_func_dict_h['hl']=extremaPatternLooper.find_higher_low_patterns
    pattern_func_dict_h['hh']=extremaPatternLooper.find_higher_high_patterns
    pattern_func_dict_l['lh']=extremaPatternLooper.find_lower_high_patterns
    pattern_func_dict_l['ll']=extremaPatternLooper.find_lower_low_patterns


    min_subset=min_max.query('minmax_type=="min"')
    max_subset=min_max.query('minmax_type=="max"')

    for k in pattern_func_dict:
        pat_dict=pattern_func_dict[k](min_max)
        print('min_max_pat:', pat_dict)
        if len(pat_dict)>0:
            plot_df_dict_item=[]
            results=get_results(subdf, min_max, pat_dict, ticker, signame, ema_, window_, local_order)
            results_dict[k]=results
            print('have pat', pat, results)
            long_idx=list(results['entry_date'])
            print(long_idx)
            sig_df=get_signal_table(subdf, colname='Close', long_idx=long_idx).query('Buy>0').fillna(0)        
#            display(sig_df)
            plot_df_dict_item.append(subdf[['Close']])        
            plot_df_dict_item.append(sig_df)
            plot_df_dict[f'ex_{k}']=plot_df_dict_item        

    for k in pattern_func_dict_l:
        if k[-1]=='l':
            pat_dict=pattern_func_dict_l[k](min_subset)
        else:
            pat_dict=pattern_func_dict_l[k](max_subset)            
#        print('min_pat:',pat_dict)

        if len(pat_dict)>0:
            plot_df_dict_item=[]
            results=get_results(subdf, min_max, pat_dict, ticker, signame, ema_, window_, local_order)
            results_dict[k]=results        
            print('have pat', pat, results)
            short_idx=list(results['entry_date'])
            sig_df=get_signal_table(subdf, signame,  colname='Close', short_idx=short_idx).query('Sell>0').fillna(0)
            
#            display(sig_df)
            plot_df_dict_item.append(subdf[['Close']])        
            plot_df_dict_item.append(sig_df)        
            plot_df_dict[f'ex_{k}']=plot_df_dict_item    

    for k in pattern_func_dict_h:
        if k[-1]=='l':
            pat_dict=pattern_func_dict_h[k](min_subset)
        else:
            pat_dict=pattern_func_dict_h[k](max_subset)            
#        print('max pat:', pat_dict)
        if len(pat_dict)>0:
            plot_df_dict_item=[]
            results=get_results(subdf, min_max, pat_dict, ticker, signame, ema_, window_, local_order)
            results_dict[k]=results        
            print('have pat', pat, results)
            long_idx=list(results['entry_date'])

            sig_df=get_signal_table(subdf, signame,  colname='Close', long_idx=long_idx).query('Buy>0').fillna(0)  
            
#            display(sig_df)
            plot_df_dict_item.append(subdf[['Close']])
#            xxyy
            plot_df_dict_item.append(sig_df)
            plot_df_dict[f'ex_{k}']=plot_df_dict_item
            
    ax_arr, fig=flex_plot_2(plot_df_dict, ofname=ofname, box_ticker=None, title=ticker) 
    from datalib.patternReviewUtil import plot_fib
    plot=True
    import sys
    if sys.stdin and sys.stdin.isatty():
        # running interactively
        plot=False
        print("running interactively")
    plot_fib(pdf, plot_bars=len(pdf), minmax_bars=40, ex_info=None,plot=plot, ax=ax_arr[0])

    from datalib.taStopEstimater import taStopEstimater
    stop_ndf=taStopEstimater.get_multi_stop_df(pdf, nbars=30, ax=ax_arr[0])
    for col in stop_ndf.columns:
        pdf[col]=stop_ndf[col]
    boxout_dict=plot_all_with_breakout(ticker=ticker, pdf=pdf, ax=ax_arr[0], enddate=enddate, short_trade=short_trade)
    fig.savefig(ofname, bbox_inches = "tight")
    if boxout_dict is None:
        return None
    all_ret_dict={}    
    all_ret_dict['results_dict']=results_dict
    all_ret_dict['plot_df_dict']=plot_df_dict
    all_ret_dict['boxout_dict']=boxout_dict
    all_ret_dict['ticker']=ticker
    all_ret_dict['imgfname']=ofname
    all_ret_dict['pdf']=pdf
    all_ret_dict['raw_signal_df']=signal_df
    
    return all_ret_dict


def expand_flag_pdf(pdf, exclude=[]):
    '''
        get all columns starting with f_ and treat them as flag to display on sample chart band
    '''
    df=pd.DataFrame()
    i=1.0
    for col in pdf.columns:
        if not col[:2]=='f_':
            continue
        if col in exclude:
            continue
        colname='%s_%d' % (col, i)
        df[colname]=pdf[col]*i    
        i=i+1
#        print('flag:',df.tail())
    return df

def plot_labeled(subdf, cols=None):
    if not cols is None:
        subdf=subdf[cols]
    else:
        cols=subdf.columns
    import numpy as np

    from matplotlib import pyplot as plt
    from scipy.stats import chi2, loglaplace

    from labellines import labelLine, labelLines
#    X = np.linspace(0, 1, 500)
#    A = [1, 2, 5, 10, 20]
#    funcs = [np.arctan, np.sin, loglaplace(4).pdf, chi2(5).pdf]

#    fig, axes = plt.subplots(ncols=1, nrows=1, constrained_layout=True, figsize=(8, 8))
#    axes = axes.flatten()
    ax=subdf.plot()
#    ax = axes[0]
    for col in cols:
#        ax.plot(subdf.index, subdf[col], label=str(col))
        print('subidx:',subdf.index)
#        ax.plot(subdf.index, subdf[col].values, label=str(col))

#    labelLines(ax.get_lines(), zorder=2.5)

    lines = ax.get_lines()

    labelLines(lines, yoffsets=0.01, align=False, backgroundcolor="none")
    return ax



def flex_plot_2(df_dict, txt_arr=[], title='ticker', point_list=[], 
                ofname='tmp/flexplot.jpg', ax=None, startdate=None, enddate=None, box_ticker=None):
    import numpy as np
    import logging
    import matplotlib
    matplotlib.use("Agg")
    vcp_ax=None
    vcp_df=None
    mpl_logger = logging.getLogger('matplotlib')
    mpl_logger.setLevel(logging.WARNING)
    if type(df_dict)==type(pd.DataFrame()):
        print('plot input type = df')
        cnt=len(df_dict.columns)
    else:
        print('plot input type = arr')                                                                                                                                                          
        cnt=len(df_dict.keys())


    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    #fig = plt.figure()
    top_band_size=8
    if ax is None:
        import mplfinance as mpf
#        fig=plt.figure(figsize=(18, cnt+3))
#        fig=plt.figure(figsize=(19, cnt+top_band_size))
#        s = mpf.make_mpf_style(base_mpf_style='yahoo', rc={'font.size': 10})
#        s = mpf.make_mpf_style(base_mpf_style='charles', rc={'font.size': 11})
#        s = mpf.make_mpf_style(base_mpf_style='binance', rc={'font.size': 11})
#        s = mpf.make_mpf_style(base_mpf_style='sas', rc={'font.size': 11})
        s = mpf.make_mpf_style(base_mpf_style='blueskies', rc={'font.size': 11})
        s = mpf.make_mpf_style(base_mpf_style='brasil', rc={'font.size': 11})
        s = mpf.make_mpf_style(base_mpf_style='classic', rc={'font.size': 11})
#        s = mpf.make_mpf_style(base_mpf_style='mike', rc={'font.size': 11})
#        s = mpf.make_mpf_style(base_mpf_style='nightclouds', rc={'font.size': 11})
#        s = mpf.make_mpf_style(base_mpf_style='starsandstripes', rc={'font.size': 11})
        fig=mpf.figure(figsize=(22, cnt+top_band_size), style=s)
#        fig=plt.figure(figsize=(18, 36))
        fig.tight_layout()
#        gs = gridspec.GridSpec(cnt+3,1)
        gs = gridspec.GridSpec(cnt+top_band_size,1)
        fig.subplots_adjust(hspace=0)
    ax_arr=[]
#    for rect in rect_list[::-1]:
#    print('cnt is ',cnt, gs)
    i=0
    

###        mpf.plot(subdf, type='candle',   ax=ax2, volume=ax3, show_nontrading=True)
    for k in df_dict.keys():
        df=df_dict[k]
        
#        print('i is :',i, len(df_dict.keys()))
        if i==0:
#            ax=fig.add_subplot(gs[0:i+4])
#            ax=fig.add_subplot(gs[0:i+4])
#            ax=fig.add_subplot(gs[0:top_band_size+1])
            ax=fig.add_subplot(gs[0:top_band_size])
#            ax1=fig.add_subplot(gs[0:top_band_size])
#            ax_v=fig.add_subplot(gs[top_band_size+1])
#            ax1=fig.add_subplot(gs[0:top_band_size-2])
            ax1=ax
            ax_v=fig.add_subplot(gs[top_band_size], sharex=ax1)
        
        else:
#            ax=fig.add_subplot(gs[i+3], sharex=ax1)
            ax=fig.add_subplot(gs[i+top_band_size], sharex=ax1)

            
        def trim_df(df, start, end):
            if start is None:
                start=df.index[0]
            if end is None:
#                print('a b4:',df.shape)
                df=df.loc[start:]
#                print('after:',df.shape)            
            else:
#                print('b b4:',df.shape)      
                df=df.loc[start:end]
#                print('after:',df.shape)    
            return df

        if type(df)==type([]):
            cnt=0
            marker=''
            for _ in df:
                band_df=_
                band_cols=_.columns
                cnt=cnt+1
                try:
                    trim_df(_, startdate, enddate)
                except Exception as e:
                    import traceback
                    print('exception:', e)
                    traceback.print_exc()
                    import sys,os,datetime,gzip,pickle,glob,time,traceback,random,matplotlib,subprocess,importlib,inspect
                    from debuglib.debugUtil import frame2func, save_func_input_state
                    save_func_input_state(frame2func(sys._getframe(  )),  locals())

#                if 'up' in _.columns:
#                    v_idx=_.query('up>0').index
#                    for _idx in v_idx:
#                        ax1.axvline(_idx, color='lightgreen')
#                if 'dn' in _.columns:
#                    v_idx=_.query('dn>0').index
#                    for _idx in v_idx:
#                        ax1.axvline(_idx, color='pink')

                if 'Buy' in _.columns:
                    v_idx=_.query('Buy>0').index
                    for _idx in v_idx:
                        ax.axvline(_idx, color='b')
#                        ax1.axvline(_idx, color='b')
#                        ax1.text(_idx, _.loc[_idx].Close, 'b%s' % v_idx )
                        if cnt==1:
                            marker='^'
                    v_idx=_.query('Sell>0').index
                    for _idx in v_idx:
                        ax.axvline(_idx, color='r')
#                        ax1.axvline(_idx, color='r')                        
#                        ax1.text(_idx, _.loc[_idx].Close, 's%s' % v_idx )
                        if cnt==1:
                            marker='v'
                if 'Close' in _.columns and 'Volume' in _.columns:
                    ax.set_ylim(_.Low.min()*0.99,_.High.max())
#                    mpf.plot(_, type='candle',   ax=ax1, volume=ax_v, show_nontrading=True)
#                    ax.plot(_.Close)
                    ret1=enhance_ohlc_plot(_, ax, bin_cnt=20, rwf=0.03)
                    if 'snr_mid' in _.columns:
                        width=4
                        print('now add snr mid')
                        ax.plot(_['snr_mid'], linewidth=width)
                    if 'tm_snr_mid' in _.columns:
                        width=3
                        print('now add tm snr mid')
                        ax.plot(_['tm_snr_mid'], linewidth=width, color='black')

#                    print(ret1)
#                    mpf.plot(_, type='candle',   ax=ax, volume=ax_v, show_nontrading=True)
                    mpf.plot(_, type='candle',   ax=ax1, volume=ax_v, show_nontrading=True)
                else:

                    ax.plot(_)
#                    if 'up' in _.columns:
#                        v_idx=_.query('up>0').index
#                        for _idx in v_idx:
#                            ax1.axvline(_idx, color='lightgreen')
#                    if 'dn' in _.columns:
#                        v_idx=_.query('dn>0').index
#                        for _idx in v_idx:
#                            ax1.axvline(_idx, color='pink')

                
            df=_
        else:
            trim_df(df, startdate, enddate)   
            ax.plot(df)
            if 'f_v2_contr' in df.columns:
                vcp_ax=ax
                vcp_df=df

        t='%s' % list(df.columns)
        if i==0:
            t='%s:%s' % (title, t)
        ax.set_title(t)      
        for point in point_list:
            idxs=df.loc[point:].index
            _point=idxs[0]
            ax.axvline(x=_point)
        txt=f'sig:{k}'
        
        ax.text(0.05, 0.9, txt, horizontalalignment='left',
            verticalalignment='top', transform=ax.transAxes, bbox=dict(facecolor='yellow', alpha=0.2))

#        if 'up' in df.columns:
#            v_idx=df.query('up>0').index
#            for _idx in v_idx:
#                ax.axvline(_idx, color='lightgreen')
#        if 'dn' in df.columns:
#            v_idx=df.query('dn>0').index
#            for _idx in v_idx:
#                ax.axvline(_idx, color='pink')

        flag_cols=[col for col in df.columns if col[:2]=='f_']
        print('flexi plot!!!!!!!!!!!! all flag?', df.columns, flag_cols)
        if len(flag_cols) ==len(df.columns):
            df=expand_flag_pdf(df)
        ax.plot(df)
        i=i+1
        ax_arr.append(ax)
    if not box_ticker is None:
        plot_all_with_breakout(ticker=box_ticker, ax=ax1, enddate=enddate)
        if not vcp_df is None:
        #XXX
            if vcp_df.f_v2_contr.rolling(3).max().iloc[-1]>0:
                ax.scatter(df.index[-2], 2, marker=marker, s=200)
    plt.savefig(ofname)
    print('saving flexplot to ',ofname)
    return ax_arr, fig


def plot_date_arrow(ax, x_arr, y_arr, color='b', ls='--'):
    st=x_arr[0]
    ed=x_arr[1]
    sy=y_arr[0]
    ey=y_arr[1]
    import matplotlib.dates as md
    md_td=md.date2num(ed)-md.date2num(st)
    dy=ey-sy
#    arr=ax.arrow(st,sy,md_td,dy, edgecolor='black', width=0.2, head_width=2, ls=ls)
    arr=ax.arrow(st,sy,md_td,dy, edgecolor='black', ls=ls)
    arr.set_facecolor(color)

def add_box_pbv(ret_dict, ax=None):
    import matplotlib.pyplot as plt
    box_start=ret_dict['start_date']
    box_end=ret_dict['end_date']    
    duration=(box_end-box_start).days
    box_mid=box_start+pd.to_timedelta('%s days' % int(duration/2))
    ub=ret_dict['ub']
    lb=ret_dict['lb']    
#    pdf=ret_dict['subdf']
#    box_subdf=pdf.loc[box_start:box_end]
    box_subdf=ret_dict['box_subdf']
    from datalib.boxCandlePatternUtil import boxCandlePatternUtil
    cdl_score_dict=boxCandlePatternUtil.get_multi_tf_candle_pattern_summary(box_subdf)    
    print(f'cdl score dict:{cdl_score_dict}', cdl_score_dict.keys())
    net_cdl_score=cdl_score_dict['net_cdl_score']
#    cdl_score_dict={'net_cdl_score':0}
#    net_cdl_score=0
    close_price=box_subdf.iloc[-1].Close
    print(box_subdf.head())
    print(box_subdf.columns)
    box_ta_dict=get_box_ta_matrix(box_subdf, cdl_score_dict)
    ta_score=box_ta_dict['score']
    rs_plus=box_ta_dict['rs_plus']    
    snr_mid_plus=box_ta_dict['snr_mid_plus']        
    if not ax is None:
        volu_bias=0
        if 'volu_bias' in ret_dict.keys():
            volu_bias=ret_dict['volu_bias']

            box_stop=ret_dict['box_stop']            
            box_stop_adj=ret_dict['box_stop_adj']                        
            box_stop_idx=ret_dict['box_stop_idx']                        
            box_stop_adj_idx=ret_dict['box_stop_adj_idx'] 
            box_ta_dict['box_stop_adj']=box_stop_adj
            box_ta_dict['box_stop_idx']=box_stop_idx            
            box_ta_dict['box_stop_adj_idx']=box_stop_adj_idx                        
            box_ta_dict['box_stop']=box_stop
            box_ta_dict['duration']=duration            

        box_ta_dict['recent_vol_bias_plus']=volu_bias
        fmt='--'
        color='gray'    
        ax.plot_date([box_start, box_start], [ub, lb], fmt=fmt, color=color)
        ax.plot_date([box_end, box_end], [ub, lb], fmt=fmt, color=color)
        ax.plot_date([box_start, box_end], [ub, ub], fmt=fmt, color=color)
        ax.plot_date([box_start, box_end], [lb, lb], fmt=fmt, color=color)
        msg_list=[]
        if not volu_bias==0 or box_stop_idx>0.5:
            fmt='-'
            ax.plot_date([box_start, box_start], [ub, lb], fmt=fmt, color=color)
            ax.plot_date([box_end, box_end], [ub, lb], fmt=fmt, color=color)

        
        ax.plot_date([box_start, box_end], [box_stop, box_stop], fmt='--', color='black')

        if volu_bias>0:
            color='green'
            ax.plot_date([box_end, box_end], [lb, ub], fmt='^-', color=color, linewidth =2.2)
            msg_list.append('recent volu_bias up')
        if volu_bias<0:
            color='red'
            ax.plot_date([box_end, box_end], [ub, lb], fmt='v-', color=color, linewidth =2.2)
            msg_list.append('recent volu_bias down')
        if box_stop_idx>=0.5:
            color='green'            
            ax.plot_date([box_start, box_start], [lb, ub], fmt='^-', color=color, linewidth =2.2)        
            msg_list.append('box_stop_idx>=0.5')
        if box_stop_idx<0.5:
            color='red'            
            ax.plot_date([box_start, box_start], [lb, ub], fmt='v-', color=color, linewidth =2.2)        
            msg_list.append('box_stop_idx<0.5')            
            
#            ax.plot_date([box_start, box_end], [ub, lb], fmt='-', color='red', linewidth =4)        
        print(f'xxxx {box_stop_adj_idx},{box_stop_idx}, {volu_bias} duration{duration} tascore:{ta_score}' )
        ax.plot_date([box_start, box_end], [box_stop, box_stop_adj], fmt='^-', color='orange', linewidth =2.2)
        if rs_plus>0:
            ax.plot_date([box_start, box_mid], [lb, ub], fmt='^-', color='green', linewidth =1.5)     
            msg_list.append('rs plus bias up')                        
        if rs_plus<=0:
            ax.plot_date([box_start, box_mid], [ub,lb], fmt='v-', color='red', linewidth =1.5)            
            msg_list.append('rs plus bias down')                                    
        if snr_mid_plus>0:
            ax.plot_date([box_mid, box_mid], [lb, ub], fmt='^-', color='green', linewidth =2.2)     
            msg_list.append('snr midplus bias up')                        
        if snr_mid_plus<=0:
            ax.plot_date([box_mid, box_mid], [ub,lb], fmt='v-', color='red', linewidth =2.2)            
            msg_list.append('snr midplus bias down')                                    
        up_th=5
        dn_th=3
        if ta_score>up_th:
#            ax.plot_date([box_mid, box_end], [lb, ub], fmt='^-', color='green', linewidth =2)     
            plot_date_arrow(ax, [box_mid, box_end], [lb,ub], color='green', ls='--')
            msg_list.append('ta score bias up %s' % box_ta_dict)
        if ta_score<dn_th:
#            ax.plot_date([box_mid, box_end], [ub,lb], fmt='v-', color='red', linewidth =2)            
            plot_date_arrow(ax, [box_mid, box_end], [ub,lb], color='red', ls='--')
            msg_list.append('ta score bias down %s' % box_ta_dict)   

        if net_cdl_score>0 and snr_mid_plus>0 and ta_score>up_th and rs_plus>0:
            print('D1DDDDDDDDDDDDDDDDDDDDDDDDDDD double vol bias up!')
            #ax.plot_date([box_start, box_end], [lb, ub], fmt='^-', color='blue', linewidth =3)        
            #ax.plot_date([box_start, box_end], [lb, close_price], fmt='^-', color='blue', linewidth =3)        
            plot_date_arrow(ax, [box_start, box_end], [lb,close_price], color='blue', ls='--')
            msg_list.append('both and recent vol box_stop_idx bias down')            

        if net_cdl_score<0 and snr_mid_plus<0 and ta_score<dn_th and rs_plus<0:
            print('D3DDDDDDDDDDDDDDDDDDDDDDDDDD double vol bias down!')
#            ax.plot_date([box_start, box_end], [ub, lb], fmt='v-', color='red', linewidth =3)  
#            ax.plot_date([box_start, box_end], [ub, close_price], fmt='v-', color='red', linewidth =3)  
            plot_date_arrow(ax, [box_start, box_end], [ub,close_price], color='red', ls='--')
            msg_list.append('both and box_stop_idx bias down')            
                    
            
    from vcplib.vcpUtil import vcpUtil as vcu 
#    print(' zzzzzzzzzzzzz box_subdf shape:', box_subdf.shape, box_subdf.index[0], box_subdf.index[-1], ret_dict['start_date'], ret_dict['end_date'])
    if len(box_subdf)<2:
        print('too few rows for box_subdf, input is %s ' % ret_dict)
    x=vcu.get_price_by_volume_obv_df(box_subdf, bin_cnt=6)
    max_v=x.volume_by_price.max()
    box_length=0.1

    def _hline(y, xmin=None, xmax=None, color='red'):
        if xmin is None or xmas is None:
            return None
        if not ax is None:
            ax.axhline(y, xmin, xmax=xmax, color=color)        
        return None
    for price_level in x.index:

        price_level_f=float(price_level)*0.95
        volume_by_price=x.loc[price_level,'volume_by_price']
        if volume_by_price==max_v:

            _hline(y=price_level_f, color='green')
            stop_level=price_level
        pbv_up=x.loc[price_level, 'pbv_up']
        pbv_down=x.loc[price_level, 'pbv_down']        
        r_length=x.loc[price_level,'volume_by_price']/max_v*box_length*0.9
        sub_ratio=pbv_up/volume_by_price

    box_ret_dict={}
    box_ret_dict['desc']=msg_list
    box_ret_dict['stop_level']=float(stop_level)
    
    box_ret_dict.update(box_ta_dict)
    box_ret_dict.update(cdl_score_dict)
    return box_ret_dict

from datalib.patternReviewUtil import quick_pbv_breakout_review_extra

def add_price_by_volume_obv_plot_from_df_v2(pbv_df, ax, subdf, near_term=False, on_right=False):
    pbv_max=pbv_df.volume_by_price.max()
    _=list(pbv_df.index)
#    print('pbv index 1:',pbv_df.index)
    float_idx=[float(v) for v in _]
#    print('pbv index 2:',float_idx)
    band_max=max(float_idx)
    band_min=min(float_idx)
    width=(band_max-band_min)/len(float_idx)*1*0.4
    width2=(band_max-band_min)/len(float_idx)*1*0.9*2
    width=10
#    print('pbv_min:', width2,  width, band_max, band_min)

    alpha=0.3
    color='b'
    if near_term:
        alpha=0.5
        color='g'

    for k,row in pbv_df.iterrows():

        length=row.volume_by_price*0.4/pbv_max
        up_length=row.pbv_up*0.4/pbv_max
        down_length=row.pbv_down*0.4/pbv_max
#        print(length, up_length, down_length)
        ki=float(k)        
        if on_right:
            ax.axhline(y=ki, xmin=1-length, xmax=1, linewidth=width, alpha=alpha, color='red')
            ax.axhline(y=ki, xmin=1-up_length, xmax=1, linewidth=width, alpha=alpha, color='green')                    
        else:
            ax.axhline(y=ki, xmax=length, linewidth=width, alpha=alpha, color='r')
            ax.set_xlim(subdf.index[0], subdf.index[-1])
            ax.axhline(y=ki, xmax=up_length, linewidth=width, alpha=alpha, color='g')
#            ax.text

#        print(k,row.volume_by_price, length, width)
    return pbv_df

def plot_rescale_volume(pdf, ax, height=0.3):
    
    max_yc=pdf.High.max()
    min_yc=pdf.Low.min()
    max_v=pdf.Volume.max()
    min_v=pdf.Volume.min()
    v_height=max_v-min_v
    if v_height==0:
        v_height=100
    c_height=max_yc-min_yc
    scale=0.3
    def map_vol_to_close_height(x):
        ret=(x-min_v)/v_height*scale*c_height+min_yc
        return ret
    pdf['v2']=pdf.Volume.apply(lambda x:map_vol_to_close_height(x))
    cols=['Volume', 'Close', 'v2']
    #display(pdf[cols])
    ax.fill_between(pdf.index, pdf.v2, pdf.v2.min(), color='lightgray')    
    return pdf

def enhance_ohlc_plot(subdf, ax, bin_cnt=20, rwf=0.02):
    from vcplib.vcpUtil import vcpUtil as vcu 
    left_pbv_df=vcu.get_price_by_volume_obv_df(subdf, bin_cnt=bin_cnt)
    right_pbv_df=vcu.get_price_by_volume_obv_df(subdf, bin_cnt=bin_cnt, rwf=0.05)
#    print(left_pbv_df)
#    print(right_pbv_df)
    add_price_by_volume_obv_plot_from_df_v2(left_pbv_df, ax, subdf)
    add_price_by_volume_obv_plot_from_df_v2(right_pbv_df, ax, subdf, on_right=True)
    ret=plot_rescale_volume(subdf, ax, height=0.3)
    ret_dict={}
#    ret_dict['ret']=ret
    ret_dict['left_pbv_df']=left_pbv_df
    ret_dict['right_pbv_df']=right_pbv_df    
    ret_dict['subdf']=subdf
    
    return ret_dict

class extremaPlotter:

    @staticmethod
    def plot_all_extreama_pattern(ticker, ofname='tmp/pattern.jpg',  enddate=None, barcnt=280, short_trade=False, rticker='IWV'): 
        ret=plot_all_extreama_pattern(ticker, ofname,  enddate, barcnt, short_trade, rticker=rticker) 
        return ret 
    @staticmethod    
    def plot_all_with_breakout(ticker='ARKG', pdf=None, barcnt=300, ax=None, estimate_long_stop=True, enddate=None, short_trade=False):
        ret=plot_all_with_breakout(ticker, pdf, barcnt, ax, estimate_long_stop, enddate, short_trade)
        return ret

    @staticmethod    
    def flex_plot_2(df_dict, txt_arr=[], title='ticker', point_list=[],
                ofname='tmp/flexplot.jpg', ax=None, startdate=None, enddate=None, box_ticker=None):
        ret=flex_plot_2(df_dict, txt_arr,  title, point_list, ofname, ax, startdate, enddate, box_ticker)
        return ret
    
    @staticmethod
    def view_stk_struct_suggestions(stk_struct):
        stk_struct['boxout_dict'].keys()
        suggest_cols=['suggest_buy', 'strong_buy', 'suggest_sell', 'strong_sell', 'snr_mid_plus', 'rs_plus']
        ndf=stk_struct['boxout_dict']['box_signal_df'][suggest_cols].copy()
        ndf['ticker']=stk_struct['ticker']
        return stk_struct['boxout_dict']['box_signal_df'][suggest_cols]    

def test():
    #from datalib.extremaPlotter import plot_all_extreama_pattern
    ticker='TSLA'
#    ticker='AR'
#    all_ret_dict=plot_all_extreama_pattern(ticker, ofname='tmp/pattern.jpg')
    all_ret_dict=plot_all_extreama_pattern(ticker, ofname='tmp/pattern.jpg', short_trade=True)    
    cmd='telegram-send --image tmp/pattern.jpg'
#test()

def test2():

    ticker='IPG'
    stk_struct=plot_all_extreama_pattern(ticker, ofname='tmp/pattern.jpg', enddate='20210521')
    box_signal_df=stk_struct['boxout_dict']['box_signal_df']
    trades_summary_df=stk_struct['boxout_dict']['trades_summary_df']    
    trades_df=stk_struct['boxout_dict']['trades_df']        
    print(trades_summary_df)

