import pandas as pd
import numpy as np
import os,pickle,gzip,time
from backtest.chandelierExitBacktester  import chandelierExitBacktester as ceb 
from datalib.commonUtil import commonUtil as cu
from vcplib.vcpUtil import get_signal_cross

#from datalib.alphaDivergenceProfileReviewer import alphaDivergenceProfileReviewer
#from datalib.sharpe_stock_info import getFullYahooEOD
#from datalib.multiBreadthWorkflow import get_cup_features
from datalib.patternTraderUtil import gen_vcp_signal
import pandas as pd

def summarise_atr_based_exit_trade_from_features(subdf, ticker='ARKK', rticker='SPY', trade_type='long',
                                               signame='f_w50_cup_signal'):
    pdf=subdf
    bidx=pdf[signame].diff(1)>0
    trade_dates=pdf[bidx].index
    getFullYahooEOD(ticker)
    ret=alphaDivergenceProfileReviewer.get_div_profile_info_table(ticker, rticker=rticker, ret_th=0, rs_th=1, div_th=0.1)
#    ret['alpha_div_summary'].T
    r2=alphaDivergenceProfileReviewer.test_trade_signal_with_atr_rules(ticker, trade_dates)
    if r2 is None:
        return []
    r2['rticker']=rticker

    rprice_df=cu.read_quote(rticker)
    from sig_matrix.matrixTraderUtil import tradeDateBackTester, signalTradeReviewer
    tdbt=tradeDateBackTester() 
    trades_df=tdbt.backtest_from_tradedates_df(r2, pdf, rprice_df, trade_type=trade_type)
    srt=signalTradeReviewer()
    if len(trades_df)==0:
        return []
    tradesummary=srt.summarise_trades(trades_df)
    summary_dict={k:v for k,v in tradesummary.items() if not type(v)==type(pd.DataFrame())}
    xdf=pd.DataFrame.from_dict({ticker:summary_dict}, orient='columns')
    xdf
    return xdf


def test_ticker_vcp_cup_atr_exit_perf(ticker='ASML', barcnt=400, rticker='SPY'):
    benchmark_col=['exposure_adjusted_annualized_gain', 'exposure_adjusted_annualized_gain_per_mean_drawdown',
                   'trade_cnt', 'win_trade_cnt', 'sum_pct_profit', ]    
    ret_dict={}
    
    pdf=cu.read_quote(ticker)
    if pdf is None:
        return ret_dict
    pdf=pdf.iloc[-barcnt:]
    pdf=get_cup_features(pdf, get_flag=False)
    subdf=gen_vcp_signal(pdf)    
    if subdf is None:
        return None
#    display(subdf2)
    signame1='f_w50_cup_signal'
    signal_perf_df=summarise_atr_based_exit_trade_from_features(subdf, ticker, rticker, signame=signame1)
    signal_perf_df
    if len(signal_perf_df)>0:
        print('columns are:',signal_perf_df.columns, signal_perf_df.index)
        ret_dict[f'{ticker}_cup']=dict(signal_perf_df.loc[benchmark_col, ticker])        
    signame2='f_breakout_comb'
    signal_perf_df=summarise_atr_based_exit_trade_from_features(subdf, ticker, rticker, signame=signame2)
    signal_perf_df
    if len(signal_perf_df)>0:    
        ret_dict[f'{ticker}_vcp']=dict(signal_perf_df.loc[benchmark_col, ticker])
    signame3='xcp_signal'
    subdf[signame3]=subdf[signame1].rolling(15, min_periods=1).max()*subdf[signame1].rolling(15, min_periods=1).max()
    signal_perf_df=summarise_atr_based_exit_trade_from_features(subdf, ticker, rticker, signame=signame3)
    signal_perf_df
    if len(signal_perf_df)>0:    
        ret_dict[f'{ticker}_xcp']=dict(signal_perf_df.loc[benchmark_col, ticker])
    
    return ret_dict

def batch_test_vcp_cup_atr_exit_perf(tlist, barcnt=500):
    barcnt=500
    all_data_dict={}
    for ticker in tlist:
        ret_dict=test_ticker_vcp_cup_atr_exit_perf(ticker, barcnt=barcnt)        
        if ret_dict is None:
            continue
        all_data_dict.update(ret_dict)

    df=pd.DataFrame.from_dict(all_data_dict, orient='index')
    print(df)
#    ret=df.T.sort_values(by='exposure_adjusted_annualized_gain', ascending=False)
    return df


def get_parabolic_sar_long_exit(pdf, ret_details=False,  af=0.02, maximum=0.2, atrbuff=0.5):
    ret_df=pdf.copy()
    import pandas_ta
#    ret_df['SAR'] = talib.SAR(pdf.High, pdf.Low, acceleration=af, maximum=maximum)    
#    ret_df['SAR'] = pdf.ta.psar(af, max_af=maximum)['PSARl_0.02_0.2']   
#    ret_df['SAR'] = 
    #_=pdf.ta.psar(af, max_af=maximum)
    _=pdf.ta.psar()
    ret_df['SAR']=_['PSARl_0.02_0.2']   

    atr_bars=14    
#    ret_df['ATR']=talib.ATR(pdf.High, pdf.Low, pdf.Close, atr_bars)   
    ret_df['ATR']=pdf.ta.atr(atr_bars)
    ret_df['sar_stop']=ret_df.eval('SAR-ATR*%s' % atrbuff)
    if ret_details:
        return ret_df
    else:
        return ret_df['sar_stop']
    
def get_ma_and_atr_trailing_exit(pdf, ma=50, n_atr=0.9, atr_bars=14, ret_details=False):
    import pandas_ta
    ret_df=pdf.copy()
#    ret_df['ATR']=talib.ATR(pdf.High, pdf.Low, pdf.Close, atr_bars)   
    ret_df['ATR']=pdf.ta.atr(atr_bars)
    ret_df['mid_ma']=ret_df.Close.rolling(ma, min_periods=10).mean()
#    ret_df['mid_ma']=ret_df.Close.rolling(ma, min_periods=10).max()    
    signame='ma_atr_trail_stop'
    ret_df[signame]=ret_df.eval('mid_ma+ATR*%s' % n_atr)
    cols=['Close', signame]
#    ret_df[cols].iloc[-300:].plot(figsize=(15,5))    
    
    if ret_details:
        return signame, ret_df.iloc[-400:]
    else:
        return ret_df[stop_col]
    
def get_ma_and_atr_trailing_exit(pdf, ma=50, n_atr=0.9, atr_bars=14, ret_details=False):
    import pandas_ta
    ret_df=pdf.copy()
#    ret_df['ATR']=talib.ATR(pdf.High, pdf.Low, pdf.Close, atr_bars)   
    ret_df['ATR']=pdf.ta.atr(atr_bars)
    ret_df['mid_ma']=ret_df.Close.rolling(ma, min_periods=10).mean()
#    ret_df['mid_ma']=ret_df.Close.rolling(ma, min_periods=10).max()    
    signame='ma_atr_trail_stop'
    ret_df[signame]=ret_df.eval('mid_ma+ATR*%s' % n_atr)
    cols=['Close', signame]
#    ret_df[cols].iloc[-300:].plot(figsize=(15,5))    
    
    if ret_details:
        return signame, ret_df.iloc[-400:]
    else:
        return ret_df[signame]
    
def get_ma_and_atr_trailing_exit_signal(pdf, ma=50, n_atr=0.9, atr_bars=14, ret_details=False, smooth_bars=3):    
    stop_col, ret_df=get_ma_minus_atr_trailing_exit(pdf, ma=ma, n_atr=n_atr, atr_bars=atr_bars, ret_details=True)
    signame='%s_exit_signal' % stop_col
    ret_df[signame]=(get_signal_cross(ret_df, 'Close', stop_col)<0)*1.0
#    cols=['Close', stop_col, signame]
    cols=[signame]
    ret_df[cols].iloc[-400:].plot(figsize=(15,2))
    cols2=[stop_col, 'Close']
    ret_df[cols2].iloc[-400:].plot(figsize=(15,2))    
    if ret_details:                                                                                                                                                                  
        ret_df[signame]=ret_df[signame].rolling(smooth_bars).max()
        return signame, ret_df
    else:
        return signame, ret_df[signame].rolling(smooth_bars).max()        

    


def get_ma_minus_atr_trailing_exit(pdf, ma=50, n_atr=0.9, atr_bars=14, ret_details=False):
    import pandas_ta
    ret_df=pdf.copy()
    #ret_df['ATR']=talib.ATR(pdf.High, pdf.Low, pdf.Close, atr_bars)   
    ret_df['ATR']=pdf.ta.atr(atr_bars)
    ret_df['mid_ma']=ret_df.Close.rolling(ma, min_periods=10).mean()
    stop_col='ma_atr_trail_stop'
    ret_df[stop_col]=ret_df.eval('mid_ma+ATR*%s' % n_atr)
    if ret_details:
        return signame, ret_df
    else:
        return ret_df[stop_col]
    
def get_ma_plus_atr_trailing_exit(pdf, ma=20, n_atr=3, atr_bars=14, ret_details=False):
    import pandas_ta
    ret_df=pdf.copy()
    #ret_df['ATR']=talib.ATR(pdf.High, pdf.Low, pdf.Close, atr_bars)   
    ret_df['ATR']=pdf.ta.atr(atr_bars)
    ret_df['short_ma']=ret_df.Close.rolling(ma, min_periods=10).mean()
    stop_col='ma_atr_climax_stop'    
    ret_df[stop_col]=ret_df.eval('short_ma+ATR*%s' % n_atr)

    if ret_details:
        return stop_col, ret_df
    else:
        return ret_df[stop_col]    
    
def get_bb_related_exit_and_stop(pdf, nbars=30, ax=None, atr_bars=14):
    import pandas_ta
    #import talib
    nbars=30
    ma=pdf.Close.rolling(nbars).mean()
    std=pdf.Close.rolling(nbars).std()
    #atr=talib.ATR(pdf.High, pdf.Low, pdf.Close, 50)
    atr=pdf.ta.atr(atr_bars)
    #std=atr
    u2factor=2.2
    bbu1=ma+std*1
    bbu2=ma+std*u2factor
    bbl1=ma-std*1
    bbl2=ma-std*u2factor
    if not ax is None:
        ax.plot(ma)
#        ax.plot(bbu1)
#        ax.plot(bbu2)
#        ax.plot(bbl1)
#        ax.plot(bbl2)
        bidx=pdf.Close>bbu1
        idx=pdf[bidx].index
        print(idx)
        ax.plot(pdf.Close)
        #for b in idx:
        #ax.fill_between(idx, bbu2.loc[idx], bbu1.loc[idx], )
        ax.fill_between(bbu2.index, bbu2, bbu1, where=bidx, color='pink', alpha=0.5)


    bb_hold_long_flag=pdf.Close>bbu1
    bb_hold_short_flag=pdf.Close<bbl1
    bb_hold_long_trail_exit=pdf.Close+std*0.5
    bb_hold_short_stop_exit=pdf.Close-std*0.5
    dfta=pd.DataFrame()
    dfta['bbu1']=bbu1
    dfta['bbu2']=bbu2
    dfta['bbl1']=bbl1
    dfta['bbl2']=bbl2
    dfta['bb_hold_long_flag']=bb_hold_long_flag*1.0
    dfta['bb_hold_short_flag']=bb_hold_short_flag*1.0  
    dfta['bb_ma_exit']=ma
    return dfta    



class taStopEstimater:
    
    @staticmethod
    def get_multi_stop_df(pdf, nbars=30, ax=None):
        bb_stop_df=get_bb_related_exit_and_stop(pdf, nbars, ax)
        sar_exit=get_parabolic_sar_long_exit(pdf)    
        atr_trail_exit=get_ma_and_atr_trailing_exit(pdf)
        atr_minus_exit=get_ma_minus_atr_trailing_exit(pdf)
        nday_low_exit=pdf.Close.rolling(nbars, min_periods=5).min().shift(1)

        signame, _=ceb.get_chandelier_long_short_exit_signal(pdf, atr_bars=nbars, retrace_atr_multiple=1,  smooth_bars=3, ret_details=True)

        atr_plus_exit=get_ma_plus_atr_trailing_exit(pdf)


        ndf=bb_stop_df
        ndf['chand_exit']=_['chand_ex']
        ndf['atr_plus_exit']=atr_plus_exit
        ndf['atr_minus_exit']=atr_minus_exit
        ndf['nday_low_exit']=nday_low_exit
        ndf['sar_exit']=sar_exit
        ndf['Close']=pdf.Close
        long_stop_exit_cols=['chand_exit', 'sar_exit', 'atr_minus_exit']
        ndf['trail_long_exit']=ndf[long_stop_exit_cols].apply(lambda x: max(x), axis=1)
        profit_exit=get_signal_cross(ndf, 'Close', 'atr_plus_exit')
#        trail_exit=get_signal_cross(ndf,  'sar_exit', 'Close')
        trail_exit=get_signal_cross(ndf,  'trail_long_exit', 'Close')
        bidx=profit_exit>0
        idx1=profit_exit[bidx].index

        bidx=trail_exit>0
        idx2=trail_exit[bidx].index
        join_exit=pdf
        if 'snr_mid' in pdf:
            ndf['snr_mid_exit']=pdf.snr_mid
        if not ax is None:
    #        bidx=pdf.Close>sar_exit
    #        ax.fill_between(atr_plus_exit.index, pdf.Close, sar_exit, where=bidx)        
            ex_idx=pdf[(pdf.Close>atr_minus_exit)].index
            bidx=(pdf.Close>atr_minus_exit)
            
            ax.fill_between(atr_plus_exit.index, pdf.Close, atr_minus_exit, where=bidx, alpha=0.5, color='lightgray')        
#            ax.fill_between(ex_idx, pdf.Close, atr_minus_exit, where=ex_idx, alpha=0.4, color='lightgray')        
            #ex_idx=pdf[(pdf.Close<=atr_minus_exit)].index
            ex_idx=pdf[(pdf.Close<=pdf.snr_mid)].index
            bidx=(pdf.Close<=atr_minus_exit)
            ax.fill_between(atr_plus_exit.index, atr_minus_exit, pdf.Close,  where=bidx, alpha=0.3, color='red')        
#            ax.fill_between(ex_idx, atr_minus_exit, pdf.Close,  where=ex_idx, alpha=0.2, color='red')        
#            ax.fill_between(ex_idx, pdf.snr_mid, pdf.Close,  where=ex_idx, alpha=0.2, color='red')        

#            ax.plot(atr_plus_exit)
#            ax.plot(sar_exit)    
#            ax.plot(atr_trail_exit)
#            ax.plot(atr_minus_exit)    
#            ax.plot(ndf['chand_exit'])    
            for d in idx1:
                ax.axvline(d, color='green')

            for d in idx2:
                ax.axvline(d, color='blue')
            ax.plot(pdf.High, lw=2, ls='--', color='blue')

        return ndf

def test2():
    import pandas as pd

    pdf=cu.read_quote('TSM')
    pdf=pdf.iloc[-300:]
    ax=pdf.Close.plot(figsize=(18,8))
    taStopEstimater.get_multi_stop_df(pdf, nbars=30, ax=ax)
    
def test():
    _1=all_struct_dict['A']['boxout_dict']['new_box_list']['A_20210412_20210528']
    _2=all_struct_dict['F']['boxout_dict']['new_box_list']['F_20210319_20210528']
    subdf1=_1['box_subdf']
    subdf2=_2['box_subdf']
    subdf1.columns
    fname='tmp/adhoc_forecast.csv'
    #fname='tmp/tech_forecast.csv'
    df=pd.read_csv(fname, index_col=0)
    #display(df)
    subdf=df.query('signame=="forecast"').copy()
    subdf.set_index('ticker', inplace=True)
    stop_cols=['stop_level', 'box_stop_adj', 'snr_mid', 'tm_snr_mid']
    #highest_stop=df.apply(lambda x:x[stop_cols].argmax(), axis=1)
    highest_stop=subdf.apply(lambda x:[max(x[stop_cols]), min(x[stop_cols] )], axis=1)
    xdf=subdf.apply(lambda x:pd.Series({'max_stop':max(x[stop_cols]), 'min_stop':min(x[stop_cols]), 'Close':x['Close']}), axis=1)
    for col in xdf.columns:
        subdf[col]=xdf[col]

    t='3690.hk'
    pdf=cu.read_quote(t).iloc[-300:]
    signame, _=ceb.get_chandelier_long_short_exit_signal(pdf, atr_bars=20, retrace_atr_multiple=1,  smooth_bars=3, ret_details=True)
    subdf['chand_ex']=0
    signal_date=subdf.loc[t, 'signal_date']
    signal_date
    subdf.loc[t,'chand_ex']=_.loc[signal_date, 'chand_ex']
    subdf
    ax=_.Close.plot(figsize=(18,8))
#ax.plot(_.chand_ex)


    ndf=get_multi_stop_df(pdf, nbars=30, ax=ax)
    ndf

#test2()
def test_signal_perf():
    tlist=['ARKK', 'TSM', 'ASML', 'ARKG', 'BLD.ax', 'FBU.ax', 'A2M.ax', 'CBA.ax']
    df=batch_test_vcp_cup_atr_exit_perf(tlist, barcnt=500)
    print(df)
    return df
#test_signal_perf()
