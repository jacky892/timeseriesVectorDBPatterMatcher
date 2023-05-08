# tdoc breakout after mid 2020-Jan with VCP
import matplotlib
matplotlib.use("Agg")
#from macrolib.mktBreadthProcessor import gen_rs_macro_feats 
#from datalib.priceByVolumeUtil import add_price_by_volume_plot, get_price_by_volume_ranged, get_price_by_volume_df, get_price_by_volume_obv_df
from vcplib.vcpUtil import  get_price_by_volume_df, get_price_by_volume_obv_df
#from datalib.momentumMetricsUtil import get_signal_cross
#from datalib.taxonomyUtil import read_tlist
from datalib.commonUtil import commonUtil as cu
import pandas as pd
pd.options.mode.chained_assignment = None 
import numpy as np

import matplotlib.pyplot as plt

def get_signal_cross(signal_df, col1, col2):
    if col2 not in signal_df.columns:
        signal_df['_%s' % col2]=col2
        col2='_%s' % col2
    _=(signal_df[col1]-signal_df[col2])
    cross=(_*_.shift(1)<0)*1
    cross_sign=(_>0)*cross-(_<0)*cross
    return cross_sign


def add_volatility_contract_by_quantile(pdf, ma_days=15, percentile=0.15, timeperiod=6, lookback_days=40, smooth_days=5, plot=False):
    '''
    add the hv_flag_n based on percentle of atr for n period ATR (atr)
    hv_rflag_n  based on relative atr (atr/Close_ma6)
    flag/rflag are flag not yet smoothed with rolling max

    '''
    #import talib
    import pandas_ta
#    pdf.loc[:,'atr']=talib.ATR(pdf['High'], pdf['Low'], pdf['Close'], timeperiod=timeperiod)
    try:
#        tmp=talib.ATR(pdf['High'], pdf['Low'], pdf['Close'], timeperiod=timeperiod)
        tmp=pdf.ta.atr(timeperiod)
    except Exception as e:
        print('error getting pandas ta atr attributes ', e)
        return None
#    print(tmp, pdf.index)
#    pdf.loc[tmp.index, 'atr']=tmp.values
    pdf.drop_duplicates(inplace=True)
    idx=tmp.index.intersection(pdf.index)
    try:
        pdf.loc[idx, 'atr']=tmp.loc[idx].values
    except Exception as e:
        import traceback
        print('exception:', e)
        traceback.print_exc()
        print(tmp.index, tmp.values, pdf.shape)
        return None
    pdf.loc[:,'rolling_atr_low']=pdf['atr'].rolling(lookback_days, min_periods=1).quantile(percentile)

    #pdf['atr_q10']=pdf['atr'].pct_change(6).rolling(50, min_periods=1)
#    from datalib.patternTraderUtil import flex_plot, gen_vcp_signal
    pdf['price_ma']=pdf.Close.rolling(ma_days, min_periods=1).mean()
    pdf['rolling_atr_low2']=pdf['atr'].rolling(lookback_days*2, min_periods=1).quantile(percentile)
#    pdf['f_vola_contr']=pdf.eval('atr<rolling_atr_low and Close>price_ma')*1.0
    pdf['f_vola_contr']=pdf.eval('(atr<rolling_atr_low or atr<rolling_atr_low2 )and Close>price_ma')*1.0
    cols=['atr', 'rolling_atr_low', 'f_vola_contr']
    pdf['f_vola_contr']=pdf['f_vola_contr'].rolling(smooth_days, min_periods=1).max()
    cols=['atr', 'rolling_atr_low', 'f_vola_contr']

    return cols

def add_volume_contraction_flag(pdf, ma_days=3,  percentile=0.1, smooth_days=3, lookback_days=80):
    '''
    return the set of cols added in the process, the last one being the flag
    '''
    pdfx=pdf
    pdfx.loc[:,'volume_ma']=pdfx.Volume.rolling(ma_days, min_periods=1).mean()
    pdfx['volume_max']=pdfx.Volume.rolling(ma_days, min_periods=1).mean()
    pdfx['volume_max_low']=pdfx['volume_max'].rolling(lookback_days, min_periods=1).quantile(percentile)
    pdfx['f_volu_contr']=pdfx.eval('volume_ma<volume_max_low')*1.0
    # smoothing for 5 days
    pdfx['f_volu_contr']=pdfx['f_volu_contr'].rolling(smooth_days, min_periods=1).max()
    cols=[ 'volume_ma', 'volume_max', 'volume_max_low', 'f_volu_contr']
    #pdfx[cols].iloc[-400:].plot()
    return cols

def add_vcp_features_by_pctile(ticker, pdf=None, volu_pctile=0.25, vola_pctile=0.25, smooth_days=5):
    if pdf is None:
        pdf=cu.read_quote(ticker)
    vola_cols=add_volatility_contract_by_quantile(pdf=pdf, smooth_days=smooth_days, percentile=vola_pctile)
    if vola_cols is None:
        return None, None
        
#    print(pdf.columns)
    volu_cols=add_volume_contraction_flag(pdf=pdf, smooth_days=smooth_days, percentile=volu_pctile)
    if len(vola_cols)==0 or len(volu_cols)==0:
        print('no volu or vola contraction!')
    else:
        qstr='%s>0 and %s>0' % (vola_cols[-1],volu_cols[-1])
        flag_cols=[vola_cols[-1],volu_cols[-1], 'f_v2_contr']
        qstr='%s>0 and %s>0' % (flag_cols[0],flag_cols[1])
#    print(qstr)
        pdf['f_v2_contr']=pdf.eval(qstr)*1.0
#    ret_dict={}
#    ret_dict['pdf']=pdf
#    ret_dict['flag_cols']=flag_cols

    return pdf, flag_cols

def get_absolute_rank(rs_matrix_df, idx, reverse=True):
#    colsize=len(rs_matrix_df.columns )
    row=rs_matrix_df.loc[idx]
    valid_df=(row>0)*1.0
    cnt=sum(valid_df)

    _=row[valid_df>0].rank()
    if reverse:
        _=cnt-_+1
#    print('cnt is ', cnt, len(row), _.index)
    na_col=[col for col in rs_matrix_df.columns if col not in _.index ]
#    print('na_col:', na_col)
    for col in na_col:
        _.loc[col]=-2
#    print(_)
    return _

def get_relative_rank(rs_matrix_df, idx='20201106'):
#    colsize=len(rs_matrix_df.columns )
    row=rs_matrix_df.loc[idx]
    valid_df=(row>0)*1.0
    cnt=sum(valid_df)

    _=row[valid_df>0].rank()/cnt
#    print('cnt is ', cnt, len(row), _.index)    
    na_col=[col for col in rs_matrix_df.columns if col not in _.index ]
#    print('na_col:', na_col)
    for col in na_col:
        _.loc[col]=-2
#    print(_)        
    return _

def gen_rs_rank_matrix_from_rs_df(rs_df):
    xdf=pd.DataFrame({'date':rs_df.index}, index=rs_df.index)
    rel_rank_df=xdf.date.apply(lambda x:get_relative_rank(rs_df, x))
    
    return rel_rank_df


def gen_rs_rank_matrix(tag='all_non_etf', startdate='20190101', ref_day=150, rticker='IWM'):
    from macrolib.doubleRSTrader import get_updated_rs_df_by_list_fname
    print('startdate:', startdate)
    ret_df=get_updated_rs_df_by_list_fname(tag, ref_day=ref_day, rticker=rticker, startdate=startdate)
    #                                               batch_data=batch_data)
    rs_matrix_df=ret_df



    xdf=pd.DataFrame({'date':rs_matrix_df.index}, index=rs_matrix_df.index)
    #_2=rs_matrix_df.index.apply(lambda x:get_relative_rank(rs_matrix_df, x))
    rel_rank_df=xdf.date.apply(lambda x:get_relative_rank(rs_matrix_df, x))
    rel_rank_df.tail()
    
    return rel_rank_df

def rolling_price_by_volume_(ser, pdf):
    _=(pdf.loc[ser.index])
    ret=get_price_by_volume_df(_)    
    big_dict[ser.index[-1]]=ret
    return ser.index[-1]

def rolling_price_by_volume_resistance_(ser, pdf):
    _=(pdf.loc[ser.index])
    ret=get_price_by_volume_df(_)    
    idx=ret.volume_by_price.argmax()
    max_col=ret.index[idx]
    return float(max_col)
def rolling_price_by_volume_resistance_up_(ser, pdf):
    _=(pdf.loc[ser.index])
    cur_idx=ser.index[-1]
    curprice=pdf.loc[cur_idx, 'Close']
    ret=get_price_by_volume_df(_)    
    price_bin=[float(price) for price in ret.index]
    ret['bin_in_f']=price_bin

    subset_idx=ret[(ret['bin_in_f']>=curprice)].index
    print('subset_idx:',subset_idx)    
    if len(subset_idx)==0:
        return curprice
    idx=ret.loc[subset_idx].volume_by_price.argmax()
    max_col=ret.index[idx]
    return float(max_col)

def rolling_price_by_volume_support_down_(ser, pdf):
    _=(pdf.loc[ser.index])
    cur_idx=ser.index[-1]
    curprice=pdf.loc[cur_idx, 'Close']
    ret=get_price_by_volume_df(_)    
    price_bin=[float(price) for price in ret.index]
    ret['bin_in_f']=price_bin

    subset_idx=ret[(ret['bin_in_f']<=curprice)].index
    print('subset_idx:',subset_idx)    
    if len(subset_idx)==0:
        return curprice
    idx=ret.loc[subset_idx].volume_by_price.argmax()
    max_col=ret.index[idx]
    return float(max_col)

def get_sup_res_array(pdf, daycnt_list=[60,120, 160, 200]):
    ret_dict={}
    for daycnt in daycnt_list:
        rol = pdf.Close.rolling(window=daycnt)
        y=rol.apply(rolling_price_by_volume_resistance, raw=False)
        ret_dict['snr_%s' % daycnt]=y
        
    return ret_dict

def get_rolling_slope(ser, winsize=50):

    from scipy.stats import linregress
    ret=ser.rolling(winsize).apply(lambda x:linregress(range(winsize), x).slope)
    return ret


def get_vcp_features_old_version(pdf_, start='20190510', end='20200310'):    
#    import talib
    import pandas_ta
    pdf=pdf_.copy()
    volu_ma=pdf.Volume.rolling(5).mean()
#    volu_slope=talib.LINEARREG_SLOPE(volu_ma, 100)        
    volu_slope=get_rolling_slope(volu_ma, winsize=100)

    pdf['volu_ma']=volu_ma
    pdf['volu_slope']=volu_slope  
    
#    std=pdf.Close.pct_change(5).rolling(20).std()
    highs = pdf['High']
    lows = pdf['Low']
    closes = pdf['Close']

#    atr = talib.ATR(highs, lows, closes, 6)
    atr=pdf.ta.atr(6)
    std=atr
    std_ma=std.rolling(20).mean()
#    std_slope=talib.LINEARREG_SLOPE(std_ma, 100)    
    std_slope=get_rolling_slope(std_ma, winsize=100)
    pdf['std_ma']=std_ma

#    std_slope=talib.LINEARREG_SLOPE(std_ma, 150)
    std_slope=get_rolling_slope(std_ma, winsize=150)
    slope_sum=std_slope.rolling(50).sum()    
    pdf['std_slope']=std_slope

    def rolling_cnt_negative(ts):
        _=ts.apply(lambda x: (x<=0)*1.0)
        return _.sum()
    def rolling_cnt_sign(ts):
        _=ts.apply(lambda x: (x<=0)*1.0)
        _2=ts.apply(lambda x: (x>0)*1.0)        
        return (_-_2).sum()
    
    pdf['std_slope_neg_cnt']=std_slope.rolling(50).apply(rolling_cnt_negative)
    pdf['std_slope_sign_cnt']=std_slope.rolling(50).apply(rolling_cnt_sign)    

    pdf['volu_slope_neg_cnt']=volu_slope.rolling(50).apply(rolling_cnt_negative)
    pdf['volu_slope_sign_cnt']=volu_slope.rolling(50).apply(rolling_cnt_sign)    
    
    return pdf

     

def add_pbv(pdf, pbv_days=100):
    '''
    note: this function is for ploting with midpoint reference 
      so the displayed volume profile only count upto the cutoff_date = midpoint
    '''
    subdf=pdf
#    if not cutoff_date is None:
#        subdf=pdf.loc[:cutoff_date].copy()
    def rolling_price_by_volume(ser):
        return rolling_price_by_volume_(ser, subdf )
    def rolling_price_by_volume_resistance(ser):
        return rolling_price_by_volume_resistance_(ser, subdf )
    def rolling_price_by_volume_resistance_up(ser):
        return rolling_price_by_volume_resistance_up_(ser, subdf )
    def rolling_price_by_volume_resistance_down(ser):
        return rolling_price_by_volume_resistance_down_(ser, subdf )
    rol = pdf.Close.rolling(window=pbv_days)
    pdf['pbv_%s' % pbv_days]=rol.apply(rolling_price_by_volume_resistance, raw=False)
    return pdf

def add_snr_related_feats(df, snr_day_arr=[40, 70, 120], use_weekly_data=False, resample_days=1, normalize=True):
    from datalib.patternTraderUtil import get_snr_with_skip, get_snr_with_consolid_ratio
    from vcplib.vcpUtil import resample_df
    if resample_days>1:
        print('resample')
        df=resample_df(df, rule=f'{resample_days}D')
    old_cols=list(df.columns)
    ref_col=df['Close'].rolling(20).mean()
    pdf2, bins=get_price_by_volume_df(df, retbins=True, bin_cnt=20)
    for snr_day in snr_day_arr:
#        df=get_snr_with_consolid_ratio(df, col_prefix='snr', snr_day=snr_day, skip_cnt=4, normalize=normalize)
        df=get_snr_with_consolid_ratio(df, col_prefix='snr', snr_day=snr_day, bins=bins, skip_cnt=4, normalize=normalize)
        if df is None:
            return None, None

    new_cols=list(df.columns)
    for col in old_cols:
        if col in new_cols:
            new_cols.remove(col)
    return df, new_cols


def get_snr_with_skip(pdf, col_prefix='snr_mid_', snr_day=100, skip_cnt=4,  bins=None, include_base=False, update_src=False, normalize=False):
    import numpy as np
    def rolling_price_by_volume__(ser, pdf):
        _=(pdf.loc[ser.index])
        ret=get_price_by_volume_df(_)

        big_dict[ser.index[-1]]=ret
        return ser.index[-1]

    def rolling_price_by_volume_resistance_idx(idx, pdf, pbv_days=150, bins=None, include_base=False, normalize_col=None):
        _=(pdf.loc[:idx].iloc[-pbv_days:])
    #    print('subset shape:',_.shape, _.tail())
#        ret=get_price_by_volume_df(_,  bins= bins)  
        ret=get_price_by_volume_obv_df(_,  bins= bins)  
#        print(ret)        
        if ret is None:
            if not include_base:
                return np.nan
            import numpy as np
            ret={f'base_density_{snr_day}':np.nan, f'snr_{snr_day}':np.nan}
            return ret
        max_idx=ret.volume_by_price.argmax()
        max_col=ret.index[max_idx]
        if 'pbv_up_ratio_at_price' in ret.columns:
            snr_obv_up_ratio=ret.loc[max_col].pbv_up_ratio_at_price
        else:
            snr_obv_up_ratio=0.5

#        band_width=float(ret.index[4])-float(ret.index[3])
#        print('idx in rolling func is ', idx, max_idx)
        max_col=ret.index[max_idx]
        snr=float(max_col)
        snr
        if not normalize_col is None:
            try:
                snr=snr/normalize_col.loc[idx]
            except Exception as e:
                import numpy as np
                import traceback
                print('RrrrrrrrRRRRRRRRRRRRRRRRRR  get_snr_with_skip exception:', e)
                traceback.print_exc()
                print('pdf tail:', pdf.tail())
                print('idx, max_idx, index first 10:', idx, max_idx, pdf.index[:10])
                print('normalize col is ',normalize_col, normalize_col.shape)
                snr=np.nan
        if not include_base:
            return float(max_col)
#        print('idx, sum ', idx, sum((ret.volume_by_price.fillna(0))))
        vol_sum= sum(ret.volume_by_price.fillna(0))
        base_density=0.001
        if vol_sum>0:
            base_density=ret.volume_by_price[max_idx]/vol_sum
        ret={f'base_density_{snr_day}':base_density, f'snr_{snr_day}':snr, f'snr_obv_up_ratio_{snr_day}':snr_obv_up_ratio}
        return ret
                 
    allidx=list(pdf.index)
    subidx=list(set(allidx[snr_day::skip_cnt]))
    if len(subidx)==0 or len(pdf)==0:
        return '', None
    subidx.append(pdf.index[-1])
    subidx=list(set(subidx))
    subidx=sorted(subidx)
    longma=pdf.Close.rolling(snr_day+100, min_periods=0).mean().loc[subidx]       
#    print('longma',longma)
    if normalize:
        snr_with_skip2={idx:rolling_price_by_volume_resistance_idx(idx, pdf, snr_day, include_base=include_base, normalize_col=longma) for idx in subidx}
    else:
        snr_with_skip2={idx:rolling_price_by_volume_resistance_idx(idx, pdf, snr_day, include_base=include_base, normalize_col=None) for idx in subidx}
    if include_base:
#        print('include base')
        subdf=pd.DataFrame(snr_with_skip2).T
        for colname in subdf.columns:
            pdf.loc[subidx, colname]=subdf[colname]
            pdf.ffill(inplace=True)
        if update_src:
            return subdf.columns, pdf
        else:
            return subdf.columns, pdf[subdf.columns]

    else:
#        print('not include base', snr_with_skip2)
        ret_ser=pd.Series(snr_with_skip2)
#        print('not include base', ret_ser)
        colname=f'{col_prefix}_{snr_day}'
        pdf[colname]=np.nan
        pdf.loc[subidx, colname]=ret_ser
        pdf.ffill(inplace=True)
#        print('colname:',colname)
#        print('pdf.columns:',pdf.columns)
#        print('retser :',pdf[colname])
#        return colname, pdf[colname]
        if update_src:
            return colname, pdf
        else:
            return colname, pdf[colname]
    


def get_snr_with_consolid_ratio(pdf, col_prefix='snr', snr_day=100, bins=None, skip_cnt=4, normalize=False):
    max_look_back=50
    range_th=0.05
    ub=1+range_th
    lb=1-range_th
    snr_colname=f'{col_prefix}_{snr_day}'
    cols, pdf=get_snr_with_skip(pdf, col_prefix=col_prefix, snr_day=snr_day, bins=bins, include_base=True, update_src=True, normalize=normalize)
    if pdf is None:
        return None
#    print('colnames:',pdf.columns)
    near_snr=pdf.eval(f'Close>{snr_colname}*{lb} and Close>{snr_colname}*{ub}')*1.0
    pdf[f'consolid_ratio_{snr_day}']=near_snr.rolling(max_look_back, min_periods=10).sum()/max_look_back
    return pdf

    
def gen_vcp_signal(pdf, ticker=None, snr_day=100, end_datestr=None, volu_pctile=0.1, vola_pctile=0.1):
    if not end_datestr is None:
        pdf=pdf.loc[:end_datestr].iloc[-500:]
    
    def rolling_price_by_volume(ser):
        return rolling_price_by_volume_(ser, pdf )
    def rolling_price_by_volume_resistance(ser):
        return rolling_price_by_volume_resistance_(ser, pdf )
    def rolling_price_by_volume_resistance_up(ser):
        return rolling_price_by_volume_resistance_up_(ser, pdf )
    def rolling_price_by_volume_resistance_down(ser):
        return rolling_price_by_volume_resistance_down_(ser, pdf )

    try:
        pdf, flag_cols=add_vcp_features_by_pctile(ticker, pdf, volu_pctile=volu_pctile, vola_pctile=vola_pctile, smooth_days=5)
        if pdf is None:
            return None
        pdf['date']=pdf.index
#        pdf=get_vcp_features(pdf)
#        pdf=gen_minervini_filter_criteria(pdf.copy())
    except Exception as e:
        import traceback
        print(ticker , 'exception:', e)
        traceback.print_exc()
        return None
    
    daycnt=snr_day
    #rol = pdf.Close.rolling(window=daycnt)
    #pdf['snr150']=rol.apply(rolling_price_by_volume_resistance, raw=False)
#    pdf['snr_mid']=rol.apply(rolling_price_by_volume_resistance, raw=False)
#    snr_col, snr=get_snr_with_skip(pdf, snr_day=snr_day)
#    pdf['snr_mid']=snr
    pdf_tm=pdf.copy()
    pdf_tm['Volume']=pdf_tm.eval('Volume*Close')
    tm_snr_col, tm_snr=get_snr_with_skip(pdf_tm, snr_day=snr_day)
    if tm_snr is None:
        return None
    pdf['tm_snr_mid']=tm_snr
    del pdf_tm
    pdf['snr_mid']=pdf['tm_snr_mid']
#    _col=roltm.apply(rolling_price_by_volume_resistance, raw=False)
    pdf['ma15']=pdf.Close.rolling(15).mean()
#    pdf=org_pdf
#    pdf['tm_snr_mid']=_col
#    _=pdf.query('not snr_mid==tm_snr_mid') 
    sig=get_signal_cross(pdf, 'Close', 'snr_mid')
    pdf['breakout_sig']=sig
    try:
#        print('pdf shape before snr compare:',pdf.shape, pdf.dtypes)
        pdf['f_breakout_long']=(pdf.Close>pdf.snr_mid)*1.0
    
    except Exception as e:
        import traceback
        print('exception:', e)
        traceback.print_exc()
        import sys,os,datetime,gzip,pickle,glob,time,traceback,random,matplotlib,subprocess,importlib,inspect
        from debuglib.debugUtil import frame2func, save_func_input_state
        save_func_input_state(frame2func(sys._getframe(  )),  locals())
        return None
#    pdf['vcomb_flag']=pdf.eval('volu_slope_sign_cnt>0 and std_slope_sign_cnt>0 and mmfilter_flag>0')*1.0
    if 'f_v2_contr' in pdf.columns:
        pdf['f_vcomb_flag']=pdf['f_v2_contr'].rolling(10, min_periods=1).max()*1.0
    else:
        pdf['vcomb_flag']=pdf.eval('volu_slope_sign_cnt>0 and std_slope_sign_cnt>0 and mmfilter_flag>0').rolling(10, min_periods=1).max()*1.0
#    pdf['output_flag_2']=pdf.eval('vcomb_flag_2>0 and not breakout_sig==0')*1.0
    pdf['f_breakout_comb']=pdf.eval('f_vcomb_flag>0 and f_breakout_long>0')*1.0
#    pdf['f_mmfilter']=pdf.mmfilter_flag*1.0
    return pdf

def gen_minervini_filter_criteria(pdf,th=0.25):
    long_ma=200
    short_ma=50
    mid_ma=150
    pdf['ma_long']=pdf.Close.rolling(long_ma).mean()
    pdf['ma_short']=pdf.Close.rolling(short_ma).mean()    
    pdf['ma_mid']=pdf.Close.rolling(mid_ma).mean()        
    pdf['multi_ma_flag']=pdf.eval('ma_short>ma_mid and ma_mid > ma_long')
#    pdf['multi_ma_flag']=(pdf.ma_short>pdf.ma_mid) and (pdf.ma_mid > pdf.ma_long)
    
    pdf['low_52w']=pdf.Close.rolling(long_ma+short_ma).min()
    pdf['high_52w']=pdf.Close.rolling(long_ma+short_ma).max()    
    pdf['mm_th_above_low_52w_flag']=pdf.Close>(pdf.low_52w*1.25)
    pdf['mm_th_near_high_52w_flag']=pdf.Close>(pdf.high_52w*0.75)
    pdf['mmfilter_flag']=pdf.eval('multi_ma_flag and mm_th_above_low_52w_flag and mm_th_near_high_52w_flag') * 1.0
    return pdf

def plot_vcp_signal(pdf, end=None, title='ticker'):
#    if not 'breakout_sig' in pdf.columns:
#        pdf=get_vcp_features(pdf)
    
    #cols1=['Close', 'snr_150', 'ma15']
    cols1=['Close', 'snr_mid', 'ma15']
    df1=pdf[cols1]
    cols2=['volu_slope', 'volu_ma']
    #cols2=['std_slope_sign_cnt', 'std_slope_neg_cnt']
#    cols2=['std_slope_sign_cnt']
#    df2=pdf[cols2]
    #cols3=['volu_slope_sign_cnt', 'volu_slope_neg_cnt']
    cols3=['volu_slope_sign_cnt']
    df3=pdf[cols3]


    cols=['breakout_sig']
    df4=pdf[cols]
    cols=['f_breakout_long']
    df5=pdf[cols]
    plot_df_arr=[df1, df3, df4, df5]
    for col in pdf.columns:
        if '_flag' in col:
            plot_df_arr.append(pdf[[col]])
    flex_plot(plot_df_arr, title, end=end)


def convert_exposure_flag_to_raw_entry_exit(pdf2, colname='tmp1'):
    pdf3=pdf2.copy()
    pdf3['dummy_line']=0.1
    pdf3['raw_entry']=get_signal_cross(pdf3, colname, 'dummy_line')>0
    pdf3['raw_exit']=get_signal_cross(pdf3,  'dummy_line', colname)>0
    plot_arr=[]
    plot_arr.append(pdf2[['breakout_long']])
    plot_arr.append(pdf2[['multi_ma_flag']])
    plot_arr.append(pdf2[['tmp1']])
#    plot_arr2=plot_arr.copy()
    plot_arr.append(pdf3[['raw_entry']])
    plot_arr.append(pdf3[['raw_exit']])
    flex_plot(plot_arr)
    return pdf3, plot_arr


def backtest_from_chart_df(ticker, pdf, rticker='IWM'):
    '''  
    pdf needs to be from read_yfgz with 'raw_entry', 'raw_exit' added
    '''
    entry_signal_df=pdf.query('raw_entry>0')
    entry_signal_df['entry_date']=entry_signal_df.index
    entry_signal_df['exit_date']=entry_signal_df['entry_date']+pd.to_timedelta('5 days')
    exit_signal_df=pdf.query('raw_exit>0')
    from sig_matrix.backtestUtil import match_exit, get_tradedates_df_from_signal
    for entry_date in entry_signal_df.index:
        exit_date=match_exit(entry_date, exit_signal_df.index)
        if exit_date is None:
            exit_date=pdf.index[-1]
        print(entry_date, exit_date)
        entry_signal_df.loc[entry_date, 'exit_date']=exit_date                       
    #entry_signal_df    
    tradedates_df=get_tradedates_df_from_signal(entry_signal_df, ticker, rticker='SPY', price_df=None)                         
    tradedates_df.tail()
    #entry_signal_df

    from sig_matrix.matrixTraderUtil import tradeDateBackTester, signalTradeReviewer
    rprice_df=cu.read_quote(rticker)
    tdbt=tradeDateBackTester()
    trades_df=tdbt.backtest_from_tradedates_df(tradedates_df, pdf, rprice_df, trade_type='long')

    sigrt=signalTradeReviewer()
        #sigrt.review_all_ticker(trades_df)
    print('trade df len : %s ' % len(trades_df))
    if len(trades_df)==0:
        return None
    ret_dict=sigrt.summarise_trades(trades_df)
    ret_dict['features_df']=pdf
    return ret_dict   

def breakout_test_v1(ticker, pdf=None):
    if pdf is None:
        pdf=cu.read_quote(ticker)
    pdf2=gen_vcp_signal(pdf, ticker=ticker)
    if pdf2 is None:
        return None
    pdf2=gen_minervini_filter_criteria(pdf2.copy())
    pdf2['tmp1']=pdf2.eval('breakout_long>0 and multi_ma_flag>0')
    plot_arr=[]
    plot_arr.append(pdf2[['breakout_long']])
    plot_arr.append(pdf2[['multi_ma_flag']])
    plot_arr.append(pdf2[['tmp1']])
    flex_plot(plot_arr)
    title=ticker
    plot_vcp_signal(pdf2, title=title)
    pdf3, plot_arr=convert_exposure_flag_to_raw_entry_exit(pdf2, colname='tmp1')
    ret_dict=backtest_from_chart_df(ticker, pdf3)
    return ret_dict

def get_summary_dataframe(big_dict):
    col_list=[]
    data_dict={}
    ex_dict={}
    type_list=[type(pd.DataFrame()), type(pd.Series())]    
    for key in big_dict.keys():
        ret_dict=big_dict[key]
        if ret_dict is None:
            continue
        new_dict=[ret_dict[col] for col in ret_dict.keys() if not type(ret_dict[col]) in type_list]
        ex_new_dict=[ret_dict[col] for col in ret_dict.keys() if type(ret_dict[col]) in type_list]
#        new_dict=ret_dict
        ex_col_list=[col for col in ret_dict.keys() if not type(ret_dict[col]) in type_list]
        
        data_dict[key]=(ret_dict)
        ex_dict[key]=ex_new_dict
    df=pd.DataFrame.from_dict(data_dict, orient='index')
    df['total_trade_cnt']=df.eval('win_trade_cnt+loss_trade_cnt')
    df['radj_ret_per_trade']=df.eval('mean_pct_profit/mean_day_in_trade')
    return df, ex_dict

def batch_breakout_test_v1(tlist, sample_size=10, test_row_cnt=350):
    import random
#    _tlist=read_tlist('all_non_etf')
    sample_size=min(len(tlist), sample_size)
    tlist=random.sample(tlist, sample_size)
    big_dict={}
    train_big_dict={}
    test_big_dict={}    
    for ticker in tlist:
        print('ticker:',ticker)
        pdf=cu.read_quote(ticker)
        if pdf is None:
            continue
        train_pdf=pdf.iloc[:-test_row_cnt]
        train_ret_dict=breakout_test_v1(ticker, train_pdf)        
        if test_row_cnt > 200:
            test_pdf=pdf.iloc[-test_row_cnt-250:]        
            test_ret_dict=breakout_test_v1(ticker, test_pdf)               
            if not test_ret_dict is None:
                 test_big_dict[ticker]=test_ret_dict
        if train_ret_dict is None:
            continue
        train_big_dict[ticker]=train_ret_dict
    train_df, train_ex_dict=get_summary_dataframe(train_big_dict)
    test_df, test_ex_dict=get_summary_dataframe(test_big_dict)    
    breakout_dict={}
    breakout_dict['test_df']=test_df
    breakout_dict['train_df']=train_df    
    breakout_dict['train_ex_dict']=train_ex_dict
    breakout_dict['test_ex_dict']=test_ex_dict    
    return breakout_dict


def add_higher_high_and_low_flag(pdf, show_plot=False):
#    pdf.Close[-300:].max(), pdf.Close[-300:].min(), x
#    import talib
    pdf['trend_top_']=pdf.Close.rolling(50).apply(lambda x:np.percentile(x, 99))
    pdf['trend_bottom_']=pdf.Close.rolling(50).apply(lambda x:np.percentile(x, 1))

#    hslope=talib.LINEARREG_SLOPE(pdf['trend_top_'], 300)
    hslope=get_rolling_slope(pdf['trend_top_'], winsize=300)
#    lslope=talib.LINEARREG_SLOPE(pdf['trend_bottom_'], 300)
    lslope=get_rolling_slope(pdf['trend_bottom_'], winsize=300)
    pdf['hslope_flag_']=(hslope>0)*1.0
    pdf['lslope_flag_']=(lslope>0)*1.0
    pdf['wedge_flag']=pdf.eval(' hslope_flag_ * lslope_flag_')
    pdf['wedge_diff']=pdf.eval('trend_top_-trend_bottom_')
#    pdf['wedge_diff_slope']=talib.LINEARREG_SLOPE(pdf['wedge_diff'], 150)
    pdf['wedge_diff_slope']=get_rolling_slope(pdf['wedge_diff'], winsize=150)
    ll=lslope>0
#    hslope.plot()
#    lslope.plot()
#    pdf['wedge_flat'].plot()
#    pdf.Close.plot()
#    pdf['trend_top_'].plot()
#    pdf['trend_bottom_'].plot()    
#    pdf['wedge_diff'].plot()
#    (pdf['wedge_diff_slope']*100).plot()
    pdf['wedge_diff_slope_flag']=(pdf['wedge_diff_slope']<0)*1.0
    pdf['wedge_diff_slope_flag_sum']=pdf.rolling(25).wedge_diff_slope_flag.sum()
    pdf['wedge_diff_slope_flag_sum_flag']=pdf.wedge_diff_slope_flag_sum>15    
#    (hl*1.0).plot()
#    (ll*1.0).plot()

    pdf['dummy']=0.1
    pdf['sig_wedge_diff_slope_flag_sum_flag']=get_signal_cross(pdf, 'dummy', 'wedge_diff_slope_flag_sum')

    
    
    def _plot(pdf):
        df_arr=[]
        cols=['trend_top_', 'trend_bottom_', 'Close']
        pdf['h0']=0
        df_arr.append(pdf[cols])
        cols=['wedge_diff_slope', 'h0']
        df_arr.append(pdf[cols])
        cols=['wedge_diff_slope_flag']
        df_arr.append(pdf[cols])
        cols=['wedge_diff_slope_flag_sum']
        df_arr.append(pdf[cols])

        cols=['wedge_diff_slope_flag_sum_flag']
        df_arr.append(pdf[cols])
        for col in pdf.columns:
            if col[:4]=='sig_':
                df_arr.append(pdf[[col]])
        flex_plot(df_arr)
    if show_plot:
        _plot(pdf)
    return pdf

def get_relative_gain_diff_ttest(pdf, flag_col='wedge_diff_slope_flag', tday=2):
    from scipy import stats
    pdf['_lb_nday_chg']=pdf.Close.pct_change(tday).shift(-tday)
    
    df_inpos=pdf.query('%s>0' % flag_col).dropna()
    df_nopos=pdf.query('%s<=0' % flag_col).dropna()
    tt_ret=stats.ttest_ind(a=df_inpos._lb_nday_chg,b=df_nopos._lb_nday_chg, equal_var=False)
    ret={}
    ret['tt_ret']=tt_ret
    ret['inpos_size']=len(df_inpos._lb_nday_chg)
    ret['nopos_size']=len(df_nopos._lb_nday_chg)    
    ret['inpos_mean']=df_inpos._lb_nday_chg.mean()
    ret['nopos_mean']=df_nopos._lb_nday_chg.mean()
    ret['inpos_std']=df_inpos._lb_nday_chg.std()    
    ret['nopos_std']=df_nopos._lb_nday_chg.std()    
    ret['inpos_kurt']=stats.kurtosis(df_inpos._lb_nday_chg)
    ret['nopos_kurt']=stats.kurtosis(df_nopos._lb_nday_chg)    
    #print(ret)
    return ret

def get_sell_into_strength_return(pdf, entry_date='20200323', longshort='long', share_cnt=100, wstop_step=0.1, trail_stop=0.05):
    subdf=pdf.loc[entry_date:]
    entry_price=subdf.iloc[0].Close

    print('entry_price:',entry_price)
    commission_pct=0.0001
    trades=[]
    pos=share_cnt
    pos_cost=entry_price*pos    
    ilongshort=1
    stage_cnt=4
    q=share_cnt/stage_cnt
    if longshort=='short':
        ilongshort=-1
    pos=pos*ilongshort
    q=q*ilongshort
    total_gain=0
    realized_gain=0
    wstop=wstop_step
    loss_stop=-1*trail_stop
    trade_stage=0
    peak_ret=0
    peak_drawdown=0
    equity_curve_dict={}
    for k,row in subdf.iterrows():
        curprice=row.Close
        ret=ilongshort*(row.Close-entry_price)/entry_price

            
        peak_ret=max(ret, peak_ret)
        retrace=peak_ret-ret
        unrealized_gain=ret*pos*entry_price
        total_gain=unrealized_gain+realized_gain
        total_gain_pct=total_gain/pos_cost
        equity_gain=unrealized_gain
        peak_drawdown=max(peak_drawdown, retrace)
        snapshot={}
        snapshot['unit_ret']=ret
        snapshot['wstop']=wstop
        snapshot['peak_ret']=peak_ret
        snapshot['retrace']=retrace
        snapshot['pos']=pos
        snapshot['remain_pos_value']=pos*curprice      
        snapshot['trail_stop']=trail_stop
        snapshot['trade_stage']=trade_stage        
        snapshot['unrealized_gain']=unrealized_gain        
        snapshot['total_gain']=total_gain                
        snapshot['realized_gain']=realized_gain
        snapshot['total_gain_pct']=total_gain_pct
        equity_curve_dict[k]=snapshot
#        print('snapshot:',snapshot)

        def _get_trade(pos, q, stop_type='win'):
            #print('in get trade pos:', pos, q)
            trade={}
            trade['entry_date']=entry_date            
            trade['exit_date']=k
            trade['entry_price']=entry_price
            in_trade_td=k-entry_date
            trade['days_in_trade']=in_trade_td.days
            trade['exit_price']=curprice            
            trade['trade_stage']=trade_stage
            trade['before_pos']=pos
            trade['trade_size']=q
            trade['peak_drawdown']=peak_drawdown
            trade['peak_ret']=peak_ret            
            trade['commission']=curprice*    commission_pct          
            trade['pct_gain_from_entry']=ret            
            trade['stop_type']=stop_type
            trade['after_pos']=pos-q
            gain=(curprice-entry_price)
            trade['gain']=gain        

            return trade, trade['after_pos']
        if ret<loss_stop:
            print('now stop loss!, ', k, pos)
            trade, pos=_get_trade(pos, pos, 'stop_loss')
            trades.append(trade)                
            break
            
        if ret>wstop and trade_stage<(stage_cnt-1):
            trade_stage=trade_stage+1
            wstop=wstop+wstop_step

            trade, pos=_get_trade(pos, q)
            gain=trade['gain']
            realized_gain=realized_gain+trade['trade_size']*gain         
            trades.append(trade)
            print('new stop %s, trade_stage:%s, total_gain:%s, pos:%s' % (wstop, trade_stage, total_gain, pos))
            print('sell into strength %s' % trade)            
        else:
            if retrace>trail_stop:

                trade, pos=_get_trade(pos, pos, 'trail_stop')                
                print('trail stopped %s' % trade)                
                trades.append(trade)                
        if pos==0:
            break
    equity_df=pd.DataFrame(equity_curve_dict).T
    return trades, equity_df

def get_trades_performance(trades, pdf):
    if pdf is None or trades is None:
        return None
    if len(trades)==0:
        return None
    pos_cost=0.01
    start_pos=0
    total_gain=0
    total_commission=0
    total_days_in_trade=0
    _=pdf.index[-1]-pdf.index[200]
    total_days=_.days
    entry_date=None
    exit_date=None
    trade_cnt=len(trades)
    for trade in trades:
        if pos_cost==0:
            start_pos=trade['before_pos']
            pos_cost=start_pos*trade['entry_price']
            entry_date=trade['entry_date']
        total_gain=total_gain+trade['gain']*trade['trade_size']
        total_days_in_trade=total_days_in_trade+trade['days_in_trade']
        total_commission=total_commission+trade['commission']
        exit_date=trade['exit_date']
    pct_gain=(total_gain-total_commission)/pos_cost
    ret_dict={}
    ret_dict['total_gain']=total_gain
    ret_dict['entry_date']=entry_date
    ret_dict['exit_date']=exit_date
    ret_dict['total_days_in_trade']=total_days_in_trade
    ret_dict['average_exposure_days']=total_days_in_trade/len(trades)
    ret_dict['total_commission']=total_commission
    ret_dict['pos_cost']=pos_cost
    ret_dict['pct_gain']=pct_gain
    return ret_dict


def backtest_sell_into_strength_from_signal(pdf, trade_dates):
    t_pct_ret=0
    t_days_in_trade=0
    all_ret_dict={}  
    _=pdf.index[-1]-pdf.index[200]
    total_days=_.days
    for k in trade_dates:
        #print(k)
        trades, equity_df=get_sell_into_strength_return(pdf, entry_date=k, wstop_step=0.15, trail_stop=0.1)
        ret_dict=get_trades_performance(trades, pdf)
        if ret_dict is None:
            continue

        ret_dict['test_duration']=total_days
        pct_gain=ret_dict['pct_gain']
        t_days_in_trade=ret_dict['average_exposure_days']
        
        print(k,'******* pct gain:%s' % (pct_gain))
        all_ret_dict[k]=ret_dict        
        t_pct_ret=t_pct_ret+pct_gain

        
#    trades, equity_df=get_sell_into_strength_return(pdf, entry_date='20200323', wstop_step=0.15, trail_stop=0.1)
#    equity_df                          


    print(k, 'total_pct_ret:',t_pct_ret)
    df_s2s=pd.DataFrame(all_ret_dict).T
    return df_s2s

def get_exposure_adjusted_return(df_s2s):
    t_intrade=df_s2s.total_days_in_trade.sum()
    t_exposure=df_s2s.average_exposure_days.sum()    
    t_duration=df_s2s.iloc[0].test_duration
    t_years=t_duration/360
    exposure=t_intrade/df_s2s.iloc[0].test_duration
    t_pct_gain=df_s2s.pct_gain.sum()
    t_pct_gain, exposure
    annualized_pct_gain=t_pct_gain/t_years
    exp_adj_annualized_pct_gain=annualized_pct_gain*t_duration/t_intrade
    exp_adj_annualized_pct_gain_s2s=annualized_pct_gain*t_duration/t_exposure    
    annualized_pct_gain, exp_adj_annualized_pct_gain, t_pct_gain, exposure, t_years, t_intrade
    ret_dict={}
    ret_dict['annualized_pct_gain']=annualized_pct_gain
    ret_dict['exp_adj_annualized_pct_gain']=exp_adj_annualized_pct_gain    
    ret_dict['exp_adj_annualized_pct_gain_s2s']=exp_adj_annualized_pct_gain_s2s        
    ret_dict['t_pct_gain']=t_pct_gain    
    ret_dict['exposure']=exposure    
    ret_dict['t_years']=t_years        
    ret_dict['t_intrade']=t_intrade            
    return ret_dict

def test_wedge_trend(ticker='TSLA'):
    pdf=cu.read_quote(ticker)    
    pdf=add_higher_high_and_low_flag(pdf, show_plot=True)
    trade_dates=pdf.query('sig_wedge_diff_slope_flag_sum_flag>0').index
    backtest_sell_into_strength_from_signal(pdf, trade_dates)
    df_s2s=backtest_sell_into_strength_from_signal(pdf, trade_dates)
    display(df_s2s)
    return df_s2s
    
def test_run():
    tlist=['TSLA', 'FB', 'AMD', 'SQ', 'FSLY', 'GOOG', 'AMZN', 'NFLX', 'NVDA']
    big_dict={}
    for ticker in tlist:
        print('ticker:',ticker)
        pdf=cu.read_quote(ticker)
        
        ret_dict=breakout_test_v1(ticker, pdf)
        if ret_dict is None:
            continue
        big_dict[ticker]=ret_dict
    df=get_summary_dataframe(big_dict)
    display(df)
    return df

def test_vcp_s2s(ticker='TSLA'):
    print('ticker:',ticker)
    pdf=cu.read_quote(ticker)

    ret_dict=breakout_test_v1(ticker, pdf)

    pdf2=ret_dict['features_df']

    trade_dates=pdf2.query('raw_entry>0').index
    trade_dates
    df_s2s=backtest_sell_into_strength_from_signal(pdf, trade_dates)
    display(df_s2s)
    return df_s2s
    
#df_s2s=test_vcp_s2s()
#df_s2s=test_wedge_trend()
#ret_dict=get_exposure_adjusted_return(df_s2s)
#ret_dict
#test_run()
#df_s2s=test_wedge_trend()
#plt.subplots(2,2,figsize=(15,15))


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
#        display('flag:',df.tail())
    return df

def flex_plot(df_arr, title='ticker', point_list=[], ofname='tmp/flexplot.jpg', start=None, end=None, mini_plot=False):
    if type(df_arr)==type(pd.DataFrame()):
        print('a plot input type = df')
        cnt=len(df_arr.columns)
    else:
        print('a plot input type = arr')                                                                                                                                                          
        cnt=len(df_arr)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    import mplfinance as mpf
    print('a22')
    s = mpf.make_mpf_style(base_mpf_style='yahoo', rc={'font.size': 10})
#    s = mpf.make_mpf_style(base_mpf_style='charles', rc={'font.size': 10})
    #fig = plt.figure()
#    fig=plt.figure(figsize=(15, cnt+3))

    plot_width=18
    if mini_plot:
        plot_width=9
        df1=df_arr[-1]
        half=int(len(df1)/2)
        start=df1.index[half]
    #fig=plt.figure(figsize=(plot_width, cnt*1.5+5))
    print('a22a')
    fig=mpf.figure(figsize=(plot_width, cnt*1.5+5), style=s)
    print('a22b')
    gs = gridspec.GridSpec(cnt+3,1)
    print('a22c')
    fig.subplots_adjust(hspace=0)
    ax_arr=[]
#    for rect in rect_list[::-1]:
    print('xcnt is ',cnt, gs)
    for i in range(cnt):
        print('i is :',i, len(df_arr))
        if i==0:
            ax=fig.add_subplot(gs[0:i+4])
            ax1=ax

        else:
            ax=fig.add_subplot(gs[i+3], sharex=ax1)
        twinx=False
        print('23a')
        multi_df=None
        if type(df_arr)==type([]):
            df=df_arr[i]
            if type(df)==type([]):
                multi_df=df
                df=df[0]
        else:
            cols=df_arr.columns
            col=cols[i]
            df=df_arr[[col]]
        if start is None:
            start=df.index[0]
        if end is None:
            print('a b4:',df.shape)
            df=df.loc[start:]
            print('after:',df.shape)            
        else:
            print('b b4:',df.shape)      
            df=df.loc[start:end]
            print('after:',df.shape)            
#        ax=fig.add_axes(rect)
        t='%s' % list(df.columns)
        if i==0:
            t='%s:%s' % (title, t)
        ax.set_title(t)      
        print('22abb')
        for point in point_list:
#            print('point is ',point)
            idxs=df.loc[point:].index
            _point=idxs[0]
            ax.axvline(x=_point)

        flag_cols=[col for col in df.columns if col[:2]=='f_']
        print('flexi plot!!!!!!!!!!!! all flag?', df.columns, flag_cols)
        if len(flag_cols) ==len(df.columns):
            df=expand_flag_pdf(df)
        if not multi_df is None:
            _1=multi_df[0]
#            ax=_1.plot(figsize=(18,8))
            ax.plot(_1, label=list(_1.columns))
#            ax.legend(loc='upper left')
            ax2=ax.twinx()
            ax.legend()
#            mpf.plot(df_arr[0], type='candle',   ax=ax2, volume=ax3, show_nontrading=True)
            if 'Open' in _1.columns:
                mpf.plot(_1, type='candle',   ax=ax, volume=ax2, show_nontrading=True)
            else:
                print('_1.columns:', _1.columns)
                print('multi df:', multi_df)
                mpf.plot(_1,   ax=ax, volume=ax2, show_nontrading=True)
#                mpf.plot(_1, type='line',   ax=ax, volume=ax2, show_nontrading=True)
                
            for _ in multi_df[1:]:
                print('multi df 2:',_)
                line,=ax2.plot(_, label=list(_.columns),  color='green')
                ax2.legend(loc='upper right')
        else:
            line=ax.plot(df,  label=list(df.columns))
            ax.legend(loc='upper left')
        i=i+1
        ax_arr.append(ax)
    plt.savefig(ofname)
    print('saving flexplot to ',ofname)
    return ax_arr
