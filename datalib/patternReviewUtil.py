from datalib.patternTraderUtil import gen_vcp_signal, add_pbv, add_volatility_contract_by_quantile, add_volume_contraction_flag, add_vcp_features_by_pctile
#from rrgv3.chartPlotterXin1 import highlight_plot_by_flag
import pandas as pd
#import talib
import matplotlib
matplotlib.use("Agg")
from datalib.commonUtil import commonUtil as cu

#from rrgv3.chartPlotterXin1 import highlight_plot_by_flag
def plot_fib(pdf, plot_bars=300, minmax_bars=40, ex_info=None,plot=True, ax=None):
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from scipy.signal import argrelextrema

    # Generate a noisy AR(1) sample

    pdf['data']=pdf.Close
    df=pdf.iloc[-plot_bars:]

    n = minmax_bars# number of points to be checked before and after

    # Find local peaks

    df['min'] = df.iloc[argrelextrema(df.data.values, np.less_equal,
                        order=n)[0]]['data'].ffill()
    df['max'] = df.iloc[argrelextrema(df.data.values, np.greater_equal,
                        order=n)[0]]['data'].ffill()
    maximum_price=df['max'].ffill()
    minimum_price=df['min'].ffill()
    df['min']=minimum_price
    df['max']=maximum_price    
    difference = maximum_price - minimum_price #Get the difference        
    l1= maximum_price - difference * 0.236   
    l2= maximum_price - difference * 0.382  
    l3= maximum_price - difference * 0.5     
    l4= maximum_price - difference * 0.618  

    ul1=maximum_price+difference * 0.236   
    ul2=maximum_price+difference * 0.382
    ul3=maximum_price+difference * 0.5
    ul4=maximum_price+difference * 0.618
    
    dl1=minimum_price-difference * 0.236   
    dl2=minimum_price-difference * 0.382
    dl3=minimum_price-difference * 0.5    
    #difference.plot()
    if plot:
        from labellines import labelLine, labelLines
        if ax is None:
            ax=df.Close.plot()
        ax.plot(l1, label='0.236_%0.2f' % l1.iloc[-1])
        x_idx=df.index[-10]
        labelLine(ax.get_lines()[-1], x=x_idx, zorder=1)         
        print(ax.get_lines(), len(ax.get_lines()), ax.get_lines()[-1])
        ax.plot(l2, label='0.382_%0.2f' % l2.iloc[-1])
        labelLine(ax.get_lines()[-1], x=x_idx, zorder=1) 
        print(3, ax)
        
        ax.plot(l3, label='0.5_%0.2f' % l3.iloc[-1])     
        labelLine(ax.get_lines()[-1], x=x_idx, zorder=1)         
        ax.plot(l4, label='0.618_%0.2f' % l4.iloc[-1])
        labelLine(ax.get_lines()[-1], x=x_idx, zorder=2)             
        ax.plot(ul1, label='u236_%0.2f' % ul1.iloc[-1])
        labelLine(ax.get_lines()[-1], x=x_idx, zorder=2)                     
        ax.plot(ul2, label='u382_%0.2f' % ul2.iloc[-1])
        labelLine(ax.get_lines()[-1], x=x_idx, zorder=2)                                     

        ax.plot(dl1, label='d236_%0.2f' % dl1.iloc[-1])
        labelLine(ax.get_lines()[-1], x=df.index[-1], zorder=2)                     
        ax.plot(dl2, label='d382_%0.2f' % dl2.iloc[-1])
        labelLine(ax.get_lines()[-1], x=df.index[-1], zorder=2)                                     
        ax.plot(dl3, label='d5_%0.2f' % dl3.iloc[-1])
        labelLine(ax.get_lines()[-1], x=df.index[-1], zorder=2)                                     
        
        #ul3.plot()
        #ul4.plot()
        ax.plot(minimum_price, label='pmin')
        labelLine(ax.get_lines()[-1], x=df.index[-1], zorder=3)                 
        ax.plot(maximum_price, label='pmax')
        labelLine(ax.get_lines()[-1], x=df.index[-1], zorder=3)                 
        ax.scatter(df.index, df['max'], c='g')
        ax.scatter(df.index, df['min'], c='r')
#        labelLines(ax.get_lines(), zorder=3)     
#        print(ax.get_lines(), len(ax.get_lines()))        
#        plt.scatter(df.index, df['min'], c='r')
#        plt.scatter(df.index, df['max'], c='g')
        
#        plt.plot(df.index, df['data'])
#        labelLines(ax.get_lines(), zorder=3)        
    return df


def quick_pbv_breakout_review_extra(ticker='HTZ', datestr=None, pdf=None, daycnt=250, extra_tag=[], chart_tag=None, df_extra=None, ex_cols=[],  info_df=None,
        hl_flags=[], pbv2_days=120, debug=False):
    '''
    df_extra[ex_cols] will also be ported
    hl_flags is the col in pdf which will be used to mark highlight period
    if set debug to true, return all dataframe as dict
    '''
    from pandas.plotting import register_matplotlib_converters
    from IPython.display import Image 
    from datalib.zcatbreadthUtil import gen_breadth_df_by_ticker
    from datalib.extremaPlotter import plot_all_with_breakout

    register_matplotlib_converters()
#    from rrgv3.chartPlotter3in1 import chartPlotter3in1
    from rrgv3.chartPlotterXin1 import chartPlotterXin1, get_default_plot_params
    charter=chartPlotterXin1()
    from rrgv3.priceMovementScannerYHDS import priceMovementScannerYHDS

    if pdf is None:
        pmsy=priceMovementScannerYHDS()    
        #_=pmsy.read_yahoo_df_combined(ticker)
        _=cu.read_quote(ticker)
        if _ is None:
            return None
    else:
        _=pdf
        if df_extra is None:
            df_extra=pdf
    dateidx=pd.to_datetime(datestr)
    startdate=dateidx-pd.to_timedelta('%d days' % (daycnt+50))
    startstr=startdate.strftime('%Y%m%d')
    enddate=dateidx+pd.to_timedelta('%d days' % (daycnt-50))    
    endstr=min(pdf.index[-1], enddate).strftime('%Y%m%d')
    #print(f'{startdate}  and {enddate} pbv vs {dateidx}', pdf.shape, pdf.tail())
    print(f'{startdate}  and {enddate} pbv vs {dateidx}', pdf.dtypes, pdf.tail())
#    _=_.loc[startdate:enddate].copy()
    _=_.loc[startdate:].copy()
#    _=_.between_time(startstr, endstr)
#    _=_.between_time(pd.to_datetime(startstr), pd.to_datetime(endstr))
   
    if not pbv2_days is None:
        if not 'breakout_long' in _.columns:
            _=gen_vcp_signal(_, ticker=ticker, snr_day=pbv2_days)
    if _ is None:
        return None
    g2_overlay_col='snr_mid'
#    if not pbv2_days is None:
#        print('add pbv')
#        if not 'pbv_%s' % pbv2_days in _.columns: 
#            _=add_pbv(_, pbv_days=pbv2_days)            
#        g2_overlay_col=['snr_mid', 'pbv_%s' % pbv2_days]
#        display('pbv snr:', _[g2_overlay_col])
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
#    cols=['breakout_long', 'output_flag_2',  'breakout_comb_flag_3', 'mmfilter_flag_4']
#    cols=['breakout_long', 'vcomb_flag_2',  'breakout_comb_flag_3', 'mmfilter_flag_4']
    print('xxx:',_.columns)
#    flag_df=_[cols].copy()
#    flag_df=expand_flag_pdf(_[cols])
    flag_df=expand_flag_pdf(_, exclude=['f_breakout_comb'])
#    display('no expand:',flag_df.tail(), _.columns)
    if _ is None:
        print('no date for ',ticker,' to review')
        return None
    from datetime import datetime as dt
    if datestr is None:
        ts=pd.to_datetime('today')
    else:
        ts=pd.to_datetime(datestr)
    dateback_ts=ts-pd.to_timedelta('365 days')
    taParams=get_default_plot_params()
    if not df_extra is None:
        for col in ex_cols:
            _[col]=0
            _[col]=df_extra[col]
        _.bfill(inplace=True)
        _.ffill(inplace=True)
        
    row_idx=_.index[-1]
    _['Close']=_['Adj Close']
    _.ffill(inplace=True)
    close=_.iloc[-1]['Close']
    close2=_.iloc[-6]['Close']
    chg_5d=[close-close2]/close2*100
    #extra_tag.append(f'{close} from {close2} in 5days ({chg_5d})')
    extra_tag.append('%.3f from %.3f in 5days %.3f)' % (close, close2, chg_5d))
    extra_tag.append('%s %s' % (ticker, datestr))
#    print(extra_tag, dateback_ts, ts)
    max250d=max(_.loc[dateback_ts:ts].Close)
    min250d=min(_.loc[dateback_ts:ts].Close)
    diff2max=(max250d-close)
    diff2min=(close-min250d)
    extra_tag.append('min:%.2f, max:%.2f, cur:%.2f' % (min250d, max250d, close) )
    extra_tag.append('diff(%.2f, %.2f,  %.2f)' % ( diff2min,  close, diff2max) )
    if chart_tag is None:
        chart_tag='%s label %s' % (ticker, datestr)
    bdth_df=gen_breadth_df_by_ticker(ticker, dateback_ts)
    if bdth_df is None:
        bdth_df=pd.DataFrame()
    if len(hl_flags)>0:
        if type(hl_flags[0])==type(''):
            chart_tag='mid:%s flg:%s' % (datestr, hl_flags)
    try:
        if len(ex_cols)==0:
            df_arr=[flag_df, bdth_df]
        else:
            df_arr=[flag_df, df_extra[ex_cols], bdth_df]
        if 'rs_rank' in pdf.columns:
            pdf['rs_rank_ma20']=pdf.rs_rank.rolling(20, min_periods=1).mean()
            df_arr.append(pdf[['rs_rank', 'rs_rank_ma20']])
        if 'rs' in pdf.columns:
            pdf['rs_ma20']=pdf.rs.rolling(20, min_periods=1).mean()
            df_arr.append(pdf[['rs', 'rs_ma20']])
        if  'tm_coc' in pdf.columns:
            coc_cols=['tm_coc', 'vol_coc']
            if  'tm_coc_flag' in pdf.columns:
                coc_cols.append('tm_coc_flag')
                coc_cols.append('vol_coc_flag')
                pdf['th_4']=4.0
                coc_cols.append('th_4')
            df_arr.append(pdf[coc_cols])
        from datalib.addonPlotter import addonPlotter
        from datalib.extremaPatternUtil import divergence_screener, extrema_screener
        tlist=[ticker]
        pset=['ad_bull_div', 'adsoc_bull_div', 'macd_bull_div', 'rsi_bull_div', 'obv_bull_div', 'mfi_bull_div']
        div_ret_dict=divergence_screener(tlist=tlist, ema_list=[3], window_list=[10], local_order=1, ex_cond='up,up', pset=pset,
                plot=False, results=True, startdate=startdate, enddate=enddate, debug=True, remove_overlap=True, resample_rule='2D')

        regrouped_pat_dict, ret_df_dict=addonPlotter.get_extra_df_arr_for_review_addon(div_ret_dict)
        ta_ax_dict={}
        print('about to add div_ta_col xxxx:', regrouped_pat_dict)
        from datalib.tickerTextAnalyzer import tickerTextAnalyzer
        tta=tickerTextAnalyzer()
        print('b4 patternReviewUtil tta')
        senti_df=tta.plot_sent_on_existing_ax(ticker, ax=None)
        print('after patternReviewUtil tta')
        if len(senti_df)>30:
            df_arr.append(senti_df)
        print('df_arr before len is ',len(df_arr))
        for ta_col,_df in ret_df_dict.items():
            print('extra_df_arr appending rows:', ta_col)
            df_arr.append(_df)
            ta_ax_dict[ta_col]=[]
        print('df_arr after len is ',len(df_arr))

        
        ret_dict=charter.plot_df_by_mid_point(ts, _, ticker,  chart_tag=chart_tag, taParams=taParams,  g2_overlay_col=g2_overlay_col, 
                                                  df_arr=df_arr, hl_flags=hl_flags, add_label=True, daycnt=180, extra_tag=extra_tag, return_state=True)
        fname=ret_dict['fname']
        fig=ret_dict['fig']
        ax=ret_dict['ax']
        dfta=ret_dict['dfta']
        more_ax=ret_dict['more_ax'] 

        idx=-1
        for key in list(ta_ax_dict.keys())[::-1]:
            idx=idx-1
            ta_ax_dict[key]=more_ax[idx]
            print('set ta_ax_dict col:',key, more_ax[idx])
        ta_ax_dict['price_ax']=ax
        plot_fib(dfta, plot_bars=len(dfta), minmax_bars=40, ex_info=None,plot=True, ax=ax)
        print('passing ta_ax_dict to addonPlotter:',ta_ax_dict)
        plot_all_with_breakout(ticker, pdf=dfta, barcnt=daycnt,  ax=ax)
        addonPlotter.plot_extrema_pattern_addon(ax, ticker, startdate=dfta.index[0])
        div_ret_dict['pdf']=div_ret_dict['pdf'].loc[startdate:]
        addonPlotter.plot_regrouped_pattern(div_ret_dict, regrouped_pat_dict, ta_ax_dict)
        if not info_df is None:
            print('info_df:',info_df, info_df.dtypes)
            row=info_df.loc[ticker]
            print('info_df row:',row, row.keys())
            entry_date=row['entry_date']
            target_price=row['profit_mit_price']
            stop_price=row['stop_price']
            ymin,ymax=ax.get_ylim()
            if ymax>target_price:
                ax.set_ylim(ymin, ymax)
            ax.axhline(target_price, color='blue')
            ax.axhline(stop_price, color='red')
            ax.axvline(pd.to_datetime(entry_date), color='yellow')
            ax.text(dfta.index[20], target_price, 'target price=%s' % round(target_price,4))
            ax.text(dfta.index[20], stop_price, 'stop_priceprice=%s entrydate:%s' % (round(stop_price,4), entry_date))
    except IndexError as e:
        print('returning None for image fname when review ticker chart index error for ',ticker, e)
        import traceback
        print('exception:', e)
        traceback.print_exc()
        fname=None
    fig.savefig(fname)
    if not debug:     
        return fname
    else:
        ret={'pdf':_, 'df_arr':[flag_df, df_extra[ex_cols], bdth_df], 'imgfname':fname}
        ret_dict['extra_df']=ret
        return ret_dict

def ax_highlight_period(ax, period_dict, color='lightgray'):
    for k in period_dict:
        start=k
        end=period_dict[k]
        ax.axvspan(start, end, color=color)  
        
def group_hlidx_arr(pdf, flag_col=None):                                                                                                                                     
    if flag_col is None:
        hlidx_arr=(pdf>0).index
        delta=pdf.pct_change(1)
    else:
        hlidx_arr=pdf.query('%s>0' % flag_col).index
        delta=pdf[flag_col].pct_change(1)
    prev_idx=hlidx_arr[0]
    print('pt1:',prev_idx)
    ret_dict={}
    ret_dict[prev_idx]=prev_idx
    for idx in hlidx_arr[1:]:
        if delta.loc[idx]==0:
            ret_dict[prev_idx]=idx
        else:
            prev_idx=idx
    return ret_dict

def group_period_dict(period_dict, days=10, startdate=None, enddate=None):
    klist=list(period_dict.keys())
    if startdate is None:
        startdate=pd.to_datetime('20000101')
    else:
        startdate=pd.to_datetime('%s' % startdate)
    if enddate is None:
        enddate=pd.to_datetime('20490101')                            
    else:
        enddate=pd.to_datetime('%s' % enddate)
        
    prev_enddate=period_dict[klist[0]]
    ret_dict={}
#    for k in klist[1:]:
    for k in klist:
        if k<startdate:
            continue
        if k>enddate:
            break
        td=(k-prev_enddate)
        if td.days < days:
#            print('case1, group', td)
            ret_dict[prev_enddate]=period_dict[k]
        else:
#            print('case2, new', td)            
            prev_enddate=k            
            ret_dict[k]=period_dict[k]

    return ret_dict



def review_with_contraction_pattern(ticker, datestr='20190711', flag_cols=['f_vola_contr', 'f_volu_contr'], vola_pctile=0.2, volu_pctile=0.2, pdf=None):
    '''
    return pdf with added signal f_v2_contr
    '''
    t=ticker
    if pdf is None:
        datets1=pd.to_datetime(datestr)+pd.to_timedelta('50 days')
        datets2=pd.to_datetime(datestr)-pd.to_timedelta('400 days')
        pdf=cu.read_quote(t)
        pdf=pdf.loc[datets1:datets2]
#        vola_cols=add_volatility_contract_by_quantile(pdf, smooth_days=5, percentile=vola_pctile)
    #    print(pdf.columns)
#        volu_cols=add_volume_contraction_flag(pdf, smooth_days=5, percentile=volu_pctile)
#        qstr='%s>0 and %s>0' % (vola_cols[-1],volu_cols[-1])
#        flag_cols=[vola_cols[-1],volu_cols[-1], 'f_v2_contr']        
        pdf, flag_cols=add_vcp_features_by_pctile(ticker, volu_pctile=volu_pctile, vola_pctile=vola_pctile)
    from datalib.zcatbreadthUtil import gen_breadth_df_by_ticker
    bdth_df=gen_breadth_df_by_ticker(ticker=ticker, startdate=datestr)
    bdth_df.plot()
    qstr='%s>0 and %s>0' % (flag_cols[0],flag_cols[1])
    print(qstr)
    pdf['f_v2_contr']=pdf.eval(qstr)*1.0
    pdf['cat_bdth']=bdth_df['cat_bdth']
    pdf['all_bdth']=bdth_df['all_bdth']    
#    flag_cols=[vola_cols[-1],volu_cols[-1], 'f_v2_contr']
    flag_cols=['f_volu_contr', 'f_vola_contr', 'f_v2_contr']
    img_fname=quick_pbv_breakout_review_extra(t, pdf=pdf, datestr=datestr, df_extra=pdf, ex_cols=['atr', 'rolling_atr_low', 'rolling_atr_low2'], hl_flags=flag_cols, pbv2_days=120)
    return pdf

def find_trend_range(pdf, datestr, span_days=120, trend='up', daycnt=15, include_subdf=False):
    ts=pd.to_datetime(datestr)
    td=pd.to_timedelta('%s days' % span_days)
    pdf['pct_chg_n']=pdf.Close.pct_change(daycnt).shift(-daycnt)*100    
    startdate=ts-td
    enddate=ts+td
    subdf=pdf.loc[startdate:enddate]

    subdf1=pdf.loc[startdate:datestr]
    subdf2=pdf.loc[datestr:enddate]
    cols=['pct_chg_n', 'Close']
    ret={}
    ret['search_start']=startdate
    ret['search_end']=enddate
    ret['span_days_input']=span_days    
    
#    subdf[cols].plot()
    if trend=='up':
        idmax=subdf2.Close.idxmax()
        idmin=subdf1.Close.idxmin()

    else:
        idmax=subdf1.Close.idxmax()
        idmin=subdf2.Close.idxmin()        
        
        
    price_max=subdf.loc[idmax].Close
    price_min=subdf.loc[idmin].Close
#    print('idmin:',idmin)
#    print('idmax:',idmax)
    
    if idmax>idmin:

        ret['trend']='up'
        ret['trendstart']=idmin
        ret['trendend']=idmax
        ret['span']=idmax-idmin        
        ret['startprice']=price_min
        ret['endprice']=price_max
        ret['pct_chg']=price_max/price_min-1
        if include_subdf:
            ret['subdf']=subdf.loc[idmin:idmax]
    else:
        ret['trend']='down'
        ret['trendstart']=idmax
        ret['trendend']=idmin      
        ret['span']=idmin-idmax
        ret['startprice']=price_max
        ret['endprice']=price_min
        ret['pct_chg']=price_min/price_max-1
        ret['subdf']=subdf.loc[idmax:idmin]

    
    return ret
def find_trend_range_extended(pdf, datestr, max_days=150, trend='up', start_days=20):
    prev_range=find_trend_range(pdf, datestr, span_days=start_days, trend=trend)
    max_range=prev_range
    for days in range(start_days, max_days, 7):
        if days<start_days:
            continue
        if days>max_days:
            return max_range
        print('days:',days, prev_range['pct_chg'])
        trend_range=find_trend_range(pdf, datestr, span_days=days, trend=trend)
        if prev_range['trend']=='up':
            if prev_range['pct_chg']>=trend_range['pct_chg']:
                print('no improvement on bigger search range up, return ', days)
                break
        if prev_range['trend']=='down':
            if prev_range['pct_chg']<=trend_range['pct_chg']:
                print('no improvement on bigger search range down, return ', days)
                break

        max_range=trend_range
        prev_range=trend_range
    return max_range


def scan_top_movement(pdf, daycnt=40, percentile=0.95, lookback_period=100, th=0.4):
    return scan_top_movement_by_pct_chg(pdf, daycnt=daycnt, percentile=percentile, lookback_period=lookback_period, th=th)


def scan_worse_movement(pdf, daycnt=40, percentile=0.05, lookback_period=100, th=-0.2):
    return scan_worse_movement_by_pct_chg(pdf, daycnt=daycnt, percentile=percentile, lookback_period=lookback_period, th=th)


def scan_top_movement_by_reg_slope(pdf, daycnt=20, th=0.01):
    startprice=pdf.iloc[0].Close
    tmp=np.log(pdf.Close/startprice)    
#    pdf['reg_slope_n']=talib.LINEARREG_SLOPE(tmp, daycnt).shift(-daycnt+5)
    pdf['reg_slope_n']=talib.LINEARREG_SLOPE(tmp, daycnt).shift(-daycnt)    
    if th>0:
        pdf['fn_top_reg_slope_n']=pdf.eval('reg_slope_n>%s' % th)*1.0
        cols=['reg_slope_n', 'fn_top_reg_slope_n']
    if th<0:
        pdf['fn_top_reg_slope_n']=pdf.eval('reg_slope_n<%s' % th)*1.0
        cols=['reg_slope_n', 'fn_top_reg_slope_n']
    return cols



def scan_top_movement_by_pct_chg(pdf, daycnt=40, percentile=0.95, lookback_period=100, th=0.4):
    pdf['pct_chg_n']=pdf.Close.pct_change(daycnt).shift(-daycnt)
    pdf['top_pct_chg']=pdf.pct_chg_n.rolling(lookback_period, min_periods=1).quantile(percentile)
    pdf['fn_top_pct_chg']=pdf.eval('pct_chg_n>top_pct_chg and pct_chg_n>%s' % th)*1.0
    cols=['pct_chg_n', 'top_pct_chg', 'fn_top_pct_chg']
    return cols

def scan_worse_movement_by_pct_chg(pdf, daycnt=30, percentile=0.05, lookback_period=100, th=-0.2):
    pdf['pct_chg_n']=pdf.Close.pct_change(daycnt).shift(-daycnt)
    pdf['worst_pct_chg']=pdf.pct_chg_n.rolling(lookback_period, min_periods=1).quantile(percentile)
    pdf['fn_worst_pct_chg']=pdf.eval('pct_chg_n<worst_pct_chg and pct_chg_n<%s' % th)*1.0
    cols=['pct_chg_n', 'worst_pct_chg', 'fn_worst_pct_chg']
    return cols


def highlight_plot_by_flag(ax, pdf, col_name=None, color='lightgray', startdate=None, enddate=None):
    # or from rrgv3/chartPlotterXin1.py 
    #from rrgv3/chartPlotterXin1 import highlight_plot_by_flag
    if col_name is None:
        subdf=pdf
    else:
        subdf=pdf[col_name]
    period_dict=group_hlidx_arr(subdf)
    period_dict2=group_period_dict(period_dict, startdate=startdate, enddate=enddate)
    ax_highlight_period(ax, period_dict2, color=color)
    return period_dict2

def test_verify_top_movement_with_contraction_pattern(ticker='VIPS', startdate='20140101', worst_move=False, max_cnt=999, plot_ta=False):
    if not plot_ta:
        pdf, flag_cols=add_vcp_features_by_pctile(ticker, volu_pctile=0.25, vola_pctile=0.25, smooth_days=5) 
    else:
        pdf=cu.read_quote(ticker)
    if worst_move:
        cols=scan_top_movement_by_reg_slope(pdf, daycnt=20, th=-0.008)
#        cols=scan_worse_movement(pdf)
        days=30
    else:
        cols=scan_top_movement_by_reg_slope(pdf, daycnt=20, th=0.01)
#        cols=scan_top_movement(pdf)
        days=40
    label_subdf=pdf.loc[startdate:][cols[-1]]
    signal_subdf=pdf.loc[startdate:][flag_cols[-1]]
#    display('subdf:',subdf.head())
    #ax=pdf.iloc[-barcnt:].Close.plot()
#    display(pdf.tail())
    ax=pdf[cols].plot()
    
    # this line consolidate the signal
    label_hl_dict_=highlight_plot_by_flag(ax, label_subdf, color='lightgreen')
    # this line trace back the beginning of the trend
    #XXAA
    label_hl_value=[find_trend_range_extended(pdf, k, start_days=20) for k in list(label_hl_dict_.keys())]
    label_hl_keys=[v['trendstart'] for v in label_hl_value]
    label_hl_dict=dict(zip(label_hl_keys, label_hl_value))
#    display(label_hl_dict)
    label_subdf=pd.DataFrame.from_dict(label_hl_dict, orient='index')
    signal_hl_dict=highlight_plot_by_flag(ax, signal_subdf, color='lightblue')    

    i=0
#    klist=list(label_hl_dict.keys())
    klist=list(signal_hl_dict.keys())
    ret_dict={}
    ret_dict['signal_subdf']=signal_subdf.copy()
    ret_dict['label_subdf']=label_subdf.copy()    
    ret_dict['label_hl_dict']=label_hl_dict.copy()
    ret_dict['signal_hl_dict']=signal_hl_dict.copy()    
    output_match_dict={}
    major_movement_subdf_dict={}
    all_row_dict={}
    factor=1.0
#    return label_hl_dict
#    return None
#    pdf=cu.read_quote(ticker)
#    for k in klist[1:]:
    for k in klist[3:]:
        
        signal_startdate=k
#        enddate=label_hl_dict[k]
        factor=factor*(1+i*0.1)  
        _=label_subdf.loc[signal_startdate:]
        if len(_)==0:
            print('no next trend')
            continue
        next_label=_.iloc[0]
#        trend_range=find_trend_range_extended(pdf, startdate, start_days=20)
        trend_startdate=next_label.trendstart
        trend_enddate=next_label.trendend
        print('startdate %s, enddate %s' % (trend_startdate, trend_enddate))        
        i=i+1

        if i>max_cnt:
            break
    #    upsize_start=pdf.loc[ret[k]].Close.pct_change(days)
    #    upsize_end=pdf.loc[k].Close.pct_change(days)    

    #    mean_ret=list(pdf.loc[k:ret[k]].Close.pct_change(days))
    #    mean_ret=list(pdf.loc[k:ret[k]].Close.pct_change(1))
        movement_subdf=pdf.loc[signal_startdate:trend_startdate]
        if len(movement_subdf)==0:
            continue
        mean_ret=pdf.Close.pct_change(days).loc[signal_startdate:trend_startdate].mean()
        row_dict={}
        row_dict['ticker']=ticker
        row_dict['signal_startdate']=signal_startdate
        row_dict['trend_startdate']=trend_startdate
        row_dict['n_days']=days
        row_dict['mean_n_days_ret']=mean_ret
        row_dict['startprice']=movement_subdf.iloc[0].Close
        row_dict['endprice']=movement_subdf.iloc[-1].Close        
        row_dict['trading_days_cnt']=len(movement_subdf)
        row_dict['td2signal']=(trend_startdate-signal_startdate).days
        row_dict['pct_chg']=(row_dict['endprice']-row_dict['startprice'])/row_dict['startprice']
        major_movement_subdf_dict[k]=movement_subdf
#        print('xasas mean_ret %s days starting %s is %s subdf start %s' % (days, signal_startdate, trend_startdate, movement_subdf.index[0]))
        ts=pd.to_datetime(startdate)
        td=pd.to_timedelta('50 days')
#        ts2=ts-td
        all_row_dict[signal_startdate]=row_dict

        if plot_ta:
            feat_subdf=review_with_contraction_pattern(ticker, datestr=datestr)
            signal_datelist=list(feat_subdf.loc[ts2:datestr].query('f_v2_contr>0').index)
            if len(signal_datelist)==0:
                continue
            signal_date_close=feat_subdf.loc[signal_datelist[0]].Close
            end_date_close=feat_subdf.loc[enddate].Close
            if worst_move:
                if signal_date_close/end_date_close>1.1:
                    output_match_dict[startdate]=signal_datelist
            else:
                if end_date_close/signal_date_close>1.1:
                    output_match_dict[startdate]=signal_datelist
        else:
            (movement_subdf.Close*factor).plot()
            
            

#    ret_dict['all_signal']= group_hlidx_arr(subdf['f_v2_contr'])
    ret_dict['all_signal']= group_hlidx_arr(signal_subdf)
#    ret_dict['all_trend']= group_hlidx_arr(label_subdf)    

    ret_dict['output_match_dict']=output_match_dict
    ret_dict['major_movement_subdf_dict']=major_movement_subdf_dict
    ret_dict['movement_klist']=klist
    all_row_table_df=pd.DataFrame.from_dict(all_row_dict, orient='index')
    ret_dict['all_row_dict']=all_row_dict
#    ret_dict['signal_datelist']=signal_highlight_dict
    ret_dict['movement_table']=all_row_table_df
    ret_dict['flag_cols']=cols
    ret_dict['pdf']=pdf
    return ret_dict

def test():
    

    t='FLGT'
    subdf=select_dict[t]
    print(subdf)
    _=rs_rank_df[t]
    _.loc[:]=0
    _.loc[subdf.index]=1
    period_dict=group_hlidx_arr(subdf)
    period_dict2=group_period_dict(period_dict, startdate='20200101')
    #period_dict2=group_period_dict(period_dict, enddate='20200301')
#    print('b4:',period_dict)
#    print('after:', period_dict2)
    tpdf=cu.read_quote(t)
    ax=tpdf.iloc[-500:].Close.plot()
    ax_highlight_period(ax, period_dict2)
    
    
def test2():
    t='TDOC'
    pdf=cu.read_quote(t)
    vola_cols=add_volatility_contract_by_quantile(pdf)
    print(pdf.columns)
    volu_cols=add_volume_contraction_flag(pdf)
    flag_cols=[vola_cols[-1],volu_cols[-1] ]
    print('flag_cols:',flag_cols)
    #ret=quick_pbv_breakout_review_extra(t, pdf=pdfx, datestr='20200711', df_extra=pdf, ex_cols=['ratr', 'rolling_ratr_low'], hl_flags=pdf['hv_rflag_n'])
    ret=quick_pbv_breakout_review_extra(t, pdf=pdf, datestr='20200711', df_extra=pdf, ex_cols=['atr', 'rolling_atr_low'], hl_flags=flag_cols)
    
    

def test3():
    pdf2=cu.read_quote('NVTA')
    cols=add_volume_contraction_flag(pdf2)
    subdf=pdf.iloc[-barcnt:][cols[-1]]
    ax=subdf[cols].plot()
    s
    #subdf=pdfx['vflag'].iloc[-400:]
    #ax=pdfx[cols].iloc[-400:].plot()
    highlight_plot_by_flag(ax, subdf[cols[-1]])
#test2()
