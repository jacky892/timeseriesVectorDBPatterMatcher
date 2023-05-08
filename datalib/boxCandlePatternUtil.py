import pandas_ta
from datalib.commonUtil import commonUtil as cu

def get_candle_rankings():
    candle_rankings = {
        "CDL3LINESTRIKE_Bull": 1,
        "CDL3LINESTRIKE_Bear": 2,
        "CDL3BLACKCROWS_Bull": 3,
        "CDL3BLACKCROWS_Bear": 3,
        "CDLEVENINGSTAR_Bull": 4,
        "CDLEVENINGSTAR_Bear": 4,
        "CDLTASUKIGAP_Bull": 5,
        "CDLTASUKIGAP_Bear": 5,
        "CDLINVERTEDHAMMER_Bull": 6,
        "CDLINVERTEDHAMMER_Bear": 6,
        "CDLMATCHINGLOW_Bull": 7,
        "CDLMATCHINGLOW_Bear": 7,
        "CDLABANDONEDBABY_Bull": 8,
        "CDLABANDONEDBABY_Bear": 8,
        "CDLBREAKAWAY_Bull": 10,
        "CDLBREAKAWAY_Bear": 10,
        "CDLMORNINGSTAR_Bull": 12,
        "CDLMORNINGSTAR_Bear": 12,
        "CDLPIERCING_Bull": 13,
        "CDLPIERCING_Bear": 13,
        "CDLSTICKSANDWICH_Bull": 14,
        "CDLSTICKSANDWICH_Bear": 14,
        "CDLTHRUSTING_Bull": 15,
        "CDLTHRUSTING_Bear": 15,
        "CDLINNECK_Bull": 17,
        "CDLINNECK_Bear": 17,
        'CDLSTALLEDPATTERN_Bear':20,
        'CDLSTALLEDPATTERN_Bull':20,
        'CDLLONGLINE_Bull':20,
        'CDLLONGLINE_Bear':20,                
        'CDLSHORTLINE_Bull':20,
        'CDLSHORTLINE_Bear':20,       
        "CDL3INSIDE_Bull": 20,
        'CDLCOUNTERATTACK_Bull':20,
        'CDLCOUNTERATTACK_Bear':20,        
        "CDL3INSIDE_Bear": 56,
        "CDLHOMINGPIGEON_Bull": 21,
        "CDLHOMINGPIGEON_Bear": 21,
        "CDLDARKCLOUDCOVER_Bull": 22,
        "CDLDARKCLOUDCOVER_Bear": 22,
        "CDLIDENTICAL3CROWS_Bull": 24,
        "CDLIDENTICAL3CROWS_Bear": 24,
        "CDLMORNINGDOJISTAR_Bull": 25,
        "CDLMORNINGDOJISTAR_Bear": 25,
        "CDLXSIDEGAP3METHODS_Bull": 27,
        "CDLXSIDEGAP3METHODS_Bear": 26,
        "CDLTRISTAR_Bull": 28,
        "CDLTRISTAR_Bear": 76,
        "CDLGAPSIDESIDEWHITE_Bull": 46,
        "CDLGAPSIDESIDEWHITE_Bear": 29,
        "CDLEVENINGDOJISTAR_Bull": 30,
        "CDLEVENINGDOJISTAR_Bear": 30,
        "CDL3WHITESOLDIERS_Bull": 32,
        "CDL3WHITESOLDIERS_Bear": 32,
        "CDLONNECK_Bull": 33,
        "CDLONNECK_Bear": 33,
        "CDL3OUTSIDE_Bull": 34,
        "CDL3OUTSIDE_Bear": 39,
        "CDLRICKSHAWMAN_Bull": 35,
        "CDLRICKSHAWMAN_Bear": 35,
        "CDLSEPARATINGLINES_Bull": 36,
        "CDLSEPARATINGLINES_Bear": 40,
        "CDLLONGLEGGEDDOJI_Bull": 37,
        "CDLLONGLEGGEDDOJI_Bear": 37,
        "CDLHARAMI_Bull": 38,
        "CDLHARAMI_Bear": 72,
        "CDLLADDERBOTTOM_Bull": 41,
        "CDLLADDERBOTTOM_Bear": 41,
        "CDLCLOSINGMARUBOZU_Bull": 70,
        "CDLCLOSINGMARUBOZU_Bear": 43,
        "CDLTAKURI_Bull": 47,
        "CDLTAKURI_Bear": 47,
        "CDLDOJISTAR_Bull": 49,
        "CDLDOJISTAR_Bear": 51,
        "CDLHARAMICROSS_Bull": 50,
        "CDLHARAMICROSS_Bear": 80,
        "CDLADVANCEBLOCK_Bull": 54,
        "CDLADVANCEBLOCK_Bear": 54,
        "CDLSHOOTINGSTAR_Bull": 55,
        "CDLSHOOTINGSTAR_Bear": 55,
        "CDLMARUBOZU_Bull": 71,
        "CDLMARUBOZU_Bear": 57,
        "CDLUNIQUE3RIVER_Bull": 60,
        "CDLUNIQUE3RIVER_Bear": 60,
        "CDL2CROWS_Bull": 61,
        "CDL2CROWS_Bear": 61,
        "CDLBELTHOLD_Bull": 62,
        "CDLBELTHOLD_Bear": 63,
        "CDLHAMMER_Bull": 65,
        "CDLHAMMER_Bear": 65,
        "CDLHIGHWAVE_Bull": 67,
        "CDLHIGHWAVE_Bear": 67,
        "CDLSPINNINGTOP_Bull": 69,
        "CDLSPINNINGTOP_Bear": 73,
        "CDLUPSIDEGAP2CROWS_Bull": 74,
        "CDLUPSIDEGAP2CROWS_Bear": 74,
        "CDLGRAVESTONEDOJI_Bull": 77,
        "CDLGRAVESTONEDOJI_Bear": 77,
        "CDLHIKKAKEMOD_Bull": 82,
        "CDLHIKKAKEMOD_Bear": 81,
        "CDLHIKKAKE_Bull": 85,
        "CDLHIKKAKE_Bear": 83,
        "CDLENGULFING_Bull": 84,
        "CDLENGULFING_Bear": 91,
        "CDLMATHOLD_Bull": 86,
        "CDLMATHOLD_Bear": 86,
        "CDLHANGINGMAN_Bull": 87,
        "CDLHANGINGMAN_Bear": 87,
        "CDLRISEFALL3METHODS_Bull": 94,
        "CDLRISEFALL3METHODS_Bear": 89,
        "CDLKICKING_Bull": 96,
        "CDLKICKINGBYLENGTH_Bull": 96,
        "CDLKICKING_Bear": 102,
        "CDLKICKINGBYLENGTH_Bear": 102,
        "CDLDRAGONFLYDOJI_Bull": 98,
        "CDLDRAGONFLYDOJI_Bear": 98,
        "CDLCONCEALBABYSWALL_Bull": 101,
        "CDLCONCEALBABYSWALL_Bear": 101,
        "CDL3STARSINSOUTH_Bull": 103,
        "CDL3STARSINSOUTH_Bear": 103,
        "CDLDOJI_Bull": 104,
        "CDLDOJI_Bear": 104
    }
    new_rank={'CDL_%s' % col[3:]:candle_rankings[col] for col in candle_rankings.keys()}
                                                     
    #return candle_rankings
    return new_rank


def get_matched_pattern(subdf_, resample_rule='3D'):
    import pandas as pd
    import numpy as np
    import pandas_ta
    from vcplib.vcpUtil import resample_df
    cols=['Open', 'High', 'Low', 'Close', 'Volume']

    #subdf=resample_df(subdf_, rule=resample_rule).dropna().copy()
    if not resample_rule=='1D':
        subdf=resample_df(subdf_, rule=resample_rule).ffill().copy()
    else:
        subdf=subdf_.copy()
#    subdf=subdf_
#    print(subdf)
    x=subdf.ta.cdl_pattern()
#    op, hi, lo, cl=subdf['Open'], subdf['High'], subdf['Low'], subdf['Close']
    df=x

    from itertools import compress
    df[f'r{resample_rule}_candlestick_pattern'] = np.nan
    df[f'r{resample_rule}_candlestick_match_count'] = np.nan
    candle_rankings=get_candle_rankings()
    ranked_candle_names=[col[:-5] for col in candle_rankings.keys()]
    my_candle_ranked=[col for col in x.columns if col in ranked_candle_names]
    #candle_names=[col for col in x.columns if col in candle_names]
    #print(x.columns[:2], candle_names1, candle_names2)
#    print('CDL_DOJI_10_0.1' in candle_names)
    
    for index, row in df.iterrows():
        cbar=(row[my_candle_ranked].fillna(0))

        #print(cbar, len(cbar), sum(cbar==0))
        if len(cbar) - sum(cbar == 0) == 0:
            df.loc[index,f'r{resample_rule}_candlestick_pattern'] = "NO_PATTERN"
            df.loc[index, f'r{resample_rule}_candlestick_match_count'] = 0
        # single pattern found
        elif len(cbar) - sum(cbar == 0) == 1:                                                                                                     
            # bull pattern 100 or 200
            #print('ca')
            if any(cbar.values > 0):
                pattern = list(compress(cbar.keys(), cbar.values != 0))[0] + '_Bull'
                df.loc[index, f'r{resample_rule}_candlestick_pattern'] = pattern
                df.loc[index, f'r{resample_rule}_candlestick_pattern_score'] = cbar.fillna(0).sum()                
                df.loc[index, f'r{resample_rule}_candlestick_match_count'] = 1
            # bear pattern -100 or -200
            else:
                pattern = list(compress(cbar.keys(), cbar.values != 0))[0] + '_Bear'
                df.loc[index, f'r{resample_rule}_candlestick_pattern'] = pattern
                df.loc[index, f'r{resample_rule}_candlestick_pattern_score'] = cbar.fillna(0).sum()
                df.loc[index, f'r{resample_rule}_candlestick_match_count'] = 1
        # multiple patterns matched -- select best performance
        else:
            # filter out pattern names from bool list of values
            #print('c2a')
            patterns = list(compress(cbar.keys(), cbar.values != 0))
            container = []
#            print('patters:',pattern, container, candle_rankings)
#            print('patters:',pattern, container)
            for pattern in patterns:
                if row[pattern] > 0:
#                    print('bull row patter is ',row[pattern], pattern)
                    pat_bullbear=pattern + '_Bull'
                    container.append(pat_bullbear)
                else:
#                    print('bear row patter is ',row[pattern], pattern)
                    pat_bullbear=pattern + '_Bear'
                    container.append(pat_bullbear)

                if pat_bullbear not in candle_rankings.keys():
                    candle_rankings['pat_bullbear']=20
            in_rank_list = {p:p in candle_rankings.keys() for p in container}
#            print(in_rank_list, [p for p in container])
#            print(f'container {container}, inlist:{in_rank_list}')
            rank_list = [candle_rankings[p] for p in container]

            if len(rank_list) == len(container):
                #print('case8')
                rank_index_best = rank_list.index(min(rank_list))
                #df.loc[index, f'r{resample_rule}_candlestick_pattern'] = container[rank_index_best]
                df.loc[index, f'r{resample_rule}_candlestick_pattern'] = '%s' % list(container)
                df.loc[index, f'r{resample_rule}_candlestick_pattern_score'] = cbar.fillna(0).sum()                
                df.loc[index, f'r{resample_rule}_candlestick_match_count'] = len(container)
    # clean up candle columns
    df.fillna(0, inplace=True)
    box_rise=(subdf.Close[-1]-subdf.Close[0])>0
    __=list(df.columns)
    pattern_score=df.iloc[-1][f'r{resample_rule}_candlestick_pattern_score']
    pattern_list=df.iloc[-1][f'r{resample_rule}_candlestick_pattern']
#    print(subdf.shape)
#    return df.drop(candle_names, axis = 1).copy()
#    print([v for v in __ if v[:4]=='CDL_'])
    ret_dict={}
    return df
    



def get_multi_tf_candle_pattern_summary(subdf):
    import pandas as pd
    total_bar=len(subdf)
    ndf=pd.DataFrame()    
    resample_list=('1D', '2D', '3D','4D', '5D')
    all_pattern_df_dict={}
    over_all_score=0
    ret_dict={}
    for rule in resample_list:
#        print('rule is ',rule)
        _=subdf.copy()   
        try:
            df=get_matched_pattern(_, resample_rule=rule) 
#            print(df.shape)
            _=df[f'r{rule}_candlestick_pattern_score'].iloc[-3:]
#            print(_)
            curscore=_.sum()
#            print(curscore)
        except:
            curscore=0
        over_all_score+=curscore
#    print('raw overall candle score:',over_all_score)        
    ret_dict['net_cdl_score']=over_all_score/100
#    print('after adjust overall candle score:',ret_dict)        
    return ret_dict

class boxCandlePatternUtil:                                                                                                                                                 
    @staticmethod
    def get_multi_tf_candle_pattern_summary(subdf):
        return get_multi_tf_candle_pattern_summary(subdf)
    
def test():
    ticker='TSLA'
    #cu.download_quote(ticker)
    pdf=cu.read_quote('TSLA')
    xdf=pdf.ta.cdl_pattern()
    print(xdf)
    cdl_score_dict=boxCandlePatternUtil.get_multi_tf_candle_pattern_summary(pdf.iloc[-100:])
    print(cdl_score_dict['net_cdl_score'])
    net_cdl_score=cdl_score_dict['net_cdl_score']
    print(net_cdl_score)
#test()
