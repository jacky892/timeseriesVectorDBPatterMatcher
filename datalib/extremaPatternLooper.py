import numpy as np
import pandas as pd
import logging
from collections import defaultdict

class extremaPatternLooper:
    
    @staticmethod
    def find_higher_high_low_patterns(max_min, high_col='High', low_col='Low'):  
        patterns = defaultdict(list)

        # Window range is 5 units
        size=7
        for i in range(size, len(max_min)):  
            window = max_min.iloc[i-size:i]

            # Pattern must play out in less than n units
            td= (window.index[-1] - window.index[0]) 
    #        print('td is ',td, type(1))
            if type(td)==type(pd.to_timedelta('1 days')):
    #            print('path 1 date index')
                if (window.index[-1] - window.index[0]) > pd.to_timedelta('100 days'):      
                    continue          

                high_win=window.query('minmax_type=="max"')[high_col]
                low_win=window.query('minmax_type=="min"')[low_col]
                if abs(len(high_win)-len(low_win))>2:
                    continue
    #            print('high win, low win ', high_win, low_win)
                prev=high_win[0]
                breakloop=False
                for high in high_win[1:]:
                    if high<=prev or high>(prev*1.1):
    #                    print('break higher high, skip', window.index[0])
                        breakloop=True
                        break
    #                print('high is ',high, prev)                    
                    prev=high
                if breakloop:
                    continue
                prev=low_win[0]
                breakloop=False            
                for low in low_win[1:]:
                    if low<=prev or low>(prev*1.1):
    #                    print('break higher low, skip', window.index[0])
                        breakloop=True
                        break
    #                print('low is ',low, prev)                                    
                    prev=low

                if breakloop:
                    continue

                if window.iloc[-1].minmax_type=='max':
                    continue
                if high<low:
    #                print('final break of hhhl',window.index[0] )
                    continue
    #            print('found higherhl patten!', window.index[0])
                patterns['hhhl'].append((window.index[0], window.index[-1]))

        return patterns

    @staticmethod
    def find_higher_low_patterns(min_only, low_col='Low', ticker='_'):  
        patterns = defaultdict(list)    
        size=4
        for i in range(size, len(min_only)):  
            window = min_only.iloc[i-size:i]
            if (window.index[-1] - window.index[0]) > pd.to_timedelta('100 days'):              
                continue
            a, b, c, d = tuple(window.iloc[0:size][low_col])  
    #        print(tuple(window.iloc[0:size].minmax_type))
    #            if a<b and b<c and c<d and abs(b-d)<=np.mean([b,d])*0.02:
            if a<b and b<c and c<d:
    #            print('found higherlow patten!')
                patterns['hl'].append((window.index[0], window.index[-1]))

        return patterns
    @staticmethod
    def find_higher_high_patterns(max_only, high_col='Close', ticker='_'):  
        patterns = defaultdict(list)    
        size=4
        for i in range(size, len(max_only)):  
            window = max_only.iloc[i-size:i]

            if (window.index[-1] - window.index[0]) > pd.to_timedelta('100 days'):              
                continue

            a, b, c, d = tuple(window.iloc[0:size][high_col])  
    #        print(tuple(window.iloc[0:size].minmax_type))
    #            if a<b and b<c and c<d and abs(b,ticker-d)<=np.mean([b,d])*0.02:
            if a<b and b<c and c<d:
    #            print('found higherhigh patten!')
                patterns['hh'].append((window.index[0], window.index[-1]))

        return patterns
    @staticmethod
    def find_lower_high_patterns(max_only, high_col='Close'):  
        patterns = defaultdict(list)    
        size=4
        for i in range(size, len(max_only)):  
            window = max_only.iloc[i-size:i]

            if (window.index[-1] - window.index[0]) > pd.to_timedelta('100 days'):              
                continue

            a, b, c, d = tuple(window.iloc[0:size][high_col])  
    #        print(tuple(window.iloc[0:size].minmax_type))
    #            if a<b and b<c and c<d and abs(b-d)<=np.mean([b,d])*0.02:
            if a>b and b>c and c>d and  abs(b-c)<=np.mean([b,c])*0.02:
    #            print('found lower high patten!')
                patterns['lh'].append((window.index[0], window.index[-1]))

        return patterns

    @staticmethod
    def find_lower_low_patterns(min_only, low_col='Low'):  
        patterns = defaultdict(list)    
        size=4
        for i in range(size, len(min_only)):  
            window = min_only.iloc[i-size:i]

            if (window.index[-1] - window.index[0]) > pd.to_timedelta('100 days'):              
                continue

            a, b, c, d = tuple(window.iloc[0:size][low_col])  
    #        print(tuple(window.iloc[0:size].minmax_type))
    #            if a<b and b<c and c<d and abs(b-d)<=np.mean([b,d])*0.02:
            if a>b and b>c and c>d and  abs(b-c)<=np.mean([b,c])*0.02:
    #            print('found lower high patten!')
                patterns['ll'].append((window.index[0], window.index[-1]))

        return patterns

    @staticmethod
    def find_ihs_patterns(max_min, colname='Close', ticker='_'):  

        patterns = defaultdict(list)

        # Window range is 5 units
        for i in range(5, len(max_min)):  
            window = max_min.iloc[i-5:i]

            # Pattern must play out in less than n units
            td= (window.index[-1] - window.index[0]) 
    #        print('td is ',td, type(1))
            if type(td)==type(pd.to_timedelta('1 days')):
    #            print('path 1 date index')
                if (window.index[-1] - window.index[0]) > pd.to_timedelta('150 days'):      
                    continue               
                a, b, c, d, e = tuple(window.iloc[0:5][colname])
                a_, b_, c_, d_, e_ = tuple(window.iloc[0:5]['minmax_type'])                        
            else: # number based indes
#                if td > 100:      
#                    continue   
                a, b, c, d, e = window.iloc[0:5][colname]
                a_, b_, c_, d_, e_ = tuple(window.iloc[0:5]['minmax_type'])                    

            # IHS
            if a<b and c<a and c<e and c<d and e<d and abs(b-d)<=np.mean([b,d])*0.02 and abs(b-c)>=np.mean([b,c])*0.03 and c_=='min':
                #print('found ihs patten!')
                patterns['ihs'].append((window.index[0], window.index[-1]))

        return patterns


    @staticmethod
    def find_hs_patterns(max_min, colname='Close', ticker='_'):  
        patterns = defaultdict(list)

        # Window range is 5 units
        for i in range(5, len(max_min)):  
            window = max_min.iloc[i-5:i]

            # Pattern must play out in less than n units
            td= (window.index[-1] - window.index[0]) 
    #        print('td is ',td, type(1))
            if type(td)==type(pd.to_timedelta('1 days')):
    #            print('path 1 date index')
                if (window.index[-1] - window.index[0]) > pd.to_timedelta('150 days'):      
                    continue               
                a, b, c, d, e = tuple(window.iloc[0:5][colname])
                a_, b_, c_, d_, e_ = tuple(window.iloc[0:5]['minmax_type'])            
            else: # number based indes
                if td > 100:      
                    continue   
                a, b, c, d, e = window.iloc[0:5][colname]
                a_, b_, c_, d_, e_ = tuple(window.iloc[0:5]['minmax_type'])                            

            # HS
            if a>b and c>a and c>e and c>d and e>d and c>e and d<b and c_=='max' and abs(b-c)>=np.mean([b,c])*0.03:
                if abs(a-b)<=np.mean([b,d])*0.02:
                    continue
                if abs(d-e)<=np.mean([e,e])*0.02:
                    continue

    #            print('found hs patten!', window.index[0])
                patterns['hs'].append((window.index[0], window.index[-1]))

        return patterns


    @staticmethod
    def check_low_vol_breakup(max_points, min_points, col1='High', col2='Low'):
        high_low_diff_list=[0.04]
    #    for idx in min_points.index:
    #        min_pt=min_points.loc[idx]
    #        _=max_points.loc[idx:]
    #        nidx=_.index[0]
    #        next_max_pt=_.iloc[0]
        if len(max_points)<4:
    #        print('too few max points ', len(max_points))
            return False
        if len(min_points)<3:
    #        print('too few min points ', len(min_points))
            return False

    #    print('len of max points ', len(max_points))

        for idx in max_points.index[:-1]:
            max_pt=max_points.loc[idx]
            _=min_points.loc[idx:]
            if len(_)==0:
    #            print('too few data point for min', len(_))
                _=min_points.iloc[-1:]
                return False
            nidx=_.index[0]
            next_min_pt=_.iloc[0]
    #        high_low_diff=(next_max_pt.High-min_pt.Low)/min_pt.Low
            high_low_diff=(max_pt[col1]-next_min_pt[col2])/next_min_pt[col2]
            max_diff=max(high_low_diff_list)
            if high_low_diff > max_diff*1.1:
    #            print('vol increased over return False %s v %s' % (high_low_diff, max_diff))
                return False
            high_low_diff_list.append(high_low_diff)
    #        print(idx, nidx, high_low_diff)
    #    print('about to return True')
        lastpt=max_points.iloc[-1].Close
        prev_max=max(list(max_points.iloc[:-1][col1]))
        if max_points.iloc[-1].Close>prev_max:
            return True
        else:
    #        print('low vola but didnt breakout at last, return False', lastpt, prev_max)
            return False
    @staticmethod
    def find_vcp_up_patterns(max_min, ticker='_'):  
        patterns = defaultdict(list)
        min_points=max_min.query('minmax_type=="min"')
        max_points=max_min.query('minmax_type=="max"')    
        # Window range is 5 units
        key_points=max_points
        size=8
        for i in range(size, len(key_points)):  
            window = key_points.iloc[i-size:i]
            sd=window.index[0]
            ed=window.index[-1]
            # Pattern must play out in less than n units
            td= (window.index[-1] - window.index[0]) 
            sub_min_points=min_points.loc[sd:ed]
            sub_max_points=max_points.loc[sd:ed]        
    #        print('td span:', td, sub_min_points.shape, sub_max_points.shape)
            if td.days<50:
                continue

            if extremaPatternLooper.check_low_vol_breakup(sub_max_points, sub_min_points):
    #            print('found vcp_up for ',window.index[-1])
                patterns['vcp_up'].append((window.index[0], window.index[-1]))
                if len(patterns['vcp_up'])>5:
                    return patterns

        return patterns
    
    @staticmethod        
    def find_general_divergence_patterns(main_max_min, aux_max_min, main_col='Close', aux_col='macd', ex_cond='down,up', main_size=5):  
        patterns = defaultdict(list)

        # Window range is 5 units
        size=main_size
        for i in range(size, len(main_max_min)+1):
    #        if i<=len(down_max_min):
            if 1>0:
                window = main_max_min.iloc[i-size:i]
     #       else:
     #           window = down_max_min.iloc[-size:]

            # Pattern must play out in less than n units
            td= (window.index[-1] - window.index[0]) 
    #        print('td is ',td, type(1), i, len(down_max_min), window.index[0], window.index[-1] )
            if type(td)==type(pd.to_timedelta('1 days')):
    #            print('path 1 date index')
#                if td > pd.to_timedelta('150 days'):      
    #                print('exa span more than 150 days ', td)
#                    continue          

                if (window.index[-1] - window.index[0]) > pd.to_timedelta('100 days'):      
                    continue          

                main_win=window[main_col]
                aux_win=aux_max_min.loc[window.index[0]:window.index[-1]][aux_col]
    #            print('bull div up down win size:', len(main_win), len(aux_win), window.index[0], window.index[-1])

    #            print('found higherhl patten!', window.index[0])
                ret_win=extremaPatternLooper.check_divergence_by_up_down_windows(main_win, aux_win, window, ex_cond=ex_cond)
                if not ret_win is None:
                    patterns['divergence up %s v down %s' % (main_col, aux_col)].append(ret_win)

        return patterns 
    
    @staticmethod
    def check_divergence_by_up_down_windows(up_win, down_win, window, ex_cond='up,down'):
        if len(up_win)<3 or len(down_win)<3:
    #        print('down_win leng th< 2 up/down', len(up_win), len(down_win))
            return None
    #            print('high win, low win ', high_win, low_win)
        prev=up_win.iloc[0]
        up_ex=ex_cond.split(',')[0]
        down_ex=ex_cond.split(',')[1]    
        td=(window.index[-1]-window.index[0])
#        print('up ex %s, down ex %s' % (up_ex, down_ex))
        for high in up_win[1:]: 
    #        print('date:', up_win.index[0])
            diff=high-prev
            if up_ex=='up':
                if high<=prev:
    #                print('ex1 up_win < prev for up_ex:%s high:%s, prev:%s diff:%s' % (up_ex, high, prev, diff))
                    return None
            else:
                if high>=prev:
        #                    print('break higher high, skip', window.index[0])
    #                print('ex2 up_win > prev for up_ex::%s high:%s, prev:%s  diff:%s' % (up_ex, high, prev, diff))
                    return None

            prev=high
    #    print('down win:', down_win)
        prev=down_win.iloc[0]
        for high in down_win.iloc[1:]:
            diff=high-prev
            if down_ex=='down':
                if high>=prev:
    #                print('double up:', high, prev, up_win)
    #                print('ex3 down_win > prev for down_ex:%s high:%s, prev:%s diff:%s' % (down_ex, high, prev, diff))              
        #                    print('break higher low, skip', window.index[0])
                    return None
            else:
                if high<=prev:
    #                print('double down:', high, prev, down_ex)
    #                print('ex4 down_win <  prev for down_ex:%s high:%s, prev:%s diff:%s' % (down_ex, high, prev, diff))                               
        #                    print('break higher low, skip', window.index[0])
                    return None

    #                print('low is ',low, prev)                                    
            prev=high
    #    print('returning ok window:', td, window.index[0], window.index[-1] )
        return (window.index[0], window.index[-1])
    
    @staticmethod
    def find_bearish_divergence_patterns(up_max_min, down_max_min, trend_up_col='Close', trend_down_col='macd', ex_cond='up,down'):  
        patterns = defaultdict(list)

        # Window range is 5 units
        size=3
        for i in range(size, len(up_max_min)):
    #        print('i is ',i)
            window = up_max_min.iloc[i-size:i]

            # Pattern must play out in less than n units
            td= (window.index[-1] - window.index[0]) 
    #        print('td is ',td, type(1))
            if type(td)==type(pd.to_timedelta('1 days')):
    #            print('path 1 date index')
                if td > pd.to_timedelta('150 days'):      
    #                print('exa span more than 150 days ', td)
                    continue          

                up_win=window[trend_up_col]
                down_win=down_max_min.loc[window.index[0]:window.index[-1]][trend_down_col]
    #            print('bear div up down win size:', len(up_win), len(down_win), window.index[0], window.index[-1])

    #            print('found higherhl patten!', window.index[0])
                ret_win=check_divergence_by_up_down_windows(up_win, down_win, window, ex_cond=ex_cond)
                if not ret_win is None:
                    patterns['divergence up %s v down %s' % (trend_up_col, trend_down_col)].append(ret_win)

        return patterns
    
    @staticmethod
    def find_bullish_divergence_patterns(down_max_min, up_max_min, trend_down_col='Close', trend_up_col='macd', ex_cond='down,up'):  
        patterns = defaultdict(list)

        # Window range is 5 units
        size=5
        for i in range(size, len(down_max_min)+1):
    #        if i<=len(down_max_min):
            if 1>0:
                window = down_max_min.iloc[i-size:i]
     #       else:
     #           window = down_max_min.iloc[-size:]

            # Pattern must play out in less than n units
            td= (window.index[-1] - window.index[0]) 
    #        print('td is ',td, type(1), i, len(down_max_min), window.index[0], window.index[-1] )
            if type(td)==type(pd.to_timedelta('1 days')):
    #            print('path 1 date index')
                if td > pd.to_timedelta('150 days'):      
    #                print('exa span more than 150 days ', td)
                    continue          

                if (window.index[-1] - window.index[0]) > pd.to_timedelta('100 days'):      
                    continue          

                down_win=window[trend_down_col]
                up_win=up_max_min.loc[window.index[0]:window.index[-1]][trend_up_col]
                
    #            print('bull div up down win size:', len(up_win), len(down_win), window.index[0], window.index[-1])

    #            print('found higherhl patten!', window.index[0])
                ret_win=check_divergence_by_up_down_windows(down_win, up_win, window, ex_cond=ex_cond)
                if not ret_win is None:
                    patterns['divergence up %s v down %s' % (trend_up_col, trend_down_col)].append(ret_win)

        return patterns
    
