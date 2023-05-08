import pika
import re
import time
import os,pickle,gzip
import time
import json
import base64

from baselib.logUtil import dlog, log    
from baselib.tgbotHelper import tgbotHelper
from bot.simpleMQUser import simpleMQUser as smu

    
    
def extract_stock_tickers(message):
    message=f'{message} '    
    # Regular expression to match stock tickers for US, HK, AU, UK, Canada, and Paris stock markets
    ticker_pattern = r'((?:[A-Z]{1,4}\.)?[A-Z]{1,5}(?:\-[A-Z]{1,2})?(?:\.[a-z]{1,2})?)[,\ ]'

    # Find all tickers in the message using the regular expression
    tickers = re.findall(ticker_pattern, message)

    ticker_pattern = r'((?:[A-Z]{1,4}\.)?[0-9]{1,5}(?:\-[A-Z]{1,2})?(?:\.[A-Za-z]{1,2})?)[,\ ]'# for hk

    # Find all tickers in the message using the regular expression
    tickers2 = re.findall(ticker_pattern, message)
    if '.hk' in ('%s' % tickers2).lower():
        tickers.remove('HK')
        
    ret_list=[c for c in list(set(tickers+tickers2)) if not c.isnumeric()]
    return ret_list

def get_markup_dict_by_ticker_list(ticker_list):
    ret_dict={}
    for t in ticker_list:
        ret_dict[f'vcp_plot {t}']=f'vcp_plot {t}'
        ret_dict[f'pattern_search {t}']=f'pattern_search {t}'        
        ret_dict[f'sector_check {t}']=f'sector_check {t}'                
    return ret_dict

def clean_dir(dirname):
    import os
    import glob

    files = glob.glob('dirname/*')
    for f in files:
        os.remove(f)
        
class BrokerUserSession:
    def __init__(self, user_id):
        print('init broker user session')
        self.user_id = user_id
        self.messages = []
        self.tmpdir='tmp'
        clean_dir(self.tmpdir)        
        
    def add_message(self, message):
#        print('b4 append message, ', len(self.messages))        
        self.messages.append(message)
#        print('appended message, ', len(self.messages))

    def handle_messages(self):
        print('b4 handle message, ', len(self.messages))                
        for message in self.messages:
            print(f"XXXXXXX User {self.user_id} received message: {message}, {len(self.messages)}")
        print('af handle  message, ', len(self.messages))        
        #do something here with db
        latest_msg=message
        body=latest_msg
        body['reply_status']='replied'
        message_txt=body['message_txt']
        reply_rqname='tg_reply'        

        tickers=extract_stock_tickers(message_txt)
        
        if len(tickers)>0 and not 'reply_markup' in body:
            print('got tickers##########')
            reply_markup=get_markup_dict_by_ticker_list(tickers)
            body['reply_markup']=reply_markup
            smu.connect_and_post_json(body, reply_rqname)         
            
        #cmd message are triggered when user press button in the chat msg    
        if 'cmd' in body:
            cmd=body['cmd']
            print('have cmddddddddddddd')
            body['message_txt']=cmd.split(' ')[0]
            if 'pattern' in cmd:
                print('pattern button')
                target_qname='tg_pinecone'
                smu.connect_and_post_json(body, target_qname)
            if 'vcp' in cmd:
                print('vcp button')                
                target_qname='tg_vcp'
                smu.connect_and_post_json(body, target_qname)
            
            if 'sector' in cmd:
                print('sector button')                                
                target_qname='tg_sectorRotate'
                smu.connect_and_post_json(body, target_qname)
            
        else:
            print('GPTGPTGPTGPTGPTGPTGPTGPTGPT no cmd match, use gpt fallback')
            target_qname='tg_gpt'
            if 'encoded_content' in body:        
                if not os.path.exists('photos'):
                    os.makedirs('photos')
                fname=f'photos/{self.user_id}.jpg'
                img_list=[fname]
                with open(fname, 'wb') as f:
                    f.write(base64.b64decode(body['encoded_content']))
                b64_msg=tgbotHelper.get_message_dict(body['message_txt'], img_list)
                body['b64_msg']=b64_msg
            smu.connect_and_post_json(body, target_qname)                


        #self.messages = []

