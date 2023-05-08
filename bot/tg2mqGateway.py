import socket
import telebot
import json
import sys
import os,gzip
import base64
import pprint
if not os.path.dirname(__file__)==os. getcwd():
    from os.path import dirname
    sys.path.append(dirname(dirname(__file__)))    
    pprint.pprint(sys.path)
from datalib.cfgUtil import cfgUtil
from bot.simpleMQUser import simpleMQUser as smu
import threading
import datetime
import os,base64,gzip,pickle
from baselib.tgbotHelper import tgbotHelper
from baselib.threadUtil import MonitoredThreadRunner, loop_all_threads, make_thread

if 'tgbot' in dir():
    del tgbot

token=cfgUtil.get_token_from_ini()
#token='1811289968:AAFRSwFTTIkfy4P15b1TUB-d2eWn7G08TVw' #jlCronBot
tgbot = telebot.TeleBot(token)
data_dict={}
sink_qname='tg_send'

@tgbot.message_handler(content_types=['photo'])
def photo(message, img_handler_func=None, txt_handler_func=None):
    import os
    userid=message.from_user.id
    username=message.from_user.first_name
    tgmsg=message
    
    print(data_dict, 'incoming user id is ',userid, username,  img_handler_func, txt_handler_func)    
    print('message.photo =', message.photo)
    fileID = message.photo[-1].file_id
    print('fileID =', fileID)
    file_info = tgbot.get_file(fileID)
    print('file.file_path =', file_info.file_path)
    downloaded_file = tgbot.download_file(file_info.file_path)
    imgfname=file_info.file_path
    dirname=file_info.file_path
    if '.jpg' in imgfname:
        dirname=os.path.basename(imgfname)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    with open(imgfname, 'wb') as new_file:
        new_file.write(downloaded_file)
    if not message.text is None:
        caption_txt=message.text.lower()
        
    # Get the media content as bytes
    file_id = message.photo[-1].file_id if message.content_type == 'photo' else message.video.file_id
    file_info = tgbot.get_file(file_id)
    file_content = tgbot.download_file(file_info.file_path)

    encoded_content = base64.b64encode(file_content).decode()

    # Construct the message payload
    payload = {'chat_id': message.chat.id,
                'user_id': message.from_user.id,
                'user_firstname': message.from_user.first_name,
                'user_lastname': message.from_user.last_name,               
               'message_id': message.message_id,
               'message_txt': message.text, 
               'content_type': message.content_type,
               'encoded_content': encoded_content}
    body=json.dumps(payload)
    smu.connect_and_post_json(body, qname=sink_qname)
    #print(body)

@tgbot.message_handler(commands=['start'])
def start_handler(message):
    chat_id = message.chat.id
    if chat_id not in known_chat_ids:
        # This is the first time the user is chatting with the bot
        tgbot.reply_to(message, "Welcome! This is your first time chatting with me. type /help for instruction")
        known_chat_ids.append(chat_id)
    else:
        # The user has already chatted with the bot before
        tgbot.reply_to(message, "Welcome back!")
        
# Define a handler for the /help command
@tgbot.message_handler(commands=['help'])
def help_handler(message):
    # Send a message with instructions on how to use your bot
    help_message = "Welcome to my bot! Here's how to use me:\n\n1. type show AAPL\n to get suggestion about apple (AAPL) stock\n2. send me image and I will try to parse the text and come up with more information\n3. ask me any question and I will try my best to answer"
    # Pin the help message
    chat_id = message.chat.id
    help_message_id = tgbot.send_message(chat_id, help_message).message_id
    tgbot.pin_chat_message(chat_id, help_message_id)        
        
        
#@tgbot.message_handler(func=lambda message: True)
@tgbot.message_handler(content_types=['text'], func=lambda message: True)
def message_handler2( message, txt_handler_func=None):
    print('xxxx',data_dict)
    from oailib.oaiWrapperLib import enhance_note_from_image, get_discussion_question
    from bot.simpleMQUser import  simpleMQUser as smu
    userid=message.from_user.id
    username=message.from_user.first_name
    print('incoming user id is ',userid, username, txt_handler_func)
    print('from_user:',message.from_user)
    # Construct the message payload
    
    payload = { 'chat_id': message.chat.id,
                'user_id': message.from_user.id,
                'user_firstname': message.from_user.first_name,
                'user_lastname': message.from_user.last_name,
               'message_id': message.message_id,
               'message_txt': message.text, 
               'content_type': message.content_type}    
    body=json.dumps(payload)
    print(body)
    smu.connect_and_post_json(body, qname=sink_qname)
    
@tgbot.callback_query_handler(func=lambda call: True)
def handle_callback_query(call):
    chat_id=call.message.chat.id
    user_id=call.message.from_user.id    
    message={}
    message['chat_id']=chat_id
    message['user_id']=user_id
    message['message_txt']=call.data    
    message['cmd']='%s' % call.data    
    if 'reply_markup' in message:
        del message['reply_markup']
    sink_qname='tg_send'
    smu.connect_and_post_json(message, qname=sink_qname)
    
    
def tg_poll(tgbot, _data_dict, qname='tg_cmd'):
    import time
    import pandas as pd
    global sink_qname
    print('now in tgbot poll')
    counter=0
    run=True
    data_dict=_data_dict
    sink_qname=qname
    while run:
        ts1=pd.to_datetime('today')
#        tgbot.infinity_polling(interval=5, timeout=10, non_stop=True, long_polling_timeout = 5)
        tgbot.polling(interval=5, non_stop=True)
        ts2=pd.to_datetime('today')
        if ++counter%100==0:
            print(f'my poll:{ts1} {counter}')
            print(data_dict)
            print(f'my poll ed {ts2}')        
        time.sleep(2)        
        if counter>3000 or data_dict['restart']:
            print('exist upon signal or counter', counter)
            run=False


def get_markup_from_dict(markup_dict):
    from telebot import types
    button_list=[]
    markup = types.InlineKeyboardMarkup()    
    for key in markup_dict:
        plot_button = types.InlineKeyboardButton(f"{key}", callback_data=markup_dict[key])
        button_list.append(plot_button)    
        markup.add(plot_button)
    return markup

def thread_target_func1(data_dict):
    print(data_dict)
    tgbot=data_dict['tgbot']
    tg_poll(tgbot, data_dict, qname='tg_send')    
            
        
        
def on_reply(message):
    print('got tg reply msg from other agents',message.keys())    
    chat_id=message['chat_id']
    user_id=message['user_id']
    username='there'    
    if 'user_firstname' in message:
        username=message['user_firstname']    
#    text=f'Hi {username}!'              
    text=''
    if 'message_txt' in message and not message['message_txt'] is None:
        if len(message['message_txt'])>0:
            text=message['message_txt']    
    else:
        if 'reply_markup' in message:
            del message['reply_markup']        

    print(f'sending reply {chat_id}, {user_id}, {text}')
    method_frame=message['_method_frame']
    channel=message['_channel']
    channel.basic_ack(delivery_tag = method_frame.delivery_tag)

    if 'b64_msg' in message:
        print('hhhhhhhhhhhhhhh has b64_msg')
        msg_dict=tgbotHelper.b64_load_output_dict_pickle_with_img(message['b64_msg'])
        if 'img_list' in msg_dict:
            i=0
            for fname in msg_dict['img_list']:
                print('fname is ',fname)
                dirname=os.path.dirname(fname)
                print('dirname is ',dirname)
                if len(dirname)==0:
                    dirname='tmp'
                    fname=f'{dirname}/{fname}'
                elif not os.path.exists(dirname):
                    os.makedirs(dirname)

                if 'img_obj' in msg_dict:
                    print('fname is ', fname, i, len(msg_dict['img_obj']))
                    if len(msg_dict['img_obj'])>i:
                        with open(fname, 'wb') as f:
                            f.write(msg_dict['img_obj'][i])
                        tgbot.send_photo(chat_id, open(fname, 'rb'))
                    i=i+1

        if 'message_txt' in msg_dict and not msg_dict['message_txt'] is None:
            text=msg_dict['message_txt']    
        smu.send_nfty_msg(text)
        tgbot.send_message(chat_id=chat_id, text=text)
    else:
        if not text=='':
            smu.send_nfty_msg(text)
            tgbot.send_message(chat_id=chat_id, text=text)
    if 'reply_markup' in message and not message["message_txt"]=='':        
        markup=get_markup_from_dict(message['reply_markup'])
        print('markup to dict()',markup.to_dict())
        print('xxxxxxxx %s ' % message["message_txt"])        
        klist=sum([[x['text'] for x in c] for c in markup.to_dict()['inline_keyboard']], [])
        print('klist:%s' % klist)
        for k in klist:
            if k in message['message_txt']:
                print('skip reprinting markup')
                return False
        tgbot.send_message(chat_id=chat_id, text='Click one of the follow buttons for suggestions', reply_markup=markup)
    return True

def thread_reply_func1(data_dict):
    print(data_dict)
    tgbot=data_dict['tgbot']
    func=on_reply
    qname='tg_reply'
    channel=smu.connect_channel(qname)
    running=True
    run_cnt=100
    counter=0
    user_sessions = {}

    while running:
        smu.try_read_json_msg(channel, qname, func, auto_ack=False)
        if counter>run_cnt:
            running=False
        time.sleep(10)

        
def run_tg2mq_gateway():
    method_frame=None
    channel=None

    data_dict={'tgbot':tgbot, 'counter':0, 'session_dict':{}} 
    name='tg_poll'
    func=thread_target_func1
    thread_dict=make_thread(name, func, data_dict)
    name='reply_poll'
    func2=thread_reply_func1
    reply_thread_dict=make_thread(name, func2, data_dict)
    thread_dict_list=[thread_dict,reply_thread_dict]    
    loop_all_threads(thread_dict_list)
    
if __name__=='__main__':
    run_tg2mq_gateway()        
