import pika
import re
import time
import json
import base64
import pprint    
from baselib.tgbotHelper import tgbotHelper
from bot.simpleMQUser import simpleMQUser as smu
from bot.BrokerUserSession import BrokerUserSession

        
class rqSessionManager:
    
    def __init__(self, create_session_func=None, auto_ack=True):
        self.sessions={}
        self.auto_ack=auto_ack
        if not create_session_func is None:
            self.create_session_func=create_session_func
        else:
            self.create_session_func=(lambda user_id:BrokerUserSession(user_id))
        return None
    
    def get_sesssion(self, user_id):
        session = self.sessions.get(user_id)
        if session is None:
            session = self.create_session_func(user_id)
            self.sessions[user_id]=session
        return session
        
def on_message(body):
    global g_session_manager, g_prev_dtag
    print('user sessions:',g_session_manager.sessions.keys())    
    message=(body)
    dtag=g_prev_dtag
    print('msg:',message)
    if '_method_frame' in message:        
        method_frame=message['_method_frame']
        if not method_frame is None:
            dtag=method_frame.delivery_tag
            print('got tag ', dtag)     
            g_prev_dtag=dtag
        else:
            dtag=g_prev_dtag

    user_id=''
    
    if 'user_id' in message:
        user_id = message['user_id']
         
    if not user_id:
        print("Received message with no user ID, ignoring.")
        return
    
    '''
    session = g_user_sessions.get(user_id)
    
    if not session:
        print(f"Creating new session for user {user_id}.")
        session = BrokerUserSession(user_id)
        user_sessions[user_id] = session
    '''
    session=g_session_manager.get_sesssion(user_id)
        
    if session.user_id == user_id:
        session.add_message(message)
        session.handle_messages()
    else:
        print(f"Received message for user {user_id}, but agent is currently handling user {session.user_id}, ignoring.")


    if not g_session_manager.auto_ack:
        try:
            channel.basic_ack(delivery_tag=dtag)
        except Exception as e:        
            print('ack all q:', e)

            if '_channel' in message: 
                _channel=message['_channel']
                if not _channel is None:
                    _channel.basic_ack(delivery_tag=dtag)
        g_prev_dtag=dtag

g_prev_dtag=1    
g_session_manager=None
def start_service(session_manager, src_qname='tg_send', auto_ack=True):
    global channel
    global g_session_manager
    g_session_manager=session_manager
    running=True
    run_cnt=100
    counter=0
    func=on_message
    channel=smu.connect_channel(src_qname)    
    while running:

        try:
            smu.try_read_json_msg(channel, src_qname, func, auto_ack=auto_ack)
        except Exception as e:
            import traceback                                                                                                                                           
            print('exception:', e)
            traceback.print_exc()        
            print('exception in run loop, try recconnect ', e)
            channel=smu.connect_channel(src_qname)
        print('run cnt ',run_cnt, counter)
        counter=counter+1
        if counter>run_cnt:
            running=False
        time.sleep(10)
        
    
