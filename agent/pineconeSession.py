import pika
import re
import os,sys
import pprint
import time
import json
import base64
if not os.path.dirname(__file__)==os. getcwd():
    from os.path import dirname
#    print('running subdir job from root', dirname(__file__))    
    sys.path.append(dirname(dirname(__file__)))    
    pprint.pprint(sys.path)
    
from baselib.logUtil import dlog, log    
from baselib.tgbotHelper import tgbotHelper
from bot.rqSessionService import rqSessionManager, start_service
from bot.simpleMQUser import simpleMQUser as smu
from bot.BrokerUserSession import BrokerUserSession
from opkatsPatternMatcherUtil import opkatsPatternMatcherUtil

class pineconeSession(BrokerUserSession):
    """
    A class representing a session for a Pinecone user.

    Parameters:
    -----------
    user_id: str
        The ID of the user for whom the session is being created.

    Attributes:
    -----------
    messages: list
        A list of messages received during the session.

    Methods:
    --------
    handle_messages()
        Handles the messages received during the session.
    """
    def __init__(self, user_id):

        BrokerUserSession.__init__(self, user_id)
        log('init pinecore user session')        

    def handle_messages(self):
        dlog('receive emssage %s message')
        message=self.messages[-1]
        ticker=message['cmd'].split(' ')[1]
        dlog(f'ticker is {ticker}')
#        from opkatsPatternMatcherUtil import opkatsPatternMatcherUtil
        fname=opkatsPatternMatcherUtil.match_period_with_vectordb(ticker)
        fname
        message['img_list']=[fname]
        message['message_txt']=f'vector search of {ticker}'
        sink_qname='tg_reply'
        b64_msg=tgbotHelper.get_message_dict(message['message_txt'], [fname])
        message['b64_msg']=b64_msg    
        if 'reply_markup' in message:
            del message['reply_markup']        
        smu.connect_and_post_json(message, qname=sink_qname)
        


def create_pinecone_session(user_id):
    return pineconeSession(user_id)


auto_ack=True
channel=None
g_session_manager=None
g_prev_dtag=1

def run_pinecone_agent():
    global g_prev_dtag
    global g_session_manager
    g_session_manager = rqSessionManager(create_pinecone_session, auto_ack=auto_ack)
    g_prev_dtag=1

    start_service(g_session_manager, src_qname='tg_pinecone', auto_ack=auto_ack)                                                         
    
if __name__=='__main__':
    from agent.pineconeSession import run_pinecone_agent
    run_pinecone_agent()
