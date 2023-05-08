from datalib.commonUtil import commonUtil as cu
import json
import pika

global g_counter
g_counter=0


def recreate_queue(channel, queue_name, force=True):
    # Check if the queue exists
    queue = channel.queue_declare(queue=queue_name, passive=True)

    if force  or (queue.method and not queue.method.durable):
        # Delete the non-durable queue
        channel.queue_delete(queue=queue_name)
        print(f"Queue '{queue_name}' is deleted.")

        # Recreate the queue as durable
        channel.queue_declare(queue=queue_name, durable=True)
        print(f"Queue '{queue_name}' is recreated as durable.")
    elif not queue.method:
        # Create the queue as durable if it does not exist
        channel.queue_declare(queue=queue_name, durable=True)
        print(f"Queue '{queue_name}' is created as durable.")
    else:
        # The queue already exists and is durable
        print(f"Queue '{queue_name}' is already durable.")
        
    

class simpleMQUser:

    @staticmethod
    def create_channel(channel, qname):
        try:
#            recreate_queue(channel, qname)
            channel.queue_declare(queue=qname, durable=True )
            print(f"Exchange {qname} already exists.")
        except pika.exceptions.ChannelClosedByBroker:
            # If exchange doesn't exist, create it
            channel.queue_declare(queue=qname, durable=True )
            print(f"Exchange {myrmq} created.")
        return channel

    @staticmethod
    def connect_channel(qname):
        import pika
        username=cu.getProp('rq_username')
#        username='tgbot'
#        passwd='tgbot'
        passwd=cu.getProp('rq_password')    
        rbmq_ip=cu.getProp('rbmq_ip')        
#        rbmq_ip='esjo'
        rbmq_ip='127.0.0.1'
        print('rbmq_ip:',rbmq_ip, username, passwd)

        credentials = pika.PlainCredentials(username, passwd)

        connection = pika.BlockingConnection(pika.ConnectionParameters(rbmq_ip, credentials=credentials))
        channel = connection.channel()
        simpleMQUser.create_channel(channel, qname)
        try:
            channel.queue_declare(queue=qname, durable=True)
        except Exception as e:
            print(f'declaring qname {qname} ex',e )
        return channel

    @staticmethod
    def connect_and_post_txtmsg(msg, qname, close_channl=False):
        channel=simpleMQUser.connect_channel(qname)
        channel.basic_publish(exchange='', routing_key=qname, body=msg)
        if close_channl:
            channel.close()
        return channel
            
    @staticmethod
    def connect_and_post_obj(b64_objmsg, qname, close_channl=False):
        import pika
        username=cu.getProp('rq_username')
#        username='tgbot'
#        passwd='tgbot'
        passwd=cu.getProp('rq_password')    
        rbmq_ip=cu.getProp('rbmq_ip')        
#        rbmq_ip='127.0.0.1'
        print('connect info:',username, rbmq_ip)
        credentials = pika.PlainCredentials(username, passwd)
        connection = pika.BlockingConnection(pika.ConnectionParameters(rbmq_ip, credentials=credentials))
        channel = connection.channel()
        channel.basic_publish(exchange='', routing_key=qname, body=b64_objmsg)
        if close_channl:
            channel.close()
        return channel

    @staticmethod
    def connect_and_post_json(json_msg, qname, close_channl=False):
        import pika
        username=cu.getProp('rq_username')
#        username='tgbot'
#        passwd='tgbot'
        passwd=cu.getProp('rq_password')    
        rbmq_ip=cu.getProp('rbmq_ip')        
#        rbmq_ip='127.0.0.1'
        print('connect info:',username, rbmq_ip)
        credentials = pika.PlainCredentials(username, passwd)
        connection = pika.BlockingConnection(pika.ConnectionParameters(rbmq_ip, credentials=credentials))
        channel = connection.channel()
        simpleMQUser.create_channel(channel, qname)
        if type(json_msg)==type({}):
            print('converting dict to json', type(json_msg))
            for k in [k for k in json_msg.keys() if k[0]=='_']:
                json_msg[k]=None
            json_msg=json.dumps(json_msg, indent=4)
        ret=channel.basic_publish(exchange='', routing_key=qname, body=json_msg)
        print('post results:',ret, qname)
        if close_channl:
            channel.close()
        return channel

            
            
    @staticmethod
    def try_read_txt_msg(channel, qname, func, auto_ack=True):
        for method_frame, properties, body in channel.consume(qname, auto_ack=auto_ack):
            print(f'len:%s, head:%s' % (len(body), body[:20]))
            outdict=func(body)

    @staticmethod
    def try_read_obj_msg(channel, qname, func, auto_ack=True):
        import base64
        for method_frame, properties, body in channel.consume(qname, auto_ack=auto_ack):
            print(f'len:%s, head:%s' % (len(body), body[:20]))
            encoded_string=body
            bytedata=base64.b64decode(encoded_string)
            tmpfname='tmp.pkl.gz'
            with gzip.open(tmpfname, 'wb') as f:
                f.write(bytedata)
            obj_msg=cu.load_output_dict_pickle_with_img(tmpfname)
            outdict=func(obj_msg)
            
    @staticmethod
    def try_read_json_msg(channel, qname, func, auto_ack=True):
        import base64,gzip
        ret_list=[]
        for method_frame, properties, body in channel.consume(qname, auto_ack=auto_ack):
            print(f'len:%s, head:%s' % (len(body), body[:20]))
            payload = json.loads(body)
            media_msg_key='encoded_content'
            decoded_content=None
#            if media_msg_key in payload.keys():
#                decoded_content = base64.b64decode(payload[media_msg_key])
#                payload['decoded_content']=decoded_content
            payload['_channel']=channel
            payload['_method_frame']=method_frame
            payload['_header_frame']=properties
            outdict=func(payload)
    
    @staticmethod    
    def reply_func(obj_msg):
        print(obj_msg.keys())
        print('got key already!')

    @staticmethod    
    def send_nfty_msg(msg):
        import requests
        from datetime import datetime
        msg='%s %s' % (msg, datetime.now().strftime('%Y%m%d %H%M%S'))
        api_url='https://ntfy.sh/jl892'
        payload = msg.encode('utf-8')
        result = requests.post(api_url, data=payload)
        print(result)

def testrun_obj_msg(pklfname='ref_userid.pkl.gz', qname='test', img_list=[]):
    import gzip
    import base64
    obj_dict={}
    obj_dict['img_list']=img_list
    if not img_list is None:
        cu.pickle_output_dict_with_img(obj_dict, pklfname, get_b64=True)
    else:
        obj_dict=cu.load_output_dict_pickle_with_img(pklfname)

    with gzip.open(pklfname) as f:
        b64_objmsg = base64.b64encode(f.read())
    channel=simpleMQUser.connect_and_post_obj(b64_objmsg, qname)

def testget_obj_msg(qname, func=None):
    g_counter=0
    def def_func(obj_msg):
        global g_counter
        print('test get obj msg')
        g_counter=g_counter+1
        print('abc', g_counter)
        if type(obj_msg)==type({}):
            print(obj_msg.keys()) 
            print(obj_msg)
        else:
            return None

    if func is None:
        func=def_func
    print('func is ',func)
    channel=simpleMQUser.connect_channel(qname)
    simpleMQUser.try_read_obj_msg(channel, qname, func)

def testrun_txt_msg(txt='se XLK'):
    qname='sectorRotationBot'
    channel=simpleMQUser.connect_and_post_txtmsg(txt, qname)
    
def test_main():
    import time
    qname="test"
    testrun_obj_msg(pklfname='abc.pkl.gz', qname=qname, img_list=['i1.jpg', 'i2.jpg'])
    time.sleep(10)
    testget_obj_msg(qname)

if __name__=='__main__':
    test_main()

