import threading
import time
import datetime

class MonitoredThreadRunner(threading.Thread):
    """
    A class that extends the Thread class to provide monitoring functionality.

    Attributes:
        target_func (function): The function to be called by the thread.
        data_dict (dict): The data dictionary to be passed to the function.
        counter (int): A counter to keep track of the number of times the thread has run.
        running (bool): A flag indicating whether the thread is currently running.

    Methods:
        run(): The main method that runs the thread and calls the target function.
        stop(): A method to stop the thread from running.
    """

    def __init__(self, target_func, data_dict):
        """
        Initializes a new instance of the MonitoredThreadRunner class.

        Args:
            target_func (function): The function to be called by the thread.
            data_dict (dict): The data dictionary to be passed to the function.
        """
        super().__init__()
        self.counter = 0
        self.running = True
        self.target_func = target_func
        self.data_dict = data_dict        

    def run(self):
        """
        The main method that runs the thread and calls the target function.
        """
        while self.running:
            print(f"Thread {self.name} is running... Counter: {self.counter}")
            self.counter += 1
            time.sleep(20)
            self.target_func(self.data_dict)
            if self.counter % 100 == 0:
                print(f"Restarting thread {self.name}...")
                self.stop()
                self.run()

    def stop(self):
        """
        A method to stop the thread from running.
        """
        self.running = False  # Set the running flag to False to stop the thread.

        
def check_thread(thread_dict):
    """
    Checks the status of a given thread and restarts it if it has terminated.

    Args:
        thread_dict (dict): A dictionary containing information about the thread, including its target function,
                            data dictionary, thread object, and name.

    Returns:
        None
    """
    target_func = thread_dict['target_func']
    data_dict = thread_dict['data_dict']
    thread = thread_dict['thread']
    thread_name = thread_dict['name']
    starttime= thread_dict['starttime']
    restart_time=9999999
    if 'restart_time' in thread_dict:
        restart_time=thread_dict['restart_time']
    try:
        if not thread.is_alive():
            print(f"Thread {thread_name} is terminated, restarting...")
            data_dict['restart']=False
            new_thread = MonitoredThreadRunner(target_func, data_dict)
            thread_dict['thread'] = new_thread
            new_thread.start()
        else:
            curtime=datetime.datetime.now()
            lapsed_time=(curtime-starttime).seconds
            if lapsed_time>restart_time:
                print(f'{lapsed_time} lapsed time bigger than restart time {restart_time}')  
                #thread.terminate() 
                data_dict['restart']=True
                thread_dict['starttime']=curtime
            print(f"Thread {thread_name} is running: {data_dict}")
            print(f'{lapsed_time} lapsed time vs restart time {restart_time}')  
    except Exception as e:
        print(f"Error checking thread {thread_name}: {e}")


def loop_all_threads(thread_dict_list):
    """
    Loops through a list of thread dictionaries and checks their status.

    Args:
        thread_dict_list (list): A list of dictionaries, where each dictionary contains information about a thread,
                                 including its target function, data dictionary, thread object, and name.

    Returns:
        None
    """
    while True:
        time.sleep(60)
        for thread_dict in thread_dict_list:
            curtime=datetime.datetime.now()
            check_thread(thread_dict)
            thread_dict['lastcheck_ts']=curtime
           


def make_thread(name, func, data_dict):
    t3=threading.Thread(target=func, args=([data_dict]))
    t3.start()
    ts=datetime.datetime.now()
    thread_dict = {'name': 'target_func1', 'target_func': func, 'data_dict': data_dict, 'thread': t3, 'restart_time':3600, 'starttime':ts}
    return thread_dict

def test_run():
    def func(data_dict):
        print('func1:',data_dict)
        data_dict['ab']=data_dict['ab']+1

    def func2(data_dict):
        print('func2:',data_dict)
        data_dict['bc']=data_dict['bc']+1

    data_dict={'ab':1, 'bc':2}
    # Create a new thread

    thread2 = MonitoredThreadRunner(func2, data_dict)
    thread2.start()
    print('yy')

    thread = MonitoredThreadRunner(func, data_dict)
    thread.start()
    print('xx')

    thread_dict_list=[]
    ts=datetime.datetime.now()
    thread_dict_list.append({'name':'t1', 'target_func':func, 'data_dict':data_dict, 'thread':thread, 'restart_time':30, 'starttime':ts, 'restart':False})
    thread_dict_list.append({'name':'t2', 'target_func':func2, 'data_dict':data_dict, 'thread':thread2,'starttime':ts, 'restart':False})
    
    loop_all_threads(thread_dict_list)            

if __name__=='__main__':
    test_run()    
