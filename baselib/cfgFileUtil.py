import configparser
import os
import re
import logging

FORMAT = '%(asctime)s %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger('config')
logger.setLevel(logging.DEBUG)
logger.info('getting config')

def get_token_input(section='pinecone', value='api_token', prompt=None, pattern=None, cfg_fname=r"configfile.ini"):
    """
    This function prompts the user to input a token, validates it using a regular expression pattern if provided,
    and saves the token in a configuration file. The configuration file is created if it does not exist.

    Parameters:
    - section (str): The section in the configuration file to which the token will be saved. Default is 'pinecone'.
    - value (str): The key in the configuration file to which the token will be saved. Default is 'api_token'.
    - prompt (str): The prompt to display to the user when asking for the token. Default is None.
    - pattern (str): A regular expression pattern to validate the token. Default is None.
    - cfg_fname (str): The filename and path of the configuration file. Default is 'configfile.ini'.

    Returns:
    - token (str): The token entered by the user, or None if the token is not valid.

    Example usage:
    - token = get_token_input(pattern=r'^sk-[a-zA-Z0-9]{32}$')
    """
    token=input(prompt)
    
    if not pattern is None and not re.match(pattern, token):
        logger.info('error in token')
        return None

    config = configparser.ConfigParser()
    if os.path.exists(cfg_fname):
        config.read(cfg_fname)

    if not config.has_section(section):
        # Add the structure to the file we will create
        config.add_section(section)

    config.set(section, value, token)

    with open(cfg_fname, 'w') as configfile:
        print(f'writing to file {cfg_fname}', config)
        config.write(configfile)
    return token

def get_prop(section='pinecone', values=['api_token'], cfg_fname="configfile.ini"):
    config = configparser.ConfigParser()
    config.read(cfg_fname)

    try:
        print(values)
        if len(values)==1:
            ret_v= config[section][values[0]]
        print('a')
        ret_v=[]
        for v in values:
            print('v:',v)
            _=config[section][v]
            ret_v.append(_)
    except Exception as e:
        print(f"can't get {section} {values} from {cfg_fname}")
        return None
    return ret_v


class cfgFileUtil:
    @staticmethod
    def get_pinecone_token_input(cfg_fname=r"configfile.ini"):    
        pattern = r'^[a-f\d]{8}-[a-f\d]{4}-[a-f\d]{4}-[a-f\d]{4}-[a-f\d]{12}$'
        prompt='please input pinecone api token from pinecone.io'
        value='api_token'
        section='pinecone'
        api_token= get_token_input(section=section, value=value, prompt=prompt, pattern=pattern, cfg_fname=cfg_fname)
        pattern=None
        prompt='please input pinecone env from pinecone.io (e.g. us-east-1-aws)'
        value='env'
        env=get_token_input(section=section, value=value, prompt=prompt, pattern=pattern, cfg_fname=cfg_fname)
        return api_token,env

    @staticmethod
    def get_pinecone_prop(cfg_fname="configfile.ini"):
        return get_prop(section='pinecone', values=['api_token', 'env'], cfg_fname=cfg_fname)

    @staticmethod
    def get_openai_token_input(cfg_fname=r"configfile.ini"):    
        pattern = r'^[a-z]{2}-[a-zA-Z0-9]{48}$'
        prompt='please input openai api token'
        value='api_token'
        section='openai'

        retv= get_token_input(section=section, value=value, prompt=prompt, pattern=pattern, cfg_fname=cfg_fname)
        return retv

    @staticmethod
    def get_openai_prop(value='api_token', cfg_fname="configfile.ini"):
        section='openai'
        return get_prop(section=section, values=[value], cfg_fname=cfg_fname)


    @staticmethod
    def get_openai_token_input(cfg_fname=r"configfile.ini"):    
        pattern = r'^[a-z]{2}-[a-zA-Z0-9]{48}$'
        prompt='please input openai api token'
        value='api_token'
        section='openai'

        retv= get_token_input(section=section, value=value, prompt=prompt, pattern=pattern, cfg_fname=cfg_fname)
        return retv

    @staticmethod
    def get_openai_prop(value='api_token', cfg_fname="configfile.ini"):
        section='openai'
        return get_prop(section=section, values=[value], cfg_fname=cfg_fname)

    @staticmethod
    def get_telegram_token_input(cfg_fname=r"configfile.ini"):    
        pattern = r'^[0-9]{10}:[a-zA-Z0-9_-]{35}$'
        prompt='please input your telegram bot token from botFather'
        value='tg_token'
        section='telegram'
        retv= get_token_input(section=section, value=value, prompt=prompt, pattern=pattern, cfg_fname=cfg_fname)
        return retv
    
    @staticmethod
    def get_telegram_prop(value='tg_token', cfg_fname="configfile.ini"):
        section='openai'
        return get_prop(section=section, values=[value], cfg_fname=cfg_fname)



def test_run():

    ret=cfgFileUtil.get_openai_prop('api_token')
    if ret is None:
        cfgFileUtil.get_openai_token_input(cfg_fname=r"configfile.ini")
    else:
        print('open ai info:',ret)

    ret=cfgFileUtil.get_pinecone_prop()        
    if ret is None:
        cfgFileUtil.get_pinecone_token_input(cfg_fname=r"configfile.ini")
    else:
        print('pinecone ai info:',ret)
        
    ret=cfgFileUtil.get_telegram_prop('tg_token')        
    if ret is None:
        cfgFileUtil.get_telegram_token_input(cfg_fname=r"configfile.ini")
    else:
        print('pinecone ai info:',ret)        

if __name__=='__main__':
    test_run()
