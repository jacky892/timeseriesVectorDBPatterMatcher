#%%writefile chatbotPipelineUtil.py
#from sentence_transformers import SentenceTransformer, util
#from parrot import Parrot
#from transformers import pipeline
import re, yaml
import pandas as pd
#import torch
import warnings
warnings.filterwarnings("ignore")


def get_enhanced_ner_pipeline():
    from transformers import pipeline

    from transformers import AutoTokenizer, TFAutoModelForTokenClassification

    tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/camembert-ner-with-dates", 
                                              from_pt=True)

    model = TFAutoModelForTokenClassification.from_pretrained("Jean-Baptiste/camembert-ner-with-dates",
                                                              from_pt=True)

    print(model.config.id2label)

    enhanced_ner = pipeline('ner', 
                            model=model, 
                            tokenizer=tokenizer, 
                            aggregation_strategy="simple")
    #basic_ner = pipeline("ner")
    return enhanced_ner

#enhanced_ner=get_enhanced_ner_pipeline()
def parse_entities(msg, enhanced_ner=None):
    import re
    all_keywords=[]
    all_tokens=[]
    tokens = re.findall(r'\[(.*?)\]', msg)
    print(tokens)
    if not enhanced_ner is None:
        ner_list=enhanced_ner(msg.replace('[', '').replace(']', ''))
        print('ner is :',ner_list)
    for t in tokens:
        msg=msg.replace(f'[{t}]', '|')
    print(msg, tokens)
    all_tokens=all_tokens+tokens
    keywords=[t for t in msg.split(' ') if len(t)>5]
    all_keywords=all_keywords+keywords
    new_split=msg.split('|')
    print('split is ',new_split, tokens)
    print('keywords are ',keywords)
    ret_dict={}
    ret_dict['keywords']=list(set(all_keywords))
    ret_dict['tokens']=list(set(all_tokens))
    return ret_dict

def add_related_keywords(keywords):
    if 'impact' in keywords:
        return keywords+['sector', 'macro']
    return keywords

def get_tokens_keywords_from_utterence(msg_list, enhanced_ner):
    all_tokens=[]
    all_keywords=[]
    for msg in msg_list:
        # Use regular expression to find all substrings enclosed by square brackets
        ret_dict=parse_entities(msg, enhanced_ner)
        all_tokens= all_tokens+ret_dict['tokens']
        all_keywords=all_keywords+ret_dict['keywords']
    all_keywords=add_related_keywords(all_keywords)
    #!cat intent.yaml
    #msg_list
    return all_tokens , all_keywords




def augment_utterance(msg_list, parrot):
    ''' 
    uncomment to get reproducable paraphrase generations
    def random_state(seed):
      torch.manual_seed(seed)
      if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    random_state(1234)
    '''
    phrases = msg_list

    augmented_list=[]
    for phrase in phrases:
        phrase=phrase.replace('[', '').replace(']', '').replace("\n", "")
        print("-"*100)
        print("Input_phrase: ", phrase)
        print("-"*100)
        para_phrases = parrot.augment(input_phrase=phrase)
        if para_phrases is None:
            continue
        for para_phrase in para_phrases:
            print(para_phrase)
            augmented_list.append(para_phrase[0])
    return augmented_list
           


class QualityControlPipeline:
    
    def __init__(self, type):
        assert type in ['captions', 'questions', 'sentences']
        self.pipe = pipeline('text2text-generation', model=f'ibm/qcpg-{type}')
        self.ranges = {
            'captions': {'lex': [0, 90], 'syn': [0, 80], 'sem': [0, 95]},
            'sentences': {'lex': [0, 100], 'syn': [0, 80], 'sem': [0, 95]},
            'questions': {'lex': [0, 90], 'syn': [0, 75], 'sem': [0, 95]}
        }[type]

    def __call__(self, text, lexical, syntactic, semantic, **kwargs):
        assert all([0 <= val <= 1 for val in [lexical, syntactic, semantic]]), \
                 f' control values must be between 0 and 1, got {lexical}, {syntactic}, {semantic}'
        names = ['semantic_sim', 'lexical_div', 'syntactic_div']
        control = [int(5 * round(val * 100 / 5)) for val in [semantic, lexical, syntactic]]
        control ={name: max(min(val , self.ranges[name[:3]][1]), self.ranges[name[:3]][0]) for name, val in zip(names, control)}
        control = [f'COND_{name.upper()}_{control[name]}' for name in names]
        assert all(cond in self.pipe.tokenizer.additional_special_tokens for cond in control)
        text = ' '.join(control) + text if isinstance(text, str) else [' '.join(control) for t in text]
        return self.pipe(text, **kwargs)    


def qc_augment_utterances(msg_list, qcmodel):
    new_utterences=[]
    for question in msg_list:
        question=question.replace('[', '').replace(']', '').replace("\n", "")
        output=qcmodel(question, lexical=0.3, syntactic=0.5, semantic=0.8)
        print(question, output)
        new_utterences.append(output)
    return new_utterences

def compare_meaning(user_input, questions, model):
    embedding_1= model.encode(user_input, convert_to_tensor=True)
    embedding_2 = model.encode(questions, convert_to_tensor=True)
    question_scores = util.semantic_search(embedding_1, embedding_2)
 
    return question_scores
    
def load_faq(fname='faq1.txt'):
    with open(fname, 'r') as f:
        txt=f.read()
    txt[:10]
    qlist=[]
    alist=[]
    for line in txt.split("\n"):
        _=line.split(':')
        if _[0].lower()=='question':
           # print('c1',_[1])
            qlist.append(_[1])
        if _[0].lower()=='response':
           # print('c2',_[1])
            alist.append(_[1])

    ret_val=dict(zip(qlist, alist))
    return ret_val



def match_faq(user_input, faq_dict, paraphase_model):
    questions=list(faq_dict.keys())
    
    all_qscores=[]
    for question in questions:
        question_scores=compare_meaning(user_input, question, paraphase_model)
        all_qscores.append(question_scores[0][0]['score'])
        
    print('all_qscores:',all_qscores)
    i=0
    best_match=''
    best_score=0
    print(questions)
    for qscore in all_qscores:
        print('qscore is ', qscore, questions[i])
        if qscore > best_score:
            best_match=questions[i]
            best_score=qscore
        i=i+1
    print(question_scores, best_match, best_score)

    return best_match, best_score

def single_chatgpt_response(prompt_txt):
    open_ai_key='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'
    import openai
    import os
    os.environ['OPENAI_Key']=open_ai_key
    openai.api_key=os.environ['OPENAI_Key']
    input_txt='whats your question? Type exit if done!!'
  
    response=openai.Completion.create(engine='text-davinci-003',prompt=prompt_txt,max_tokens=200)
    return (response['choices'][0]['text'])
    

def keywords_extraction(text, pos_pipeline):
    import re
    cleaned_text = re.sub(r'[^\w\s]', '', text)

    out=pos_pipeline(cleaned_text)

    all_nouns=[p['word'] for p in out if p['entity']=='NOUN' and not p['word']=='<unk>']
    all_verbs=[p['word'] for p in out if p['entity']=='VERB']
    all_nouns, all_verbs
    ret_dict={}
    ret_dict['nouns']=all_nouns
    ret_dict['verbs']=all_verbs
    return ret_dict


def test_kw_extraction():
    from transformers import pipeline
    text="my text for named entity recognition is here ."
    ret_dict=keywords_extraction(text, pos_pipeline)
    ret_dict

    
    return input_msg


def stub_entity(question, reverse_stub=False):
    entity_marker_dict={}
    entity_marker_dict['stock']='AAPL'
    entity_marker_dict['economic_factor']='interest'
    stubbed=question
    if reverse_stub:
        tlist=extract_tickers(question)
        for ticker in tlist:
            stubbed=stubbed.replace(ticker, '[stock]')
        return stubbed
    else:
        for k in entity_marker_dict:
            stubbed=stubbed.replace(f'[{k}]', entity_marker_dict[k])
    return stubbed
                      
def get_all_nouns_verbs(msg_list, pos_pipeline):
    all_nouns=[]
    all_verbs=[]
    for question in msg_list:
        stubbed_q=stub_entity(question)
        ret_dict=keywords_extraction(stubbed_q, pos_pipeline)
        nouns=ret_dict['nouns']
        verbs=ret_dict['verbs']
        all_nouns=all_nouns+nouns
        all_verbs=all_verbs+verbs
    return ret_dict
    

def gen_intent_yaml_dict(intent_dict_msg, enhanced_ner, pos_pipeline):

    
    intent_yaml_dict={}
    trivials_list=['currently', 'time', 'towards', 'receiving', 'saying', 'general', 'overall', 'affected', 'effect', 'impact', 'implications', 'display']
    for k in intent_dict_msg.keys():
        print(k)
        intent_yaml_dict[k]={}
        intent_yaml_dict[k]['name']=k
        intent_yaml_dict[k]['auto']=True
        intent_yaml_dict[k]['sample_input']=stub_entity(intent_dict_msg[k]['sample_input'])

        msg_list=[re.sub(r'[^\w\s]', '', msg) for msg in intent_yaml_dict[k]['sample_input'].split('\n')]
        intent_yaml_dict[k]['msg_list']=msg_list
        all_tokens, all_keywords=get_tokens_keywords_from_utterence(msg_list, enhanced_ner)
        nouns_verbs_dict=get_all_nouns_verbs(msg_list, pos_pipeline)
        intent_yaml_dict[k]['pattern1']=all_keywords
        if not nouns_verbs_dict is None:
            intent_yaml_dict[k]['pattern2n']=nouns_verbs_dict['nouns']
            intent_yaml_dict[k]['pattern2v']=nouns_verbs_dict['verbs']
        intent_yaml_dict[k]['pattern']=[]
        for k2 in ['pattern1', 'pattern2n', 'pattern2v']:
            intent_yaml_dict[k]['pattern']=intent_yaml_dict[k]['pattern']+intent_yaml_dict[k][k2]
        intent_yaml_dict[k]['pattern']=list(set(intent_yaml_dict[k]['pattern']))
        intent_yaml_dict[k]['pattern']=[kw for kw in intent_yaml_dict[k]['pattern'] if not kw in trivials_list]

    return intent_yaml_dict

def get_list_match_count(wlist1, wlist2):
    # convert the lists to sets
    set1 = set(wlist1)
    set2 = set(wlist2)

    # get the common elements using set intersection
    common_elements = set1.intersection(set2)

    # get the count of common elements
    count = len(common_elements)
    return count


def match_intent_by_keywords(input_msg, intent_yaml_dict):
    match_dict={}
    for k in intent_yaml_dict:
        pattern_list=intent_yaml_dict[k]['pattern']
        match_cnt=get_list_match_count(input_msg.split(' '), pattern_list)
        match_dict[k]=match_cnt
    return match_dict




def match_intent_by_meanings(input_msg, intent_yaml_dict, paraphase_model):
    match_dict={}
    for k in intent_yaml_dict:
        questions=intent_yaml_dict[k]['msg_list']
        match_score=compare_meaning(input_msg, questions, paraphase_model)
        print(input_msg, questions, match_score)
        match_dict[k]=match_score
    return match_dict

def get_best_meaning_match(meaning_match_dict):
    intent_score={}
    for k in meaning_match_dict:
        max_score=max([v['score'] for v in meaning_match_dict[k][0]])
        print(k, max_score)
        intent_score[k]=max_score
    intent_score  
    max_key = max(intent_score, key=intent_score.get)
    max_key
    if intent_score[max_key]>0.5:
        return max_key
    return None
     
def extract_tickers(sentence):
    import re
    # Use a regular expression to match words that end with ".hk", ".au", or ".to"
    pattern = re.compile(r'\b[A-Z]+[.ax|.hk|.to]*\b')
    return re.findall(pattern, sentence)



def test_qc_augment_utterances(msg_list):
    qcmodel = QualityControlPipeline('sentences')
    new_utterences=qc_augment_utterances(msg_list, qcmodel)
    print(new_utterences)

def test_get_tokens_from_augmented_utterence(all_msg):
    all_tokens , all_keywords=get_tokens_keywords_from_utterence(all_msg)
    all_tokens , all_keywords


def test_compare_faq():
    fname='faq1.txt'

    faq_dict=load_faq(fname='faq1.txt')
    questions=faq_dict.keys()
    user_input='can you tell me something about double top pattern?'
    #
    from sentence_transformers import SentenceTransformer

    paraphase_model = SentenceTransformer('paraphrase-distilroberta-base-v1')
    compare_meaning(user_input, questions, paraphase_model)
    print(faq_dict.keys())

def match_intent_pipeline(input_msg, intent_yaml_dict,paraphase_model=None):
    input_msg=stub_entity(input_msg, reverse_stub=True)
    keyword_match_dict=match_intent_by_keywords(input_msg, intent_yaml_dict)
    keyword_match_dict
    keyword_match_intent=max(keyword_match_dict, key= keyword_match_dict.get)
    keyword_match_intent

    if paraphase_model is None:
        return keyword_match_intent

    meaning_match_dict=match_intent_by_meanings(input_msg, intent_yaml_dict,paraphase_model)
    meaning_matched_intent=get_best_meaning_match(meaning_match_dict)

    matched_intent=None
    if meaning_matched_intent is None:
        if keyword_match_dict[keyword_match_intent]>0:
            print('keyword mathced')        
            matched_intent=keyword_match_intent
    else:
        print('meaning mathced')
        matched_intent=meaning_matched_intent

    return matched_intent

def prepare_model():
    import re, yaml
    batch_msg="""
    How will [stock] be impacted by [economic_factor]?
    What effect will [sector rotation] have on [stock]?
    How is [stock] affected by [macro trend]?
    What is the impact of [macroeconomic event] on [stock]?
    Can you tell me how [stock] is influenced by [sector]?
    What are the implications of [industry news] on [stock]?
    """

    #Intent 2: Port the chart for a stock
    batch_msg2="""
    any pattern for ['stock']?
    Show me the chart for [stock] over the past month.
    Can you display the stock price chart for [stock]?
    Give me a graph of [stock] over the past week.
    What does the chart for [stock] look like today?
    I want to see the historical price data for [stock].
    Can you show me the trends for [stock] over the past year?
    Can you show me the patterns for [stock] over the past year?
    """
    #Intent 3: Ask for sentiment for a stock
    batch_msg3="""
    How is the market sentiment towards [stock]?
    What is the overall sentiment on [stock] in the market?
    Can you tell me the general sentiment for [stock]?
    Is [stock] currently receiving positive or negative sentiment?
    What are people saying about [stock]?
    What is the sentiment score for [stock]?
    """
    from transformers import AutoTokenizer, AutoModelForTokenClassification
    enhanced_ner=get_enhanced_ner_pipeline()
    parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5", use_gpu=False)
    pos_tokenizer = AutoTokenizer.from_pretrained("TweebankNLP/bertweet-tb2_ewt-pos-tagging")
    pos_model = AutoModelForTokenClassification.from_pretrained("TweebankNLP/bertweet-tb2_ewt-pos-tagging")
    pos_pipeline = pipeline('token-classification', model=pos_model, tokenizer=pos_tokenizer)
    paraphase_model = SentenceTransformer('paraphrase-distilroberta-base-v1')
    
    intent_dict_msg={}
    intent_dict_msg['macro']={'sample_input':batch_msg}
    intent_dict_msg['chart']={'sample_input':batch_msg2}
    intent_dict_msg['sent']={'sample_input':batch_msg3}
    intent_yaml_dict=gen_intent_yaml_dict(intent_dict_msg, enhanced_ner, pos_pipeline)
    with open('intent_chat.yaml', 'w') as f:
        yaml.dump(intent_yaml_dict, f)
    
    return intent_yaml_dict, paraphase_model

def keyword_match_intent(input_msg, intent_yaml_dict=None):

    if intent_yaml_dict is None:
        with open('intent_chat.yaml', 'r') as f:
            intent_yaml_dict=yaml.safe_load(f )
    
    input_msg2=stub_entity(input_msg, reverse_stub=True)
    stubbed_input_msg=stub_entity(input_msg2)
    matched=match_intent_pipeline(stubbed_input_msg, intent_yaml_dict)
    return matched


def response_to_input_msg(input_msg, intent_yaml_dict,paraphase_model, include_gpt=False):
    input_msg2=stub_entity(input_msg, reverse_stub=True)
    stubbed_input_msg=stub_entity(input_msg2)

    matched_intent=match_intent_pipeline(stubbed_input_msg, intent_yaml_dict,paraphase_model)
    matched_intent
    #xyz

    if not matched_intent is None:
        tickers=extract_tickers(input_msg)        
        print('call rbmq task with intent ',matched_intent, ' for tickers:', tickers)            
        if include_gpt:
            resp=single_chatgpt_response(input_msg)
            print(resp)
        

    #xyz    ticker=extract input_msg
    else:
        best_match, best_score=match_faq(input_msg, faq_dict, paraphase_model)
        if best_score<0.5:
            print('no match, fallback to gpt')
            resp=single_chatgpt_response(input_msg)
            print(resp)
        else:        
            print('from faq:', faq_dict[best_match])
            

    
        
def test_run():
    input_msg='show me the chart of GOOG'
    input_msg='how does the macro economics envirnoment affect META?'
    input_msg='how does other sectors movement affect QQQ?'

    #input_msg2=stub_entity(input_msg, reverse_stub=True)


    #stubbed_input_msg=stub_entity(input_msg2)

    #print(input_msg2, stubbed_input_msg)
    include_gpt=True
    intent_yaml_dict, paraphase_model=prepare_model()
    response_to_input_msg(input_msg, intent_yaml_dict,paraphase_model, include_gpt=include_gpt)

def test_keyword_intent_match():
    input_msg='how does other sectors movement affect QQQ?'
    #intent_yaml_dict, paraphase_model=prepare_model()
    include_gpt=True
    #
    #response_to_input_msg(input_msg, intent_yaml_dict,paraphase_model, include_gpt=include_gpt)
    matched_intent=keyword_match_intent(input_msg, intent_yaml_dict=None)
    return matched_intent


#test_run()
#x=test_keyword_intent_match()
print(x)
