import re
import pickle, gzip, base64


class tgbotHelper:
    
    @staticmethod
    def get_message_dict( msg_txt, img_list=[], send_df=None):
        obj_dict={}
        obj_dict['message_txt']=msg_txt
        if not send_df is None:
            obj_dict['send_df']=send_df
        if img_list is None:
            img_list=[]
        obj_dict['img_list']=img_list

        ret=tgbotHelper.b64_pickle_output_dict_with_img(obj_dict)
        return ret

    @staticmethod    
    def b64_pickle_output_dict_with_img(obj_dict):
        '''
        it will read all img file name list in 'img_list' params and 
        load them using PIL.Image
        required field in dict for reply ['message_txt', 'img_obj' (list of image), 'send_df' (optional)
        '''
        import pickle, gzip,os
        from PIL import Image
        save_dict=obj_dict.copy()
        if 'img_list' in save_dict:
            img_list=save_dict['img_list']
        else:
            print('missed img:' , save_dict.keys())
            img_list=[]
        save_dict['img_obj']=[]

        for imgfname in img_list:
            print('reading image ',imgfname)
            if '.jpg' in imgfname or '.png' in imgfname:
    #            save_dict['img_obj'].append(Image.open(imgfname))
                with open(imgfname, 'rb') as f:
                    save_dict['img_obj'].append(f.read())
        #print(save_dict)
        pickled_bytes = pickle.dumps(save_dict)
        # Encode the pickled bytes to base64 string
        encoded_string = base64.b64encode(pickled_bytes).decode('utf-8')
        return encoded_string

    @staticmethod
    def b64_load_output_dict_pickle_with_img(b64_encoded_string, save_img=False):
        '''
        input img_list will now be b64 encoded img_obj
        '''
        pickled_bytes = base64.b64decode(b64_encoded_string)

        # Unpickle the object from the bytes
        load_dict = pickle.loads(pickled_bytes)


        if 'img_obj' in load_dict:
            #print(load_dict)
            img_obj=load_dict['img_obj']
            if 'img_obj' in load_dict and save_img:
                for b in load_dict['img_obj']:
                    fname=f'photos/{i}.jpg'
                    fname=load_dict['img_list'][i]
                    i=i+1
                    print(fname)
                    with open(fname, 'wb') as f:
                        f.write(b)

        return load_dict    

def test_run():
    msg_txt='testing msg'
    b64msg=tgbotHelper.get_message_dict( msg_txt, img_list=['photos/file_1.jpg', 'photos/file_2.jpg'], send_df=None)
    ndict=tgbotHelper.b64_load_output_dict_pickle_with_img(b64msg)
    print(ndict.keys())
#test_run()
