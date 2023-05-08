import sqlite3
class sqliteDocStoreUtil:
    
    @staticmethod
    def init_sqlite_docstore(tablename, con):
        sql_d0=f"""
        CREATE TABLE {tablename}  (
          body TEXT,
            id TEXT GENERATED ALWAYS AS (json_extract(body, '$.id')) VIRTUAL NOT NULL);
        """
        sql_d1="""
        CREATE INDEX xid on %s (id);
        """ % tablename
        sql_d2=f"""
        ALTER TABLE {tablename} ADD COLUMN text TEXT
            GENERATED ALWAYS AS (json_extract(body, '$.text')) VIRTUAL;
        """


        cur = con.cursor()
        try:
            cur.execute(sql_d0)
            cur.execute(sql_d1)
            cur.execute(sql_d2)
            res = cur.execute("SELECT name FROM sqlite_master")
            print(res.fetchone())
        except Exception as e:
            print(e)

        return 

    @staticmethod    
    def dict2json2sqlite3(mydict, _id, tablename, con):
        cur = con.cursor()
        if 'id' not in mydict:
            mydict['id']=_id
        if not _id == mydict['id']:
            mydict['id']=_id
        dict_str="%s" % mydict
        dict_str=dict_str.replace("''", 'xxxxx')
        dict_str=dict_str.replace("'", '"')
        dict_str=dict_str.replace('xxxxx', "''")
        sql2="""
    INSERT INTO %s  VALUES(json('%s'));
    """ % (tablename, dict_str)
        print('sql insert is ',sql2)
        res = cur.execute(sql2)

        print(res)
        return
    @staticmethod
    def test_insert(con):
        cur = con.cursor()
        sql2="""
    INSERT INTO %s  VALUES(json('{"id":44, "text":"test"}'));
    """ % tablename
        print('sql2 is ',sql2)
        res = cur.execute(sql2)
        sql2="""
    INSERT INTO %s  VALUES(json('{"id":45, "text":"test"}'));
    """ % tablename
        print('sql2 is ',sql2)
        res = cur.execute(sql2)
        print(res.fetchone() is None)

    @staticmethod        
    def test_run(tablename, con):
        cur = con.cursor()    
        mydict={"id":"88","time":1678258954,"expires":1678302154,"event":"message","topic":"jl892","message":"happy new year"}
        mydict
        _id=98
        sqliteDocStoreUtil.dict2json2sqlite3(mydict, _id, tablename, con)
        _id=99
        sqliteDocStoreUtil.dict2json2sqlite3(mydict, _id, tablename, con)

        sql3=f"""
        Select * from {tablename}
        """
        res = cur.execute(sql3)

        print(res.fetchall())
        con.commit()
        con.close()
        
    @staticmethod        
    def view_data(tablename, con):
        sql3=f"""
        Select * from {tablename}
        """
        print('view data sql:',sql3)
        cur = con.cursor()    
        res = cur.execute(sql3)

        print(res.fetchall())

    @staticmethod
    def test():
        con = sqlite3.connect("tutorial.db")
        tablename='gpt_log'
        sqliteDocStoreUtil.init_sqlite_docstore(tablename, con)
        sqliteDocStoreUtil.test_run(tablename, con)    
        
    @staticmethod
    def log_gpt_chat(reponse_dict, user_tag, dbfname="gpt35.db"):
        con = sqlite3.connect(dbfname)
        tablename='gpt_log'
        sqliteDocStoreUtil.init_sqlite_docstore(tablename, con)
        sqliteDocStoreUtil.dict2json2sqlite3(reponse_dict, _id=user_tag, tablename=tablename, con=con)
        con.commit()
        con.close()
#sqliteDocStoreUtil.test()

def review_data(dbfname="gpt35.db"):
    con = sqlite3.connect(dbfname)
    tablename='gpt_log'
    sqliteDocStoreUtil.view_data(tablename, con)


