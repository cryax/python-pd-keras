# -*- coding: utf-8 -*-

import MySQLdb
import pandas as pd
#from younet_fbc.libs.SupportLibs import save_to_csv

server1 = dict()
server1['host'] = "209.133.193.146"
server1['user'] = "rnd"
server1['passwd'] = "nRd@M$sqLYnM$di2"
server1['db_name'] = "monitoring_profiles"


class UserDatabase(object):
    def __init__(self, server1_option = None, server2_option= None):
        """
        Initialize FBid tool with host ,user , password and database name for connecting to database.

        :param host: host ip
        :param user: username .
        :param passwd: password of user
        :param db_name: database name.
        """
        self.server1 = server1_option if server1_option else server1
#        self.server2 = server2_option if server2_option else server2
        
    def select_user2(self,listcountry):
        def sql(list_identity, cursor):
            select = "SELECT *"
            _from = '''FROM `monitor_user_checkins` WHERE country IN('{}')'''\
                .format("','".join(map(str,list_identity)))

            sql = select + _from
            sql = "SELECT *FROM `monitor_user_checkins` \
            WHERE country IN('Albania','Andorra','Australia','Austria','Belarus','Belgium','Bosnia and Herzegovina','Bulgaria','Canada','Croatia','Czech Republic','Denmark','Estonia','Faroe Islands','Finland','France','Germany','Gibraltar','Greece','Guernsey','Hungary','Iceland','Ireland','ital','Japan','Jersey','Kosovo','Latvia','Liechtenstein','Lithuania','Luxembourg','Macedonia','Malta','Moldova','Monaco','Montenegro','Netherlands','New Zealand','Norway','Poland','Portugal','Romania','Russia','San Marino','Serbia','Slovakia','Slovenia','South Korea','Spain','Sweden','Switzerland','Ukraine','United Kingdom','United States') and created_date > '2015-06-06 00:00:00'" 
            print ('sql selector: ',sql)
            print "2: ", len(list_identity)
            cursor.execute(sql)
            data = cursor.fetchall()
#            columns = ["identity", "fullname", "gender"]
#            df_info = pd.DataFrame(list(data), columns=columns)
#            save_to_csv(dataframe=df_info, dir="Data", file_name="User_info.csv")
            return data

        connection_server1 = MySQLdb.connect(host=self.server1['host'], user=self.server1['user'],
                                             passwd=self.server1['passwd'], db=self.server1['db_name'],
                                             charset='utf8')

        # prepare a cursor object using cursor() method
        cursor1 = connection_server1.cursor()

        data = sql(list_identity=listcountry, cursor=cursor1)

        # close connect to database
        cursor1.close()
        connection_server1.close()
        return data
    

if __name__ =='__main__':
    listcountry =["Albania","Andorra","Australia","Austria","Belarus","Belgium","Bosnia and Herzegovina","Bulgaria","Canada","Croatia","Czech Republic","Denmark","Estonia","Faroe Islands","Finland","France","Germany","Gibraltar","Greece","Guernsey","Hungary","Iceland","Ireland","ital","Japan","Jersey","Kosovo","Latvia","Liechtenstein","Lithuania","Luxembourg","Macedonia","Malta","Moldova","Monaco","Montenegro","Netherlands","New Zealand","Norway","Poland","Portugal","Romania","Russia","San Marino","Serbia","Slovakia","Slovenia","South Korea","Spain","Sweden","Switzerland","Ukraine","United Kingdom","United States"]
    data = UserDatabase().select_user2(listcountry)
    
    
    # list_id = list(pd.read_excel("result1.xlsx").id)
    # UserDatabase().select_user(list_identity=list_id)

