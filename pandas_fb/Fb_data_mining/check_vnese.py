# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
#import FileTool
import re
listViet = [u'ă',u'â',u'đ',u'ê',u'ô',u'ơ',u'ư',u'ạ',u'ả',u'ã',u'á',u'à',u'ặ',u'ẳ',u'ẵ',u'ằ',u'ắ',u'ậ',u'ẩ',u'ẫ',u'ấ',u'ầ',u'ẽ',u'ẻ',u'ệ',u'ể',u'ễ',u'ề',u'ế',u'ộ',u'ồ',u'ổ',u'ỗ',u'ố',u'ợ',u'ớ',u'ở',u'ỡ',u'ờ',u'ự',u'ử',u'ữ',u'ừ',u'ứ']
#listsurname = FileTool.read_text_file('surname.csv')
C  = 'ph|th|tr|gi|d|ch|nh|ng|ngh|kh|g|gh|c|q|k|t|r|h|b|m|v|d|n|l|x|p|s'
set1 = 'a|ie|oa|oo|ua|uo|uye|ye'
set2 = 'oa|oe|ue|uy'
set3 = 'ai|ao|au|ay|eo|eu|ia|ieu|yeu|iu|oi|oai|oao|oay|oeo|ua|ui|uu|uo|uai|uay|uoi|uou|uya|uyu|uao'
set4 = 'uo|ua|a|e|i|o|u|y'



rule1 = '(%s)?(%s)(%s)$'%(C,set1,C)
print rule1
rule2 = '(%s)?(%s)(%s)?$'%(C,set2,C)
rule3 = '(%s)?(%s)$'%(C,set3)
rule4 = '(%s)?(%s)(%s)?$'%(C,set4,C)

p = [re.compile(rule1), re.compile(rule2), re.compile(rule3), re.compile(rule4)]
def isContainVietChar(data):
    data = data.lower()
    for ele in listViet:
        if ele in data:
            return True
    return False

def isContainVietnameseSurname(data):
    data = data.lower()
    listdata = data.split(' ')
    for word in listdata:
        for ele in listsurname:
            if ele == word:
                return True
    return False

def checkRule(data):
    data = data.lower()
    listdata = data.split(' ')
    for ind,ele in enumerate(listdata):
        for sp in p:
            if sp.match(ele):
                return True
    return False

def checkAll(data):
    if isContainVietChar(data):
        return True
    elif isContainVietnameseSurname(data):
        return True
    elif checkRule(data):
        return True
    else:
        print data
        return False
if __name__ =='__main__':
    print isContainVietChar(u'hóa cht')