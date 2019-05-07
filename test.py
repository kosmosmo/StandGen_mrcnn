a = {"info":[((26,False),(39,True)),(300,300)],"test":"yest"}
import json
path = 'stands/sp/01/'
with open(path + 'info.json') as json_file:
    info = json.load(json_file)
    print(info)