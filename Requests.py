import requests
import json

#Base of url, change it depending where you host api
base = "http://localhost:5000/"

#Instance of data to test the models
data = { "0": 2804, "1": 139, "2": 9, "3": 268, "4": 65, "5": 3180, "6": 234, "7": 238, "8": 135, "9": 6121,
    "10": 1, "11": 0, "12": 0, "13": 0, "14": 0, "15": 0, "16": 0, "17": 0, "18": 0, "19": 0, 
    "20": 0, "21": 0, "22": 0, "23": 0, "24": 0, "25": 1, "26": 0, "27": 0, "28": 0, "29": 0,
    "30": 0, "31": 0, "32": 0, "33": 0, "34": 0, "35": 0, "36": 0, "37": 0, "38": 0, "39": 0,
    "40": 0, "41": 0, "42": 0, "43": 0, "44": 0, "45": 0, "46": 0, "47": 0, "48": 0, "49": 0,  
    "50": 0, "51": 0, "52": 0, "53": 0
}

#Hedders for request
headers = {'Content-Type': 'application/json','Content-Length': '75'}

#Reuqest for all algortithms
url = base + "heuristic"
response = requests.get(url, json= data,headers=headers)
print(response.content)

url = base + "knn"
response = requests.get(url, json= data,headers=headers)
print(response.content)

url = base + "dtree"
response = requests.get(url, json= data,headers=headers)
print(response.content)

url = base + "ann"
response = requests.get(url, json= data,headers=headers)
print(response.content)
