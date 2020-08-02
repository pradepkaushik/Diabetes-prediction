import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'Age':40, 'Gender':0, 
                            'Polyuria':0,
                            'Polydipsia':1,
                            'sudden weight losss':0,
                            'weakness':1,
                            'Polyphagia':0,
                            'Genital thrush':0,
                            'visual blurring':0,
                            'Itching':1,
                            'Irritability':0,
                            'delayed healing':1,
                            'partial paresis':0,
                            'muscle stiffness':1,
                            'Alopecia':1,
                            'Obesity':1})

print(r.json())