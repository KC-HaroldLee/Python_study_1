import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import time

result = pd.DataFrame()
for i in range(500001, 550000) :
    URL = 'https://www1.president.go.kr/petitions/'+str(i)

    response = requests.get(URL)
    html = response.text # 쌩 텍스트
    soup = BeautifulSoup(html, 'html.parser')
    title = soup.find('h3', class_='petitionsView_title')
    count = soup.find('span', class_='counter')
    for content in soup.select('div.petitionsView_write > div.View_write'):
        content
        pass
    a=[]
    for tag in soup.select('ul.petitionsView_info_list > li'): 
        a.append(tag.contents[1]) # 분류, s날짜, e날짜, 계정
    
    if len(a) != 0 :
        dfl = pd.DataFrame({'start' : [a[1]],
                            'end' : [a[2]],
                            'count' : [count.text],
                            'title' : [title.text],
                            'content' : [content.text.strip()[0:13000]]
                           })
        
        result = pd.concat([result, dfl])
        result.index = np.arange(len(result))

    if i % 50 == 0 :
        print("SLEEP! now is ", str(i))
        print("curr DataLength is ", str(len(result)))
        time.sleep(15)

print(result.shape)

df = result
df.head()

df.to_csv('data/crawling.csv', index = False, encoding = 'utf-8-sig')

print('end')