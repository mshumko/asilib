import requests
from datetime import datetime

from bs4 import BeautifulSoup

day = datetime(2017, 4, 13, 5, 10)

station = 'LUCK'

url = (f'https://data.phys.ucalgary.ca/sort_by_project/GO-Canada/REGO/stream0/'
        f'{day.year}/{str(day.month).zfill(2)}/{str(day.day).zfill(2)}/')

print(url)

r = requests.get(url)
soup = BeautifulSoup(r.content, 'html.parser')

hrefs = soup.find_all('a', href=True)

for href in hrefs:
    if station.lower() in href.text:
        station_url = href.text
        break
    else:
        station_url = ''

if station_url == '':
    raise NotADirectoryError(f'The the directory for {station} not found in \n{url}.')

# 'https://data.phys.ucalgary.ca/sort_by_project/GO-Canada/REGO/stream0/2017/04/13/luck_rego-649/ut05/'

url2 = url + station_url + f'ut{str(day.hour).zfill(2)}/'

r = requests.get(url2)
soup = BeautifulSoup(r.content, 'html.parser')

hrefs = soup.find_all('a', href=True)


date_str = day.strftime('%Y%m%d_%H%M')
for href in hrefs:
    # '20170413_0510'
    if date_str in href.text:
        station_url = href.text
        print(station_url)
        break
    else:
        station_url = ''

url3 = url2 + station_url
r = requests.get(url3, allow_redirects=True)
with open(station_url, 'wb') as f:
    f.write(r.content)