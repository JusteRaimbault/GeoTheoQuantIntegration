import requests
from lxml import html
import pandas

#years=range(1993, 2011, 2)
years=[1993]
url='https://thema.univ-fcomte.fr/theoq/fr/publications.php?menus=publications&annee='
# missing abstracts: could iterate by hand, but faster to list by hand
missing_abstracts={
        '1993' = [20],
        '1995' = [0,1,2],
        '1997' = [],
        '1999' = [2,3],
        '2001' = [9,17,28,36,38],
        '2003' = [9,13,16,19,22,29,30,31,37],
        '2005' = 
        }

headers={
        'User-Agent':'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36'
        }

data = pandas.DataFrame({'Title':[],'Author':[],'Abstract':[],'Year':[]})

for year in years:
    print('Collecting from : '+url+str(year))

    try:
        response = requests.get(url+str(year), headers=headers, timeout=10)
        response.raise_for_status() 
        html_content = response.text
    except requests.exceptions.RequestException as e:
        print(f"Error getting html: {e}")
        html_content = None

    if html_content:
        tree = html.fromstring(html_content)
        abstracts = ["".join(abstract.text_content()).replace("\n", " ") for abstract in tree.xpath("//div[@id='corpsavecmenu']//div[@class='resumelong']")]
        authors = [author.strip() for author in tree.xpath("//div[@id='corpsavecmenu']//div[@id='auteur']/text()")]
        titles = [title.strip() for title in tree.xpath("//div[@id='corpsavecmenu']//div[@id='titre']/a/text()")]
        print(str(len(abstracts))+' - '+str(len(authors))+' - '+str(len(titles)))
        data = pandas.concat(data,pandas.DataFrame({'Title':titles,'Author':authors,'Abstract':abstracts,'Year':[str(year)]*len(titles)})
)        

print(data)




