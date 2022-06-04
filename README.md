# Web Page Search Engine
A web page search engine with data collection, pre-processing, and four search methods

## Methods
- Boolean Retrieval
- TF-IDF
- A model based on transformers
- Fasttext

### Example
```python
from engines.tf_idf_searcher import *
tfidf = TFIDFSearcher(data)
tfidf.search('Nba final', 10)
```
### Output
```
[DataOut(url='https://www.nba.com/', score=0.6363178214536787),
 DataOut(url='https://bleacherreport.com/nba', score=0.5865015329407696),
 DataOut(url='https://en.wikipedia.org/wiki/National_Basketball_Association', score=0.5204316885560344),
 DataOut(url='https://www.nba.com/games', score=0.5161888324511272),
 DataOut(url='https://www.cbssports.com/nba/', score=0.46908253300904185),
 DataOut(url='https://www.aol.com/sports/', score=0.4554845897045064),
 DataOut(url='https://www.basketballnews.com/', score=0.4220043434704418),
 DataOut(url='https://www.basketball-reference.com/', score=0.37876215200493285),
 DataOut(url='https://www.theguardian.com/us/sport', score=0.3279570916447034),
 DataOut(url='https://www.si.com/', score=0.3227106519805717)]
```

### Contributors
| Members |
| :---:   |
| `Mohammadamin Aryan`  |
| `Yalda Shabanzadeh` |
| `Karaneh Keypour`  |
