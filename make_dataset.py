from wik import get_wiki_pages
from emb import VectorDB

from tqdm import tqdm 

if __name__=="__main__":
	db=VectorDB()
	for t in tqdm(['art','python','math','history','philosophy']):
		pages=get_wiki_pages(t)
		texts = [text for page in pages for text in page]
		print(f'number of texts in call: {len(texts)}')
		db.add(texts)
	db.save('wiki_db.pkl')