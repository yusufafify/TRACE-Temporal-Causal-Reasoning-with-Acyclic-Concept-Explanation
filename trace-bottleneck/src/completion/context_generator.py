import nltk
from sentence_transformers import SentenceTransformer
#from metapub import PubMedFetcher

import requests
from bs4 import BeautifulSoup
import json
from tqdm import tqdm
import time
import os
from duckduckgo_search import DDGS
import re

PAPER_SCRAPER_PROMPT = """Your task is to create the most effective search query to find information that answers the user's question. \
Your query will be used to search scientific articles from the web.\
From the given query, produce a query that will help to find the most relevant articles.\
NOTE: be short and concise.\

This is the question: {question}. 
Provide the final query without beackets.
"""

WWW_SCRAPER_PROMPT = """Your task is to create the most effective search query to find information that answers the user's question. \
Your query will be used to search the web using a web engine (e.g. google, duckduckgo).\
NOTE: be short and concise.\

This is the question: {question}. 
Provide the final query without beackets.
"""

EXPANDED_QUERY_PROMPT = """Rephrase the query to align semantically with similar target texts while maintaining its core meaning.\
Output the expanded query enclosed within <expanded_query> tags (e.g. <expanded_query>[example_query]</expanded_query>).\
NOTE: be very short and concise.\

Query: {query}
Expanded Query: 
"""

class Context_generator:
    def __init__(self, 
                 llm='gpt-4o_client', 
                 embedder='multi-qa-mpnet-base-dot-v1', 
                 query_strategy='standard',
                 chunking_strategy='sentence', 
                 source='pubmed', 
                 n_documents_per_source=10,
                 context_length=10,
                 dataset=None,
                 verbose=False):
        
        self.llm = llm
        self.embedder = SentenceTransformer(embedder)
        self.query_strategy = query_strategy
        self.chunking_strategy = chunking_strategy
        if chunking_strategy == 'sentence':
            nltk.download('punkt_tab')
        self.source = source
        self.n_documents_per_source = n_documents_per_source
        self.context_length = context_length
        self.dataset = dataset
        if dataset in ['colormnist', 'celeba', 'celeba_unfair', 'sachs', 'sachs_ood']:
            self.source = 'local'
        self.verbose = verbose
            

    """
    def get_abstracts_pubmed(self, query):
        fetch = PubMedFetcher()
        pmids = []
        try:
            pmids = fetch.pmids_for_query(query, retmax=self.n_documents_per_source)
        except:
            print(f"Error fetching PMIDs for query: {query}")

        abstracts = []
        for pmid in pmids:
            try:
                abstracts.append(fetch.article_by_pmid(pmid).abstract)
            except:
                print(f"Error fetching abstract for PMID: {pmid}")
        return abstracts

        retries = 0
        while retries < max_retries:
            try:
                response = requests.get(url, timeout=10)  # Adjust timeout as needed
                response.raise_for_status()  # Raise an error for HTTP codes 4xx/5xx
                return response
            except requests.RequestException as e:
                retries += 1
                print(f"Attempt {retries} failed: {e}")
                if retries < max_retries:
                    time.sleep(wait_seconds)  # Wait before retrying
                else:
                    print("Max retries reached. Giving up.")
                    raise

    """

    def get_abstracts_pubmed(self, query, sleep_time=10, retries=5):
        try:
            print('Scrpaing PubMed papers...')
            # we lmit the number of papers to 3 to avoid network errors
            url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={query}&retmax={3}&retmode=json"
            response = requests.get(url)
            data = response.json()
            id_list = data['esearchresult']['idlist']
            print('PubMed URLs obtained!')
            print('Fetching Abstracts...')
        except:
            print(f"Network Error fetching PMIDs for query: {query}")
            return []
        papers = []
        for pubmed_id in tqdm(id_list):
            fetch_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={pubmed_id}&retmode=xml"
            for attempt in range(retries):
                try:
                    fetch_response = requests.get(fetch_url)
                except requests.RequestException as e:
                    print(f"Attempt {attempt + 1} failed: {e}")
                    time.sleep(sleep_time)
            soup = BeautifulSoup(fetch_response.content, 'xml')
            # title = soup.find('ArticleTitle')
            abstract = soup.find('Abstract')
            if abstract:
                abstract_texts = abstract.find_all('AbstractText')
                abstract_plain_text = ' '.join([text.get_text() for text in abstract_texts])
                abstract = abstract_plain_text
                papers.append(abstract)
        return papers

    def get_abstracts_arxiv(self, query, sleep_time=10, retries=5):
        print('Scraping arXiv papers...')
        url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results={self.n_documents_per_source}"
        for attempt in range(retries):
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                break
            except requests.RequestException as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                time.sleep(sleep_time)
        else:
            raise Exception("Max retries reached. Giving up.")
        
        soup = BeautifulSoup(response.content, 'xml')
        entries = soup.find_all('entry')
        papers = []
        for entry in tqdm(entries):
            for attempt in range(retries):
                try:
                    abstract = entry.find('summary').text
                    papers.append(abstract)
                    break
                except Exception as e:
                    print(f"Attempt {attempt + 1} failed: {e}")
                    time.sleep(sleep_time)
        return papers

    def extract_and_clean_content(self, url):
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        # Remove script and style elements
        for script_or_style in soup(['script', 'style']):
            script_or_style.decompose()
        # Get text and clean it
        text = soup.get_text()
        text = re.sub(r'\s+', ' ', text)  
        text = text.strip()
        return text

    def get_web_search_results(self, query):
        print('Scraping the web...')
        results = DDGS().text(query, max_results=self.n_documents_per_source)
        documents = []
        for result in tqdm(results):
            try:
                doc = self.extract_and_clean_content(result['href'])
                documents.append(doc)
            except Exception as e:
                if self.verbose:
                    print(f"An error occurred while fetching the document: {e}")
        return documents

    def embedd_and_rank_text(self, query, chunks, model):
        query_embedding = model.encode(query, convert_to_tensor=True) 
        abstract_embeddings = model.encode(chunks, convert_to_tensor=True)
        if query_embedding.is_cuda:
            query_embedding = query_embedding.cpu()
        if abstract_embeddings.is_cuda:
            abstract_embeddings = abstract_embeddings.cpu()
        similarities = model.similarity(query_embedding, abstract_embeddings)[0]
        sorted_indices = similarities.argsort(descending=True)
        ranked_chunks = [chunks[i] for i in sorted_indices]
        if self.context_length != None:
            ranked_chunks = ranked_chunks[:self.context_length]
        return ranked_chunks
    
    def parse_query(self, answer):
        start_tag = "<expanded_query>"
        end_tag = "</expanded_query>"
        start_index = answer.find(start_tag) + len(start_tag)
        end_index = answer.find(end_tag)
        if start_index != -1 and end_index != -1:
            answer = answer[start_index:end_index].strip()
        return answer
    
    def expand_query(self, query):
        expanded_query = self.llm.invoke(EXPANDED_QUERY_PROMPT.format(query=query)) 
        expanded_query = self.parse_query(expanded_query)
        if self.verbose:
            print('Expanded query for retrieval:', expanded_query)
        return expanded_query

    def read_and_clean_html(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
        soup = BeautifulSoup(content, 'html.parser')
        cleaned_text = soup.get_text(separator=' ', strip=True)
        return cleaned_text

    def invoke(self, query):
        # Whether to use the user query or a different strategy (e.g. query expansion)
        if self.query_strategy == 'standard':
            query_to_embedd = query
        elif self.query_strategy == 'expansion':
            query_to_embedd = self.expand_query(query)
        else:
            raise ValueError(f"query_strategy ({self.query_strategy}) not implemented")
        
        # Get the documents from the different sources
        if self.source == 'pubmed':
            query = self.llm.invoke(PAPER_SCRAPER_PROMPT.format(question=query)) 
            documents = self.get_abstracts_pubmed(query)
            if self.verbose:
                print('Number of papers obtained for the given query:', len(documents))
        elif self.source == 'arxiv':
            query = self.llm.invoke(PAPER_SCRAPER_PROMPT.format(question=query)) 
            documents = self.get_abstracts_arxiv(query)
            if self.verbose:
                print('Number of papers obtained for the given query:', len(documents))           
        elif self.source == 'www':
            query = self.llm.invoke(WWW_SCRAPER_PROMPT.format(question=query)) 
            documents = self.get_web_search_results(query)
            if self.verbose:
                print('Number of web pages obtained for the given query:', len(documents))  
        elif self.source == 'custom':
            query = self.llm.invoke(PAPER_SCRAPER_PROMPT.format(question=query)) 
            arxiv_documents = self.get_abstracts_arxiv(query)
            query = self.llm.invoke(WWW_SCRAPER_PROMPT.format(question=query)) 
            www_documents = self.get_web_search_results(query)
            documents = arxiv_documents + www_documents
            if self.verbose:
                print('Number of documents retrieved:', len(documents))  
        elif self.source == 'local':
            documents = []
            path = os.getcwd()
            path = path.split("/outputs")[0]
            if self.dataset in ['sachs', 'sachs_ood']:
                doc_path = os.path.join(path, 'src', 'completion', 'documents', "sachs.json")
                with open(doc_path, 'r') as file:
                    data = json.load(file)
                    documents += [doc['content'] for doc in data['documents']]
                doc_path = os.path.join(path, 'src', 'completion', 'documents', f"sachs_paper.html")
                sachs_paper = self.read_and_clean_html(doc_path)
                documents += [sachs_paper]
            else:
                path = os.path.join(path, 'src', 'completion', 'documents', f"{self.dataset}.json")
                with open(path, 'r') as file:
                    data = json.load(file)
                documents += [doc['content'] for doc in data['documents']]



        else:
            raise ValueError("source must be one of 'pubmed', 'arxiv' or 'local'")

        # Now the different documents have to be divided into smaller pieces according to the chunking strategy.
        # We also remove empty chunks
        chunks = []
        for doc in documents:
            if doc!=None:
                if self.chunking_strategy == 'sentence':
                    chunks += nltk.sent_tokenize(doc)
                elif self.chunking_strategy == 'paragraph':
                    chunks += doc.split('\n')
                elif self.chunking_strategy == 'document':
                    chunks += [doc]
                elif self.chunking_strategy == 'sliding_window':
                    window_size = 512
                    step_size = 128
                    for i in range(0, len(doc), step_size):
                        chunk = doc[i:i + window_size]
                        if chunk:
                            chunks.append(chunk)
                else:
                    raise ValueError("chunk_strategy must be one of 'sentence', 'paragraph', 'document' or 'sliding window'")
        chunks = [chunk for chunk in chunks if chunk != '']
        if self.verbose:
            print('Number of chunks:', len(chunks))

        if len(chunks) == 0:
            context = 'No information found.'
        else:
            # The sorted chunks are generated by using the embedder to get the embeddings of the query and the chunks
            context = self.embedd_and_rank_text(query_to_embedd, chunks, self.embedder)
            if self.verbose:
                print('Context length:', len(context))

            # The sorted chunks are transformed into a single string
            context = '\n'.join([f'Background Information {i+1}: '+chunk for i, chunk in enumerate(context)])
        return context
