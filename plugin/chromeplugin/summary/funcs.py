import pypdf
import trafilatura
from gpt4all import Embed4All
from qdrant_client import QdrantClient, models
from rouge_score.rouge_scorer import RougeScorer
from sklearn.cluster import KMeans
import numpy as np
from scipy.spatial import distance
from transformers import pipeline
from langchain_community.document_loaders import PyPDFLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
from .config import QDRANT_API_KEY, HUGGINGFACE_API_KEY
import fitz
from rouge_score import rouge_scorer


# Functions to extract the text
def extract_text_from_url(url):
    text = trafilatura.fetch_url(url)

    text_to_summarize = trafilatura.extract(text)

    return text_to_summarize


def extract_text_from_pdf_pymupdf(file_path):
    document = fitz.open(file_path)
    text = ""

    for page in document:
        text += page.get_text()
    document.close()
    return text


def extract_text_from_pdf_pypdf(file_path):
    with open(file_path, "rb") as f:
        reader = pypdf.PdfReader(f)
        filtered_text = ''
        number_of_pages = len(reader.pages)
        print(f"There are total {number_of_pages} pages in the original document")

        for i in range(number_of_pages):
            page = reader.pages[i]
            text = page.extract_text()
            filtered_text += text + "\n"

    return filtered_text


def extract_text_from_pdf_langchain_loader(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()

    return pages


def split_text(text, chunk_size=500, chunk_overlap=200):
    # text_splitter = CharacterTextSplitter()
    # docs = text_splitter.create_documents([text])

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.create_documents([text])
    # return docs

    # text_splitter = SemanticChunker(OpenAIEmbeddings())

    # docs = text_splitter.create_documents([text])

    # t = [len(docs[i].page_content) for i in range(len(docs))]

    return docs


# Functions to extract the text

# Function to split and embed the text
def embed_texts(list_of_texts):
    # splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=200, length_function=len)

    # docs = splitter.split_text(text)

    # t = [len(docs[i].split()) for i in range(len(docs))]

    number_of_vectors = len(list_of_texts)

    print(f"There are total {number_of_vectors} vectors")

    embeder = Embed4All()

    embedding_map = {}

    for i in range(number_of_vectors):
        text = list_of_texts[i].page_content
        embedding = embeder.embed(text)
        embedding_map[text] = embedding

    return embedding_map


def save_to_vector_db(embedding_map):
    vectors = list(embedding_map.values())
    payloads = list(embedding_map.keys())
    payloads_to_load = [{"doc": item} for item in payloads]

    number_of_dimensions = len(vectors[0])
    number_of_vectors = len(vectors)

    qdrant_client = QdrantClient(
        url="https://9240ae44-20ec-436e-b4fd-8ce951d197a3.us-east4-0.gcp.cloud.qdrant.io:6333",
        api_key=QDRANT_API_KEY,
    )

    qdrant_client.recreate_collection(
        collection_name="text_to_summarize",
        vectors_config=models.VectorParams(
            size=number_of_dimensions, distance=models.Distance.COSINE
        )
    )

    ids = [i + 1 for i in range(number_of_vectors)]

    qdrant_client.upsert(collection_name="text_to_summarize",
                         points=models.Batch(
                             ids=ids,
                             vectors=vectors,
                             payloads=payloads_to_load
                         ))


def find_elbow_point(embedded_vectors, min_k=3, max_k=15):
    normalized_vectors = embedded_vectors / np.linalg.norm(embedded_vectors, axis=1, keepdims=True)
    wcss = []

    max_k = min(max_k, len(embedded_vectors))

    for i in range(min_k, max_k + 1):
        kmeans = KMeans(n_clusters=i, init="k-means++", max_iter=300, n_init=10, random_state=0)
        kmeans.fit(normalized_vectors)
        wcss.append(kmeans.inertia_)

    deltas = np.diff(wcss)
    second_deltas = np.diff(deltas)
    elbow_point = np.argmax(second_deltas) + 2 + min_k

    print(f"The optimal number of clusters is {elbow_point}")

    return elbow_point


def closest_vectors_with_text(embedded_texts, num_clusters):
    labels = list(embedded_texts.keys())
    embedded_vectors = np.array(list(embedded_texts.values()))

    kmeans = KMeans(n_clusters=num_clusters, init="k-means++", max_iter=100, n_init=10, random_state=0)
    kmeans.fit(embedded_vectors)

    cluster_means = [np.mean(embedded_vectors[kmeans.labels_ == i], axis=0) for i in range(num_clusters)]

    closest = {}

    for mean_vector in cluster_means:
        closest_index = min(range(len(embedded_vectors)),
                            key=lambda v: distance.cosine(embedded_vectors[v], mean_vector))
        closet_vector = embedded_vectors[closest_index]
        closet_key = labels[closest_index]
        closest[closet_key] = closet_vector

    return closest


def summarize_2(texts, one_by_one=True, model="mistralai/Mixtral-8x7B-Instruct-v0.1"):
    # summarizer = pipeline("summarization", model=model)

    generator = pipeline("text-generation", model=model, token=HUGGINGFACE_API_KEY)

    summaries = []
    if one_by_one:
        print("Summarizing texts, one by one...")
        for (i, text) in enumerate(texts):
            word_count = len(text.split())
            print(f"Original text: {text}\n")
            if word_count > 10:
                summary = generator(f"Summarize the following: {text}", do_sample=False)

                summaries.append(summary[0]['summary_text'])
                print(f"Summary {i}: {summary[0]['summary_text']}\n")

    else:
        print("Summarizing overall texts...")
        combined = ", ".join(texts)
        # print("Combined summaries:", combined)
        summary = generator(f"Summarize the following: {combined}", do_sample=False)
        summaries.append(summary[0]['summary_text'])
        print(f"Overall summary: {summary[0]['summary_text']}")

    return summaries


def summarize(texts, one_by_one=True, model="facebook/bart-large-cnn"):
    summarizer = pipeline("summarization", model=model)

    summaries = []
    if one_by_one:
        print("Summarizing texts, one by one...")
        for (i, text) in enumerate(texts):
            word_count = len(text.split())
            print(f"Original text: {text}\n")
            if word_count > 10:
                summary = summarizer(text, max_length=max(1, int(0.8 * word_count)),
                                     min_length=max(10, int(0.1 * word_count)))
                summaries.append(summary[0]['summary_text'])
                print(f"Summary {i}: {summary[0]['summary_text']}\n")

    else:
        print("Summarizing overall texts...")
        combined = ", ".join(texts)
        # print("Combined summaries:", combined)
        word_count = len(combined.split())
        summary = summarizer(combined, max_length=max(1, int(0.8 * word_count)),
                             min_length=max(10, int(0.1 * word_count)))
        summaries.append(summary[0]['summary_text'])
        print(f"Overall summary: {summary[0]['summary_text']}")

    return summaries


def summarize_embedded_texts(embedded_texts):
    embedded_vectors = list(embedded_texts.values())

    # Determine the optimal number of clusters
    clusters = find_elbow_point(embedded_vectors)

    # Fit and take the closed text element of each cluster

    representing = closest_vectors_with_text(embedded_texts, clusters)

    texts = list(representing.keys())

    # Summarize each representing text from each cluster

    chunks_summarized = summarize(texts)

    # Summarize the summaries

    overall_summary = summarize(chunks_summarized, one_by_one=False)

    re = calc_rouge(overall_summary[0], ", ".join(chunks_summarized))

    print(re)

    return overall_summary


def summarize_url(url):
    text = extract_text_from_url(url)

    list_of_split_texts = split_text(text)

    embedded_texts = embed_texts(list_of_split_texts)

    summary = summarize_embedded_texts(embedded_texts)

    return summary


def summarize_pdf(file_path):
    text = extract_text_from_pdf_pypdf(file_path)

    list_of_split_texts = split_text(text)

    embedded_texts = embed_texts(list_of_split_texts)

    summary = summarize_embedded_texts(embedded_texts)

    return summary

    # Step 1: Summarize
    # return summarize_long_text(raw_text)


# store in the Qdrant vector database for further retrieval and application

def calc_rouge(summary, reference):
    scorer = RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(summary, reference)

    return scores


def test_calc_rouge():
    reference = """Saudi Arabia’s first alcohol store has opened in the diplomatic quarter of its capital Riyadh, accessible to non-Muslim diplomats. 

While it only affects a select group, it’s a big change for the highly conservative Muslim kingdom, where alcohol has been banned since 1952 after a Saudi prince murdered a British diplomat in a drunken rage. Drinking is also forbidden under Islam, and most of Saudi Arabia’s local population is religiously observant.

That hasn’t stopped alcohol from flowing into the kingdom over the years — it just happened behind closed doors.

Foreign embassies are able to import alcohol under specified agreements with the Saudi government, while some have snuck booze into the kingdom in secure “diplomatic pouches” that can’t be inspected.

From there, bottles are often sold on the black market at huge markups, according to expat and local residents of the country. All those who spoke to CNBC did so on condition of anonymity due to the sensitivity of the topic.

“Everyone knows which embassies sell booze … some of them have made a whole side business out of it, selling on the black market at four, five, even ten times the normal price. It’s gotten ridiculous. The government had to do something,” one Saudi investor based between Dubai in the United Arab Emirates and Riyadh told CNBC. 

A one-liter bottle of vodka, for instance, typically costs between $500 and $600 on the black market, sources said, while they described a single bottle of Johnnie Walker Blue Label going for between $1,000 and $2,000. At-home booze making has also taken place in the kingdom for decades, according to expats who’ve previously lived there. """

    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    summary = summarizer(reference)

    re = calc_rouge(summary[0]['summary_text'], reference)

    print(f"Summary:\n{summary[0]['summary_text']}\n")
    print(f"Rouge score: {re}")
