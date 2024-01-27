import pypdf
import trafilatura
from gpt4all import Embed4All
from qdrant_client import QdrantClient, models
from sklearn.cluster import KMeans
import numpy as np
from scipy.spatial import distance
from transformers import pipeline

from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import config

# Functions to extract the text
def extract_text_from_url(url):
    text = trafilatura.fetch_url(url)

    text_to_summarize = trafilatura.extract(text)

    return text_to_summarize


def extract_text_from_pdf(file_path):
    with open(file_path, "rb") as f:
        reader = pypdf.PdfReader(f)
        filtered_text = ''
        number_of_pages = len(reader.pages)
        print(f"There are total {number_of_pages} pages in the original document")

        for i in range(number_of_pages):
            page = reader.pages[i]
            text = page.extract_text()
            text = re.sub(r'\bSources:.*?\.', '', text, flags=re.IGNORECASE)
            filtered_text += text + "\n"

    return filtered_text


# Functions to extract the text

# Function to split and embed the text
def embed_text(raw_test):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200, length_function=len,
                                              is_separator_regex=False, separators=[".", ",", ";"])

    docs = splitter.create_documents([raw_test])

    number_of_vectors = len(docs)

    print(f"There are total {number_of_vectors} vectors")

    embeder = Embed4All()

    embedding_map = {}

    for i in range(number_of_vectors):
        text = docs[i].page_content
        embedding = embeder.embed(text)
        embedding_map[text] = embedding

    return embedding_map


def save_to_vector_db(embedding_map):

    vectors = list(embedding_map.values())
    payloads = list(embedding_map.keys())

    number_of_dimensions = len(vectors[0])
    number_of_vectors = len(vectors)

    qdrant_client = QdrantClient(
        url="https://9240ae44-20ec-436e-b4fd-8ce951d197a3.us-east4-0.gcp.cloud.qdrant.io:6333",
        api_key=config.QDRANT_API_KEY,
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
                             payloads=payloads,
                             vectors=vectors,
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


def closest_vectors_with_text(vector_dict, num_clusters):
    labels = list(vector_dict.keys())
    embedded_vectors = np.array(list(vector_dict.values()))

    kmeans = KMeans(n_clusters=num_clusters, init="k-means++", max_iter=100, n_init=10, random_state=0)
    kmeans.fit(embedded_vectors)

    cluster_means = [np.mean(embedded_vectors[kmeans.labels_ == i], axis=0) for i in range(num_clusters)]

    closest = {}

    for mean_vector in cluster_means:
        closest_index = min(range(len(embedded_vectors)),
                            key=lambda v: distance.euclidean(embedded_vectors[v], mean_vector))
        closet_vector = embedded_vectors[closest_index]
        closet_key = labels[closest_index]
        closest[closet_key] = closet_vector

    return closest


def summarize(texts, one_by_one=True, model="facebook/bart-large-cnn"):
    summarizer = pipeline("summarization", model=model)

    summaries = []
    if one_by_one:
        print("Summarizing texts, one by one...")
        for (i, text) in enumerate(texts):
            word_count = len(text.split())
            print(f"Original text: {text}\n")
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


def summarize_long_text(text):
    # Step 1: Embed
    embedding = embed_text(text)
    embedded_vectors = list(embedding.values())

    # Step 2: Determine the optimal number of clusters
    clusters = find_elbow_point(embedded_vectors)

    # Step 3: Fit and take the closed text element of each cluster

    representing = closest_vectors_with_text(embedding, clusters)

    texts = list(representing.keys())

    # Step 4: Summarize each representing text from each cluster

    chunks_summarized = summarize(texts)

    # representing_summarized = [chunk["summary_text"] for chunk in chunks_summarized]

    # Step 5: Summarize the summaries

    overall_summary = summarize(chunks_summarized, one_by_one=False)

    return overall_summary


def summarize_url(url):
    text_to_summarize = extract_text_from_url(url)

    summary = summarize_long_text(text_to_summarize)

    return summary


def summarize_pdf(file_path):
    # Step 0: Extract text

    raw_text = extract_text_from_pdf(file_path)

    # Step 1: Summarize
    return summarize_long_text(raw_text)


# store in the Qdrant vector database for further retrieval and application
