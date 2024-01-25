# This is a sample Python script.
import pypdf
from gpt4all import Embed4All
from qdrant_client import QdrantClient, models
from sklearn.cluster import KMeans
import numpy as np
from scipy.spatial import distance
from transformers import pipeline

from langchain.text_splitter import RecursiveCharacterTextSplitter
import re


def find_elbow_point(embedded_vectors, min_k=3, max_k=15):
    normalized_vectors = embedded_vectors / np.linalg.norm(embedded_vectors, axis=1, keepdims=True)
    wcss = []

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


def embed_document(file_path):
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

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200, length_function=len,
                                              is_separator_regex=False, separators=[".", ",", ";"])

    filtered_texts = splitter.create_documents([filtered_text])

    number_of_vectors = len(filtered_texts)

    print(f"There are total {number_of_vectors} vectors")

    embeder = Embed4All()

    embedding_map = {}

    for i in range(number_of_vectors):
        text = filtered_texts[i].page_content
        embedding = embeder.embed(text)
        embedding_map[text] = embedding

    return embedding_map


def summarize(texts, one_by_one=True):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

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


def summarize_doc(file_path):
    # Step 1: Embed
    embedding = embed_document(file_path)
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


#
# # store in the Qdrant vector database for further retrieval and application
# qdrant_client = QdrantClient(
#     url="https://9240ae44-20ec-436e-b4fd-8ce951d197a3.us-east4-0.gcp.cloud.qdrant.io:6333",
#     api_key="4sSgvGUXsjevfHbwjKGU8q2Jrhzs6GneX2y11VmjEQXEY3h3HMVf9Q",
# )
#
# qdrant_client.recreate_collection(
#     collection_name="text_to_summarize",
#     vectors_config=models.VectorParams(
#         size=number_of_dimensions, distance=models.Distance.COSINE
#
#     )
# )
#
# ids = [i + 1 for i in range(number_of_vectors)]
#
# qdrant_client.upsert(collection_name="text_to_summarize",
#                      points=models.Batch(
#                          ids=ids,
#                          vectors=outputs
#                      ))
# # store in the Qdrant vector database for further retrieval and application
#
# # implement K-mean clustering and defining the optimal number of clusters


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # testing()

    re = summarize_doc("docs/doc.pdf")

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
