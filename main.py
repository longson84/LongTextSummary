# # store in the Qdrant vector database for further retrieval and application
# qdrant_client = QdrantClient(
#     url="https://9240ae44-20ec-436e-b4fd-8ce951d197a3.us-east4-0.gcp.cloud.qdrant.io:6333",
#     api_key="",
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

import trafilatura

if __name__ == '__main__':
    url = "https://www.slingacademy.com/article/python-ways-to-extract-plain-text-from-a-webpage/"

    text = trafilatura.fetch_url(url)

    print(trafilatura.extract(text))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
