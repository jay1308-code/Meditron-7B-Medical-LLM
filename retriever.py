from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import SentenceTransformerEmbeddings
from qdrant_client import QdrantClient


embeddings = SentenceTransformerEmbeddings(model_name='NeuML/pubmedbert-base-embeddings')

url = "http://localhost:6333/dashboard"

client = QdrantClient(
    url = url,
    prefer_grpc = False
)
print(client)

db = Qdrant(
    client = client,
    embeddings=embeddings,
    collection_name="vector_database"
)

print(db)
print("--------------------------------")

query = "what are the Common side effects of systemic therapeutic agents?"

docs = db.similarity_search_with_score(query=query,k=2)

for i in docs:
    docs,score = i
    print({"Score":score,"Content":docs.page_content,"metadata":docs.metadata})