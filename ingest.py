from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import SentenceTransformerEmbeddings

embeddings = SentenceTransformerEmbeddings(model_name='NeuML/pubmedbert-base-embeddings')

print(embeddings)

loader = DirectoryLoader("Data/",glob="**/*.pdf",show_progress=True,loader_cls=PyPDFLoader)

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

texts = text_splitter.split_documents(documents)

# Qdrant vectorDB link
url = "http://localhost:6333"

qdrant = Qdrant.from_documents(
    texts,
    embeddings,
    url=url,
    prefer_grpc = False,
    collection_name = "vector_database"
)

print("Vector Database created!!!")