from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_pinecone import PineconeEmbeddings
import os
from pinecone import Pinecone, ServerlessSpec
import time
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore


from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub

load_dotenv()

# Chunk the document based on h2 headers
with open('wd.md', 'r') as file:
    markdown_document = file.read()

headers_to_split_on = [
    ('##', "Header 2")
]

md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers_to_split_on, strip_headers=False
)
md_header_splits = md_splitter.split_text(markdown_document)


model_name = 'multilingual-e5-large'
embeddings = PineconeEmbeddings(
    model=model_name,
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
)

# Create a Pinecone index to store the document in
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Define the cloud region and index name
cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
region = os.environ.get('PINECONE_REGION') or 'us-east-1'
spec = ServerlessSpec(cloud=cloud, region=region)

index_name = "rag-getting-started"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=embeddings.dimension,
        metric="cosine",
        spec=spec
    )
    # Wait for the index to be ready
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)


# Embed and upsert each chunk as a distinct record in a namespace called wondervector5000
namespace = "wondervector5000"

docsearch = PineconeVectorStore.from_documents(
    documents=md_header_splits,
    index_name=index_name,
    embedding=embeddings,
    namespace=namespace
)

time.sleep(5)


# Use Pinecone's list and query operations to look at one of the records
index = pc.Index(index_name)
namespace = "wondervector5000"

for ids in index.list(namespace=namespace):
    query = index.query(
        id=ids[0], 
        namespace=namespace, 
        top_k=1,
        include_values=True,
        include_metadata=True
    )
    print(query)
    print("\n")


# Use the chatbot
# Document is now stored as embeddings in Pinecone
# When sending questions to the LLM, add relevant knowledge from Pinecone index to ensure
# that the LLM returns an accurate response

retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
retriever = docsearch.as_retriever()

llm = ChatOpenAI(
    openai_api_key=os.getenv('OPENAI_API_KEY'),
    model_name='gpt-4o-mini',
    temperature=0.0
)

combine_docs_chain = create_stuff_documents_chain(
    llm, retrieval_qa_chat_prompt
)

retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)


# Define a few question about the WonderVector5000
query1 = "What are the first 3 steps for getting started with the WonderVector5000?"
query2 = "The Neural Fandango Synchronizer is giving me a headache. What do I do?"





if __name__ == "__main__":
    # print(md_header_splits)
    # print('/n')


    print("Index after upsert:")
    print(pc.Index(index_name).describe_index_stats())
    print("\n")
    time.sleep(2)