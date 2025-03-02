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
from openai import OpenAI

load_dotenv()

# Azure OpenAI credentials
azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
azure_deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

# Set Azure OpenAI API parameters
# openai.api_type = "azure"
# openai.api_base = "https://ragstart.openai.azure.com/"
# openai.api_version = "2022-12-01"
# openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")


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
    # print(query)
    # print("\n")


# Use the chatbot
# Document is now stored as embeddings in Pinecone
# When sending questions to the LLM, add relevant knowledge from Pinecone index to ensure
# that the LLM returns an accurate response

retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
retriever = docsearch.as_retriever()

llm = ChatOpenAI(
    openai_api_key=azure_api_key,
    openai_api_base=azure_endpoint,
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

    # pass


    # print("Index after upsert:")
    # print(pc.Index(index_name).describe_index_stats())
    # print("\n")
    # time.sleep(2)

    # answer1_with_knowledge = retrieval_chain.invoke({"input": query1})

    # print("Answer with knowledge:\n\n", answer1_with_knowledge['answer'])
    # print("\nContext used:\n\n", answer1_with_knowledge['context'])
    # print("\n")
    # time.sleep(2)

    # answer2_with_knowledge = retrieval_chain.invoke({"input": query2})

    # print("\nAnswer with knowledge:\n\n", answer2_with_knowledge['answer'])
    # print("\nContext Used:\n\n", answer2_with_knowledge['context'])
    # print("\n")
    # time.sleep(2)


    from openai import AzureOpenAI

    # gets the API Key from environment variable AZURE_OPENAI_API_KEY
    client = AzureOpenAI(
        # https://learn.microsoft.com/azure/ai-services/openai/reference#rest-api-versioning
        api_version="2024-08-01-preview",
        # https://learn.microsoft.com/azure/cognitive-services/openai/how-to/create-resource?pivots=web-portal#create-a-resource
        azure_endpoint=azure_endpoint,
        api_key=azure_api_key
    )

    completion = client.chat.completions.create(
        model="gpt-35-turbo",  # e.g. gpt-35-instant
        messages=[
            {
                "role": "user",
                "content": "How do I output all files in a directory using Python?",
            },
        ],
    )
    print(completion.to_json())

    # pc.delete_index(index_name)


