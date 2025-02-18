__import__('pysqlite3')
import pysqlite3
sys.modules['sqlite3'] = sys.modules["pysqlite3"]
import chromadb

from langchain_ollama import ChatOllama
from langchain import hub

llm = ChatOllama(
    model="deepseek-r1:8b",
    temperature=0,
)

from langchain_core.messages import AIMessage

messages = [
    (
        "system",
        "You are a helpful assistant that translates English to French. Translate the user sentence.",
    ),
    ("human", "I love programming."),
]
ai_msg = llm.invoke(messages)
ai_msg


from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model="deepseek-r1:8b",
)
embeddings

text = """
In the lush canopy of a tropical rainforest, two mischievous monkeys, Coco and Mango, swung from branch to branch, their playful antics echoing through the trees. They were inseparable companions, sharing everything from juicy fruits to secret hideouts high above the forest floor. One day, while exploring a new part of the forest, Coco stumbled upon a beautiful orchid hidden among the foliage. Entranced by its delicate petals, Coco plucked it and presented it to Mango with a wide grin. Overwhelmed by Coco's gesture of friendship, Mango hugged Coco tightly, cherishing the bond they shared. From that day on, Coco and Mango ventured through the forest together, their friendship growing stronger with each passing adventure. As they watched the sun dip below the horizon, casting a golden glow over the treetops, they knew that no matter what challenges lay ahead, they would always have each other, and their hearts brimmed with joy.
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
chunks = text_splitter.split_text(text)

from langchain.vectorstores import Chroma
from langchain.chains import create_retrieval_chain
vector_store = Chroma.from_texts(chunks, embeddings)

retriever = vector_store.as_retriever()
chain = create_retrieval_chain(combine_docs_chain=llm,retriever=retriever)
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

from langchain.chains.combine_documents import create_stuff_documents_chain
combine_docs_chain = create_stuff_documents_chain(
    llm, retrieval_qa_chat_prompt
)
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)    
response = retrieval_chain.invoke({"input": "Can you write a short paragraph that continues the story of Coco and Mango?"})
print(response['answer'])