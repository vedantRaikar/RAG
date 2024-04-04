import os
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_openai.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma


os.environ["OPENAI_API_KEY"] = "enter your open ai api key"

loader = TextLoader("data.txt", encoding="UTF-8")

 # Replace with the correct path if needed

try:
 documents = loader.load()
except FileNotFoundError:
 print("Error: data.txt file not found. Please check the file path.")
 exit(1)  # Exit the program with an error code

text_splitter = CharacterTextSplitter(chunk_size=25000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings()
db = Chroma.from_documents(texts, embeddings)
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})

qa = RetrievalQA.from_chain_type(
   llm=OpenAI(),
   chain_type="map_reduce",
   retriever=retriever,
   return_source_documents=True,
   verbose=True,
)

while True:
 query = input("Ask a question: ")

 try:
   answer = qa.run(query)
   print("Answer:", answer)
 except Exception as e:
   print("Error:", str(e))
