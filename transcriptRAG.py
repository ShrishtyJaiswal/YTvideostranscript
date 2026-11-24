from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

#indexing
#load the transcript in code
video_id = "Gfr50f6ZBvo" # only the ID, not full URL
try:
    # If you don’t care which language, this returns the “best” one
    api = YouTubeTranscriptApi()

    transcript_list = api.fetch(video_id, languages=["en"])

    # Flatten it to plain text
    transcript = " ".join(chunk.text for chunk in transcript_list)
    print(transcript)

except TranscriptsDisabled:
    print("No captions available for this video.")

#doing text spliter
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.create_documents([transcript])

#chunk sizes
#len(chunks) 
#print("Total chunks created:", len(chunks))
#print("100th number chunks:",chunks[100])

#convert chunks into vector and store in vector db
#embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

vector_store = Chroma.from_documents(chunks, embeddings)
#print(vector_store.get()["ids"])
#print("Print chunks id ", vector_store.get_by_ids)
documents = vector_store.get(
    ids=["b0fac111-93dc-47e7-b924-e119967db3ae"],
    include=["documents", "metadatas"]
)

#print("Document:", documents["documents"])
#print("Metadata:", documents["metadatas"])


#Form Retrival
#search_kwargs={"k": 4} 4 more similar search it provide
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

result = retriever.invoke('What is deepmind')
#print("retriever result: ",result)

#Augmentation
llm = ChatOllama(
    model="llama3.2",
    temperature=0
)

#prompt = query + context
prompt = PromptTemplate(
    template="""
      You are a helpful assistant.
      Answer ONLY from the provided transcript context.
      If the context is insufficient, just say you don't know.

      {context}
      Question: 
      {question}
    """,
    input_variables = ['context', 'question']
)

question          = "is the topic of nuclear fusion discussed in this video? if yes then what was discussed"
retrieved_docs    = retriever.invoke(question)

#print("retrieved docs", retrieved_docs)

context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
#print("context : ",context_text)

final_prompt = prompt.invoke({"context": context_text, "question": question})
#print("final prompt :",final_prompt)

#Generation

answer = llm.invoke(final_prompt)
#print("final answer",answer.content)

#create a parallel chain because earlier each step invoke indiviually

def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})
parallel_chain.invoke('who is Demis')
#print("parallel chain :",parallel_chain.invoke('who is Demis'))

#runnablepassthrouh chain
parser = StrOutputParser() #string output parser

main_chain = parallel_chain | prompt | llm | parser

output = main_chain.invoke("Can you summarize theis video")

