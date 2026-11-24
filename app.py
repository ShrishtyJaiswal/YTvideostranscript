from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import re
import streamlit as st
from urllib.parse import urlparse, parse_qs
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------
# 1. Streamlit Page Configuration
# ---------------------------------

st.set_page_config(
    page_title="YouTube AI Assistant",
    page_icon="🎥",
    layout="wide"
)

st.title("🎥 YouTube AI Assistant")

st.write(
    "Enter a YouTube video URL, load its transcript "
    "and ask questions about the video."
)

# ---------------------------------
# 2. Extract YouTube Video ID
# ---------------------------------

def extract_video_id(video_url):
    try:
        parsed_url = urlparse(video_url.strip())
        hostname = parsed_url.hostname
        if not hostname:
            return None
        hostname = hostname.lower()
        # Short URL: youtu.be/VIDEO_ID
        if hostname == "youtu.be":
            video_id = parsed_url.path.strip("/").split("/")[0]
        # youtube.com URLs
        elif hostname.endswith("youtube.com"):
            if parsed_url.path == "/watch":
                video_id = parse_qs(
                    parsed_url.query
                ).get("v", [None])[0]

            elif parsed_url.path.startswith(
                ("/embed/", "/shorts/", "/live/")
            ):
                video_id = parsed_url.path.split("/")[2]

            else:
                return None

        else:
            return None

        if video_id and re.fullmatch(
            r"[A-Za-z0-9_-]{11}", video_id
        ):
            return video_id

        return None

    except Exception:
        return None

# ---------------------------------
# 3. Fetch YouTube Transcript indexing
# ----
def fetch_transcript(video_id):

    api = YouTubeTranscriptApi()

    transcript_list = api.fetch(
        video_id,
        languages=["en"]
    )

    transcript = " ".join(
        chunk.text for chunk in transcript_list
    )

    if not transcript.strip():
        raise ValueError("Transcript is empty.")

    return transcript

# ---------------------------------
# 4. Create Embeddings and Vector DB
# ---------------------------------

@st.cache_resource
def get_embeddings():

    return HuggingFaceEmbeddings(
        model_name=(
            "sentence-transformers/"
            "paraphrase-multilingual-MiniLM-L12-v2"
        )
    )

def create_vector_store(transcript):

    # Split transcript into chunks

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.create_documents(
        [transcript]
    )

# Generate embeddings

    embeddings = get_embeddings()

    # Store chunks in Chroma

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vector_store, len(chunks)

# ---------------------------------
# 5. Initialize LLM
# ---------------------------------

@st.cache_resource
def get_llm():

    return ChatOllama(
        model="llama3.2",
        temperature=0
    )

# ---------------------------------
# 6. Create RAG Chain
# ---------------------------------

def create_rag_chain(vector_store):

    # Retrieve top 4 relevant chunks

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    # Prompt Template

    prompt = PromptTemplate(
        template="""
        You are a helpful YouTube video assistant.

        Answer ONLY using the provided transcript context.

        If the answer is not available in the transcript,
        say: "I don't know based on this video."

        Do not make up information.

        Context:
        {context}

        Question:
        {question}

        Answer:
        """,
        input_variables=["context", "question"]
    )

    # Format retrieved documents

    def format_docs(retrieved_docs):

        context_text = "\n\n".join(
            doc.page_content
            for doc in retrieved_docs
        )

        return context_text

    # Parallel chain

    parallel_chain = RunnableParallel({
        "context": (
            retriever
            | RunnableLambda(format_docs)
        ),
        "question": RunnablePassthrough()
    })

 # Final RAG chain

    llm = get_llm()

    parser = StrOutputParser()

    main_chain = (
        parallel_chain
        | prompt
        | llm
        | parser
    )

    return main_chain


# ---------------------------------
# 7. YouTube URL Input
# ---------------------------------

st.subheader("1. Enter YouTube Video URL")

video_url = st.text_input(
    "YouTube URL",
    placeholder="https://www.youtube.com/watch?v=Gfr50f6ZBvo"
)

if st.button("Load Video Transcript", type="primary"):

    if not video_url.strip():

        st.warning("Please enter a YouTube URL.")

    else:

        video_id = extract_video_id(video_url)

        if not video_id:

            st.error("Invalid YouTube URL.")

        else:

            try:

                with st.spinner(
                    "Fetching transcript and creating vector database..."
                ):

                    # Fetch transcript

                    transcript = fetch_transcript(video_id)

                    # Create vector database

                    vector_store, total_chunks = (
                        create_vector_store(transcript)
                    )

                    # Create RAG chain

                    rag_chain = create_rag_chain(
                        vector_store
                    )

                    # Store in Streamlit session

                    st.session_state.video_id = video_id
                    st.session_state.video_url = video_url
                    st.session_state.rag_chain = rag_chain
                    st.session_state.total_chunks = total_chunks

                st.success("Video transcript loaded successfully!")

                st.info(f"Video ID: {video_id}")

                st.info(
                    f"Total transcript chunks: {total_chunks}"
                )

            except TranscriptsDisabled:

                st.error(
                    "Captions are disabled for this video."
                )

            except NoTranscriptFound:

                st.error(
                    "No English transcript was found for this video."
                )

            except Exception as e:

                st.error(f"Error loading video: {str(e)}")


# ---------------------------------
# 8. User Defined Question
# ---------------------------------

if "rag_chain" in st.session_state:

    st.divider()

    st.subheader("2. Ask Questions About Your Video")

    question = st.text_area(
        "Enter your question",
        placeholder=(
            "Example: What is DeepMind? "
            "Explain the key concepts discussed in this video."
        ),
        height=100
    )

    if st.button("Get Answer", type="primary"):

        if not question.strip():

            st.warning("Please enter your question.")

        else:

            try:

                with st.spinner(
                    "Searching transcript and generating answer..."
                ):

                    answer = st.session_state.rag_chain.invoke(
                        question
                    )

                st.subheader("AI Generated Answer")

                st.markdown(answer)

            except Exception as e:

                st.error(
                    f"Error generating answer: {str(e)}"
                )
