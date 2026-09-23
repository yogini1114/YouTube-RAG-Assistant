import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv(override=True)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found in .env")
    st.stop()
    
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda




st.set_page_config(
    page_title = "YouTube RAG ",
    page_icon = "✨",
    layout="wide"
)

st.title("YouTube RAG Assistant")
st.write("Ask questions about any YouTube video.")

youtube_url = st.text_input("Enter the YouTube video URL")

question = st.text_input("Ask a question about the video")

if st.button("Process Video"):
  try:
    if "v=" in youtube_url:
      video_id = youtube_url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in youtube_url:
      video_id = youtube_url.split("youtu.be/")[1].split("?")[0]
    else:
      st.error("Invalid YouTube URL")
      st.stop()

    ytt_api = YouTubeTranscriptApi()
    transcript_list = ytt_api.fetch(video_id)
    transcript = " ".join(
        chunks.text for chunks in transcript_list
    )

    splitter = RecursiveCharacterTextSplitter(chunk_size= 1000, chunk_overlap=200)
    chunks = splitter.create_documents([transcript])
    chunks = [c for c in chunks if c.page_content.strip()]
    
    if not chunks:
        st.error("Invalid Transcript")
        st.stop()

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    
    vector_store = FAISS.from_documents(chunks, embeddings)

    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k":4})

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
    

    prompt = PromptTemplate(
        template="""
        You are a helpful assistant.
        Answer ONLY from the provided transcript context.
        If the content is insufficient, just say you don't know.

        Context: {context}
        Question: {question}
        """,
        input_variables=["context", "question"]
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)


    parallel_chain = RunnableParallel(
    context=retriever | RunnableLambda(format_docs),
    question=RunnablePassthrough()
    )
    parser = StrOutputParser()
    main_chain = parallel_chain | prompt | llm | parser
    
    st.session_state["main_chain"] = main_chain
    st.success("Video processed!")
        
  except TranscriptsDisabled:
    st.error("Transcript is disabled for this video")
  except Exception as e:
    st.error(f"Could not process video: {e}")

if st.button("Get Answer"):
    if "main_chain" not in st.session_state:
        st.warning("⚠️ First process the YouTube video.")
    elif not question.strip():
        st.warning("⚠️ Please type a question first.")
    else:
        try:
            with st.spinner("Thinking..."):
                answer = st.session_state["main_chain"].invoke(question.strip())
            st.subheader("Answer")
            st.write(answer)
        except Exception as e:
            st.error(f"Error while answering: {e}")