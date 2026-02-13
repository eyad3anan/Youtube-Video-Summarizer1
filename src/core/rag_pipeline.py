import os
import re
import time
import logging
from typing import List, Optional

from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi 
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings, 
    ChatGoogleGenerativeAI, 
)

from langchain_chroma import Chroma 
from langchain_core.prompts import ChatPromptTemplate

import streamlit as st

load_dotenv() 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VidGeniusAgent")


def extract_video_id(url: str) -> Optional[str]:
    """Extracts the YouTube video ID from any valid YouTube URL."""
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    if not match:
        st.error("❌ Invalid YouTube URL.")
        return None
    return match.group(1)


class VidGeniusAgent:
    """
    Core agent for handling transcript extraction, translation,
    topic detection, note generation, and RAG-based QA.
    """

    def __init__(
        self,
        language: str = "en",
        model_name: str = "gemini-2.5-flash-lite",
        embedding_model: str = "models/gemini-embedding-001",
        temperature: float = 0.2
    ):
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables.")
        
        self.language = language 
        self.llm = ChatGoogleGenerativeAI(
            model=model_name, 
            temperature=temperature,
            google_api_key=api_key
        )
        self.embedding_model = embedding_model 
        self.vector_store = None
        logger.info(f"VidGeniusAgent initialized for language: {language}")

        

    def get_transcript(self, youtube_url: str) -> Optional[str]:
        """Fetches transcript text for a YouTube video."""
        video_id = extract_video_id(youtube_url) 
        if not video_id:
            return None
        try:
            transcript = YouTubeTranscriptApi().fetch(video_id, languages=[self.language])
            logger.info(transcript)
            full_text = " ".join([seg.text for seg in transcript])
            logger.info("Transcript fetched successfully.") 
            time.sleep(1)
            return full_text 
        except Exception as e:
            st.error(f"⚠️ Failed to fetch transcript: {e}")
            logger.exception("Transcript fetch error.")
            return None
        
    

    def translate_transcript(self, transcript: str) -> str:
        """Translate transcript to English using Gemini."""
        prompt = ChatPromptTemplate.from_template("""
        You are an expert translator. Translate the following text into English
        with complete accuracy, keeping tone and style intact.
        Text:
        {transcript}
        """)
        try:
            chain = prompt | self.llm
            response = chain.invoke({"transcript": transcript})
            logger.info("Transcript translated to English.")
            return response.content
        except Exception as e:
            st.error(f"❌ Translation failed: {e}")
            logger.exception("Translation error.")
            return transcript
        
    def get_important_topics(self, transcript: str) -> str:
        """Extract the 5 most important topics from the transcript."""
        prompt = ChatPromptTemplate.from_template("""
        Extract exactly 5 key topics from the following transcript.
        Format as a numbered list.
        Transcript:
        {transcript}
        """)
        chain = prompt | self.llm
        try:
            response = chain.invoke({"transcript": transcript})
            return response.content 
        except Exception as e:
            st.error(f"⚠️ Topic extraction failed: {e}")
            logger.exception("Topic extraction error.")
            return "Error extracting topics."

    def generate_notes(self, transcript: str) -> str:
        """Generate structured, concise notes."""
        prompt = ChatPromptTemplate.from_template("""
        You are an AI note-taker. Create bullet-point notes grouped under
        subheadings from the following transcript.
        Transcript:
        {transcript}
        """)
        chain = prompt | self.llm # chain = prompt | self.llm
        try:
            response = chain.invoke({"transcript": transcript}) 
            return response.content 
        except Exception as e:
            st.error(f"⚠️ Note generation failed: {e}")
            logger.exception("Note generation error.")
            return "Error generating notes."
        
    def create_vector_store(self, transcript: str, chunk_size: int = 8000, overlap: int = 500):
        """Split transcript into chunks and store embeddings."""
        try:
            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap) 
            docs = splitter.create_documents([transcript]) 
            embeddings = GoogleGenerativeAIEmbeddings(model=self.embedding_model, transport="grpc") 
            self.vector_store = Chroma.from_documents(docs, embeddings)
            logger.info("Vector store created successfully.")
            return self.vector_store
        except Exception as e:
            st.error(f"⚠️ Vector store creation failed: {e}")
            logger.exception("Vector store error.")
            return None
        

    def rag_answer(self, question: str) -> str:
        """Answer user query based on the stored context."""
        if not self.vector_store:
            st.warning("⚠️ No vector store available. Please process the video first.")
            return ""

        results = self.vector_store.similarity_search(question, k=4)
        context_text = "\n".join([r.page_content for r in results])
        

        prompt = ChatPromptTemplate.from_template("""
        You are a polite assistant answering based only on the provided context.
        If the answer is not in the context, say you don't know.
        Context:
        {context}
        Question:
        {question}
        Answer:
        """)
        
        chain = prompt | self.llm
        try:
            response = chain.invoke({"context": context_text, "question": question})
            return response.content 
        except Exception as e:
            st.error(f"⚠️ RAG query failed: {e}")
            logger.exception("RAG answer error.")
            return "Error generating answer."
