import os
import re
import time
import logging
from typing import List, Optional

from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi # gets subtitles (transcripts) from YouTube videos.
from langchain_text_splitters import RecursiveCharacterTextSplitter # breaks a long transcript into smaller “chunks” and then these chunks will be converted into embedding vectors.
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings, # this model will be used for the text generation
    ChatGoogleGenerativeAI, # this model will be used to create the embedding vectors for each chunk.
)
# ChatGoogleGenerativeAI & GoogleGenerativeAIEmbeddings ==> connect to Google Gemini models: ChatGoogleGenerativeAI → for chat/text generation, GoogleGenerativeAIEmbeddings → for creating vector embeddings (used in RAG).
from langchain_chroma import Chroma # chroma is a vector database that will store the chunks and their embedding vectors. so each chunk and it's corresponding embedding vector will be stored inside the vector database which is the chromadb
from langchain_core.prompts import ChatPromptTemplate

import streamlit as st

load_dotenv() # This reads your .env file (where your API key is stored) and makes it available in the program.

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VidGeniusAgent")
# This sets up a logger that prints information while the code runs — for example, when the transcript is fetched or if there’s an error.


# ---------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------
def extract_video_id(url: str) -> Optional[str]:
    """Extracts the YouTube video ID from any valid YouTube URL."""
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", url)
    if not match:
        st.error("❌ Invalid YouTube URL.")
        return None
    return match.group(1)
# this function takes the url of the youtube video as input and extracts the unique video ID from it using regular expressions. This ID is needed to fetch the transcript of the video.
# here we will get the id of the youtube video from the url provided by the user.


# ---------------------------------------------------------------------
# Main Class
# ---------------------------------------------------------------------
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
        # This is the constructor — it runs automatically when you create an object from the class.
        # self → refers to the current instance of the class.
        # The rest are parameters with default values:
            # language: which language the model should use (default = English "en").
            # model_name: the name of the Google Gemini model you want to use.
            # embedding_model: the model used to convert text into vector embeddings.
            # temperature: controls how creative the responses are.
                # Low (0.2) → more focused and consistent answers.
                # High (like 0.8) → more random or creative answers.
        api_key = os.getenv("GOOGLE_API_KEY")
        # This reads the API key from your environment variables.
        # It looks for a variable named GOOGLE_API_KEY inside your .env file (loaded earlier).
        # os.getenv() → safely retrieves environment variable values.
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in environment variables.")
        # If api_key wasn’t found (it’s empty or missing), it raises an error immediately.
        # This stops the program and shows a clear message that the key is missing.
        # ✅ Good practice for debugging configuration issues.
        self.language = language # Stores the chosen language in the object, so you can use it later in other methods.
        self.llm = ChatGoogleGenerativeAI(
            model=model_name, 
            temperature=temperature,
            google_api_key=api_key
        )
        # Creates a Google Gemini model object for chat or text generation.
        # ChatGoogleGenerativeAI is the class that connects to Gemini.
        # It uses:
            # model_name → which Gemini model to use.
            # temperature → how creative or random the responses should be.
            # google_api_key → the API key to authenticate with Google Cloud.
        self.embedding_model = embedding_model # Stores the embedding model name (used later to generate embeddings for RAG).
        self.vector_store = None
        # Initializes an empty variable for storing the vector database (e.g., Chroma or FAISS).
        # Later, it’ll hold your document embeddings.
        # Setting it to None just means “not created yet.”
        logger.info(f"VidGeniusAgent initialized for language: {language}")

        
    # -----------------------------------------------------------------
    # Transcript Handling
    # -----------------------------------------------------------------
    def get_transcript(self, youtube_url: str) -> Optional[str]:
        """Fetches transcript text for a YouTube video."""
        video_id = extract_video_id(youtube_url) # get the video id from the url from the previously defined function
        if not video_id:
            return None
        # If no ID was found (maybe the link was invalid), the function returns None immediately.
        # This means “stop the function, there’s nothing to do.”
        try:
            transcript = YouTubeTranscriptApi().fetch(video_id, languages=[self.language])
            # Starts a try block to handle errors safely.
            # Uses the YouTubeTranscriptApi library to fetch the transcript of the video.
            # It sends a request to YouTube’s subtitle system.
            # The parameter languages=[self.language] means: “Get the transcript in the language we set when initializing the class (usually English).”
            # The result is stored in transcript.
            # It’s usually a list of dictionaries, like:
            # [
            # {"text": "Hello everyone", "start": 0.5, "duration": 2.1},
            # {"text": "Welcome to the video", "start": 2.6, "duration": 3.0},
            # ...
            # ]
            logger.info(transcript)
            # Prints (or saves) the raw transcript data into the log file or console.
            # Helps for debugging — you can see what YouTube actually returned.
            full_text = " ".join([seg.text for seg in transcript])
            # Loops through each segment in the transcript list.
            # Extracts just the "text" part (the spoken words).
            # Joins all these pieces together into one big string, separated by spaces.
                # Example result:"Hello everyone Welcome to the video Hope you enjoy it"
            # Saves that full transcript as full_text.
            logger.info("Transcript fetched successfully.") # Logs a message confirming the transcript was fetched without problems.
            time.sleep(1)
            # Waits for 1 second before continuing.
            # This is usually done to:
                # Avoid sending too many requests to YouTube too quickly.
                # Make the system more stable and respectful of rate limits.
            return full_text # Returns the complete transcript text (a single large string).
        except Exception as e:
            st.error(f"⚠️ Failed to fetch transcript: {e}")
            logger.exception("Transcript fetch error.")
            return None
# In Summary: this function takes a YouTube URL, extracts the video ID, fetches the transcript in the desired language, processes it into a single string, and returns that string. If anything goes wrong (invalid URL, no transcript available, network issues), it handles the error gracefully and informs the user.
        
    
    # -----------------------------------------------------------------
    # Translation
    # -----------------------------------------------------------------
    def translate_transcript(self, transcript: str) -> str:
        """Translate transcript to English using Gemini."""
        prompt = ChatPromptTemplate.from_template("""
        You are an expert translator. Translate the following text into English
        with complete accuracy, keeping tone and style intact.
        Text:
        {transcript}
        """)
        # This line creates a chat prompt template that defines how we talk to the LLM.
        # ChatPromptTemplate.from_template() is a LangChain method used to build prompts with variables.
        # Inside the template, {transcript} is a placeholder that will later be replaced by the actual text we want to translate.
        # The prompt tells Gemini:
            # You are a translator.
            # Translate the given text into English accurately while keeping tone and style.
        # So it’s like telling the model: “Hey, here’s some text in another language — translate it to English faithfully.”
        try:
            chain = prompt | self.llm
            # The | operator pipes the prompt into the LLM model (self.llm).
            # self.llm refers to the Gemini model instance (set earlier in the class).
            # This creates a LangChain chain, meaning:
                # 1.Take the input.
                # 2.Format it using the prompt.
                # 3.Send it to Gemini for processing.
            # Basically:“When I give you a transcript, use this prompt and send it to Gemini for translation.”
            response = chain.invoke({"transcript": transcript})
            # This runs the translation process.
            # It replaces {transcript} in the prompt with the actual value passed to the function.
            # Then it sends the full prompt to Gemini.
            # The output (the translated text) is stored in response.
            logger.info("Transcript translated to English.")
            # This logs an info message saying translation worked.
            # It helps in debugging or tracking what’s happening behind the scenes.
            return response.content
        # response.content contains the translated English text returned by Gemini.
        # The function returns that translated string.
        except Exception as e:
            st.error(f"❌ Translation failed: {e}")
            logger.exception("Translation error.")
            return transcript
        
    
     # -----------------------------------------------------------------
    # Notes & Topics
    # -----------------------------------------------------------------
    def get_important_topics(self, transcript: str) -> str:
        """Extract the 5 most important topics from the transcript."""
        prompt = ChatPromptTemplate.from_template("""
        Extract exactly 5 key topics from the following transcript.
        Format as a numbered list.
        Transcript:
        {transcript}
        """)
        # This creates a prompt template for Gemini.
        # It tells the model:
            # 1.Extract exactly 5 topics (not 4, not 6).
            # 2.Format them as a numbered list (like 1., 2., 3...).
            # 3.The actual text to analyze is given as {transcript} — a placeholder that will later be replaced by real text.
        # Example of what the model would receive:Extract exactly 5 key topics from the following transcript.Format as a numbered list.Transcript:Today we discussed climate change, renewable energy, electric cars, and carbon emissions.
        chain = prompt | self.llm
        # The | (pipe) operator connects:
            # The prompt template (what to do)
            # With the LLM model (self.llm, e.g., Gemini)
        # This creates a LangChain chain that knows how to take a transcript and send it to Gemini with the correct instructions.
        try:
            response = chain.invoke({"transcript": transcript})
            # chain.invoke() runs the process.
            # It fills {transcript} in the prompt with the actual transcript text.
            # It sends that final prompt to Gemini for processing.
            # The model’s response (the list of 5 topics) is stored in response.
            return response.content # response.content contains Gemini’s output — the 5 topics.
        except Exception as e:
            st.error(f"⚠️ Topic extraction failed: {e}")
            logger.exception("Topic extraction error.")
            return "Error extracting topics."
# This function is only for identifying what the 5 key topics are in the transcript.
# It doesn’t summarize or write notes — it just analyzes the text and lists the main ideas (for example, the themes or subjects of discussion).

    def generate_notes(self, transcript: str) -> str:
        """Generate structured, concise notes."""
        prompt = ChatPromptTemplate.from_template("""
        You are an AI note-taker. Create bullet-point notes grouped under
        subheadings from the following transcript.
        Transcript:
        {transcript}
        """)
        # This builds the instruction prompt for Gemini.
        # It tells Gemini:
            # Act like a professional note-taker.
            # Write bullet points (• or -).
            # Group related points under subheadings.
        # The {transcript} variable will later be replaced with the actual text.
        # Example of what Gemini might output:
        # 🔹 Introduction
        # - The meeting began with a discussion on company growth.

        # 🔹 Financial Updates
        # - Revenue increased by 15%.
        # - Marketing costs decreased.

        # 🔹 Future Plans
        # - Expand to international markets.

        chain = prompt | self.llm # chain = prompt | self.llm
        try:
            response = chain.invoke({"transcript": transcript}) 
            # Replaces {transcript} in the prompt with the actual text.
            # Sends it to Gemini.
            # Stores the generated notes in response
            return response.content # Returns the final notes generated by Gemini.
        except Exception as e:
            st.error(f"⚠️ Note generation failed: {e}")
            logger.exception("Note generation error.")
            return "Error generating notes."
        # If anything fails (like connection error):
            # Shows an error in the Streamlit app.
            # Logs the issue.
            # Returns a fallback message.

# This function takes the entire transcript (the full text of the video) and asks Gemini to summarize it into organized notes.
    
     # -----------------------------------------------------------------
    # RAG Pipeline
    # -----------------------------------------------------------------
    def create_vector_store(self, transcript: str, chunk_size: int = 8000, overlap: int = 500):
        # chunk_size: int = 8000 ==> default maximum size of each chunk in characters (not tokens). This means each chunk will be up to ~8000 characters long unless the splitter breaks earlier (see below).
        # overlap: int = 500 ==> default number of characters to overlap between adjacent chunks. Overlap preserves context across chunk boundaries.
        """Split transcript into chunks and store embeddings."""
        try:
            splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap) 
            # This creates a text splitter that cuts the long transcript into smaller pieces (chunks).
                # Each chunk has up to 8000 characters.
                # overlap=500 means each chunk overlaps 500 characters with the next one to keep context.
            docs = splitter.create_documents([transcript]) # 👉 This actually splits the transcript into those smaller chunks and stores them as documents (a list of text pieces).
            embeddings = GoogleGenerativeAIEmbeddings(model=self.embedding_model, transport="grpc") # 👉 This loads Google’s embedding model (for example "models/gemini-embedding-001") to convert each text chunk into numbers (vectors) that represent meaning.
            self.vector_store = Chroma.from_documents(docs, embeddings)
            # 👉 This creates a Chroma vector database from all the document chunks and their embeddings.
            # Later, this database helps the AI find which chunks are most relevant to a user’s question.
            logger.info("Vector store created successfully.")
            return self.vector_store
        # 👉 Logs that everything worked and returns the created vector store.
        except Exception as e:
            st.error(f"⚠️ Vector store creation failed: {e}")
            logger.exception("Vector store error.")
            return None
        # 👉 If something goes wrong (like API key missing or bad connection), it shows an error in Streamlit and logs the problem.
        

    def rag_answer(self, question: str) -> str:
        # Defines a function called rag_answer that takes one input question (a string).
        # It returns a string (the answer).
        """Answer user query based on the stored context."""
        if not self.vector_store:
            st.warning("⚠️ No vector store available. Please process the video first.")
            return ""
        # Checks if the vector store doesn’t exist (maybe the transcript wasn’t processed yet).
        # If it doesn’t exist, it shows a warning message and returns an empty string.

        results = self.vector_store.similarity_search(question, k=4)
        # Searches inside the vector store for the 4 most similar chunks to the user’s question using embeddings.
        # These chunks are the most relevant parts of the transcript related to the question.
        # so this means that we will convert the user question into an embedding vector and then compare it(the embedding vector of the user question) to all the embedding vectors of all the chunks in the vector database to find the top 4 chunks that are closest in meaning to the question.
        # the retieved 4 chunks are called the context and these retrieved 4 chunks are the most similar chunks for the user question.
        # these 4 retrieved chunks will be used to answer on the user question.
        context_text = "\n".join([r.page_content for r in results])
        # Joins the text content of those 4 similar chunks into one big string separated by new lines.
        # This becomes the context that will be given to the model.

        prompt = ChatPromptTemplate.from_template("""
        You are a polite assistant answering based only on the provided context.
        If the answer is not in the context, say you don't know.
        Context:
        {context}
        Question:
        {question}
        Answer:
        """)
        # Creates a prompt template (the instructions for Gemini).
        # It tells the model:
            # Be polite.
            # Use only the given context.
            # If the context doesn’t have the answer, say “I don’t know.”
        # {context} and {question} will be replaced later with real values.
        # so we are telling the LLM model or the Gemini Model to answer on the user question based only on the provided context (the 4 retrieved chunks from the vector database) and if the model don't know the answer say I Don't Know.
        chain = prompt | self.llm
        # Combines the prompt with the language model (LLM) (e.g., Gemini).
        # This means: “Send this formatted prompt to Gemini for generating a response.”
        try:
            response = chain.invoke({"context": context_text, "question": question})
            # Replaces {context} and {question} in the prompt with the real text.
            # Sends the final prompt to Gemini.
            # response contains the model’s generated answer.
            return response.content # Returns the text of the model’s answer to the user.
        except Exception as e:
            st.error(f"⚠️ RAG query failed: {e}")
            logger.exception("RAG answer error.")
            return "Error generating answer."
# If something goes wrong (e.g., network error), it shows an error message and logs the problem.