import os 
import json
import shutil
from typing import List, Optional
from model_definitions import Candidate, Education, Experience
from pydantic import BaseModel, Field, EmailStr, field_validator, root_validator

import torch
import openai
from llama_parse import LlamaParse
from llama_index.llms.ollama import Ollama
from transformers import AutoTokenizer, AutoModel
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, Document, StorageContext, load_index_from_storage

import streamlit as st
from dotenv import load_dotenv
load_dotenv()

# LLAMA_CLOUD_API_KEY = st.secrets['LLAMA_CLOUD_API_KEY']

# Class for analyzing the CV contents
class CvAnalyzer:
    def __init__(self, file_path, llm_option, embedding_option):
        """
        Initializes the CvAnalyzer with the given resume file path and model options.
        """
        self.file_path = file_path
        self.llm_option = llm_option    
        self.embedding_option = embedding_option
        self.resume_content = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._configure_settings()

    def _configure_settings(self):
        """
        Configures the large language model and embedding model based on the user-provided options.
        This ensures that the selected models are properly initialized and ready for use.
        """
        # Determine the device based on CUDA availability
        if torch.cuda.is_available():
            device = "cuda"
            print("CUDA is available. Using GPU.")
        else:
            device = "cpu"
            print(f"CUDA is not available. Using CPU.")

        # Configure the LLM
        if self.llm_option == "gpt-4o-mini":
            llm = Ollama(model="gpt-4o-mini", temperature=0, device=device)
        elif self.llm_option == "mistral:latest":
            llm = Ollama(model="mistral:latest", temperature=0, request_timeout=180.0, device=device)
        elif self.llm_option == "llama3.1":
            llm = Ollama(model="llama3.1", temperature=0, request_timeout=180.0, device=device)
        elif self.llm_option == "llama3.3:latest":
            llm = Ollama(model="llama3.3:latest", temperature=0, request_timeout=180.0, device=device)
        else:
            raise ValueError(f"Unsupported LLM option: {self.llm_option}")
        
        # Configure the embedding model
        if self.embedding_option == "BAAI/bge-small-en-v1.5":
            embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
        else:
            raise ValueError(f"Unsupported embedding model: {self.embedding_option}")

        # Set the models in settings 
        Settings.embed_model = embed_model
        Settings.llm = llm 
        self.llm = llm 
        self.embedding_model = embed_model


    def extract_candidate_data(self) -> Candidate:
        """
        Extracts candidate data from the resume.
        """
        print(f"Extracting CV data. LLM: {self.llm_option}")
        output_schema = Candidate.model_json_schema()
        parser = LlamaParse(
            result_type="markdown",
            parsing_instructions="Extract each section separately based on the document structure.",
            premium_mode=True,
            api_key=os.getenv("LLAMA_API_KEY"),
            verbose=True
        )
        file_extractor = {".pdf": parser}

        # Load resume
        documents = SimpleDirectoryReader(
            input_files=[self.file_path], file_extractor=file_extractor
        ).load_data()

        # Store the pre-extracted content
        self._resume_content = "\n".join([doc.text for doc in documents])
        prompt = f"""
            You are an expert in analyzing resumes. Use the following JSON schema to extract relevant information:
            ```json
            {output_schema}
            ```json
            Extract the information from the following document and provide a structured JSON response strictly adhering to the schema above. 
            Please remove any ```json ``` characters from the output. Do not make up any information. If a field cannot be extracted, mark it as 'n/a'.
            You should just return the JSON content.
            Document:
            ----------------
            {self._resume_content}
            ----------------
            """
        try:
            response = self.llm.complete(prompt)
            if not response or not response.text:
                raise ValueError("Failed to get a response from LLM.")

            cleaned_response = response.text.strip().replace('```json', '').replace('```', '').strip()
            # print(f"Cleaned data: \n{cleaned_response}")
            parsed_data = json.loads(cleaned_response)
            return Candidate.model_validate(parsed_data)
        except Exception as e:
            print(f"Error parsing response: {str(e)}")
            raise ValueError("Failed to extract insights. Please ensure the resume and query engine are properly configured.")
        
    def _get_embedding(self, texts: List[str], model: str) -> torch.Tensor:
        """
        Generates embeddings for a list of text inputs using the specified embedding model.
        This function is called by compute_skill_scores() function
        """
        if model.startswith('text-embeddings'):
            from openai import OpenAI
            client = OpenAI(api_key=openai.api_key)
            response = client.embeddings.create(input=texts, model=model)
            embeddings = [torch.tensor(item.embedding) for item in response.data]
        elif model == "BAAI/bge-small-en-v1.5":
            tokenizer = AutoTokenizer.from_pretrained(model)
            hf_model = AutoModel.from_pretrained(model).to(self.device)

            embeddings = []
            for text in texts:
                inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512).to(self.device)
                with torch.no_grad():
                    outputs = hf_model(**inputs)
                embeddings.append(outputs.last_hidden_state.mean(dim=1).squeeze())
        else:
            raise ValueError(f"Unsupported embedding model: {model}")
        return torch.stack(embeddings)

    def compute_skill_scores(self, skills: List[str]) -> dict:
        """
        Computes semantic similarity scores between skills and the resume content.
        """
        # Extract resume content and compute is embedding
        resume_content = self._extract_resume_content()

        # Compute embeddings for all skills at once
        skill_embeddings = self._get_embedding(skills, model=self.embedding_model.model_name)

        # Compute raw similarity scores and semantic frequency for each skill
        raw_scores = {}
        # frequency_scores = {}
        for skill, skill_embedding in zip(skills, skill_embeddings):
            # Compute semantic similarity with the entire resume
            similarity = self._cosine_similarity(
                self._get_embedding([resume_content], model=self.embedding_model.model_name)[0],
                skill_embedding
            )
            raw_scores[skill] = similarity
        return raw_scores
    

    def _extract_resume_content(self) -> str:
        """
        Called by compute_skill_scores(), this function extracts and returns the raw textual content of the resume.
        """
        if self._resume_content:
            return self._resume_content
        else:
            raise ValueError("Resume content not available. Ensure `extract_candidate_data` is called first.")
        

    def _cosine_similarity(self, vec1: torch.Tensor, vec2: torch.Tensor) -> str:
        """
        Called by compute_skill_scores(), calculates the cosine similarity between two vectors
        """
        vec1, vec2 = vec1.to(self.device), vec2.to(self.device)
        return (torch.dot(vec1, vec2) / (torch.norm(vec1) * torch.norm(vec2))).item()

    
    def create_or_load_job_index(self, json_file: str, index_folder: str = "job_index_storage", recreate: bool = False):
        """
        Creates a new job vector index from a JSON dataset or loads an existing index from storage
        Returns VectorStoreIndex object for quering jobs
        """
        if recreate and os.path.exists(index_folder):
            # Delete the existing job index storage 
            print(f"Deleting the existing job dataset: {index_folder}...")
            shutil.rmtree(index_folder)
        
        if not os.path.exists(index_folder):
            print(f"Creating new job vector index with {self.embedding_model.model_name} model...")
            with open(json_file, 'r') as f:
                job_data = json.load(f)

            # Convert job descriptions to Document objects by serializing all fields dynamically 
            documents = []
            for job in job_data['jobs']:
                job_text = "\n".join([f"{key.capitalize()}: {value}" for key, value in job.items()])
                documents.append(Document(text=job_text))
            
            # Create the vector index directly from documents 
            index = VectorStoreIndex.from_documents(documents, embed_model=self.embedding_model)
            return index 
        else:
            print(f"Loading existing job index from {index_folder}...")
            storage_context = StorageContext.from_defaults(persist_dir=index_folder)
            return load_index_from_storage(storage_context)
        

    def query_jobs(self, education, skills, experience, index, top_k=3):
        "Queries the job vector index to find the top-k matching jobs based on the provided profile."
        print(f"Fetching job suggestions.(LLM: {self.llm.model}, embed_model: {self.embedding_option})")
        query = f"Education: {', '.join(education)}; Skills: {', '.join(skills)}; Experience: {', '.join(experience)}"
        # Use retriever with appropriate model
        retriever = index.as_retriever(similarity_top_k=top_k)
        matches = retriever.retrieve(query)
        return matches
    