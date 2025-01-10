import os 
import torch
import random 
import tempfile
import streamlit as st

from pydantic import BaseModel, Field, ConfigDict
from llama_index.llms.ollama import Ollama
from llama_index.core.prompts import PromptTemplate
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.query_engine import CustomQueryEngine

from CV_analyzer import CvAnalyzer

class RAGStringQueryEngine(BaseModel):
    """
    Custom Query Engine for RAG (fetching matching job recommendations).
    """
    retriever: BaseRetriever
    llm: Ollama
    qa_prompt: PromptTemplate

    # Allow arbitrary types
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    def custom_query(self, candidate_details: str, retrieved_jobs: str):
        query_str = self.qa_prompt.format(
            query_str=candidate_details, context_str=retrieved_jobs
        )

        if isinstance(self.llm, Ollama):
            # Ollama specific query handling 
            response = self.llm.complete(query_str)
        else:
            raise ValueError("Unsupported LLM type: Please use Ollama.")

        return str(response)

def main():
    st.set_page_config(page_title="CV Analyzer and Job Recommender", page_icon="🔍")
    st.title("CV Analyzer and Job Recommender")

    # Sidebar for model selection
    with st.sidebar:
        st.header("Model Selection")
        llm_option = st.selectbox(
            "Select an LLM:",
            options=["gpt-4o-mini", "llama3.1","mistral:latest", "llama3.3:latest"]
        )
        embedding_option = st.selectbox(
            "Select an embedding model:",
            options=["BAAI/bge-small-en-v1.5"]
        )
        recreate_index = st.checkbox(f"Create new job embeddings")

    st.write("Upload a CV to extract key information.")
    uploaded_file = st.file_uploader("Select your CV (PDF)", type="pdf", help="Choose a PDF file up to 5MB")

    if uploaded_file is not None:
        if st.button("Analyze"):
            with st.spinner("Parsing CV... This may take a moment."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                        temp_file.write(uploaded_file.getvalue())
                        temp_file_path = temp_file.name
                    
                    # Initialize CvAnalyzer with selected models
                    analyzer = CvAnalyzer(temp_file_path, llm_option, embedding_option)

                    # Extract insights from the resume 
                    insights = analyzer.extract_candidate_data()

                    # Load or create job vector index 
                    job_index = analyzer.create_or_load_job_index(json_file="sample_jobs.json", index_folder="job_index_storage", recreate=recreate_index)

                    # Query jobs based on resume data
                    education = [edu.degree for edu in insights.education] if insights.education else []
                    skills = insights.skills or []
                    experience = [edu.role for edu in insights.experience] if insights.experience else []
                    matching_jobs = analyzer.query_jobs(education, skills, experience, job_index)

                    # Send retrieved nodes to LLM for final output
                    retrieved_context = "\n\n".join([match.node.get_content() for match in matching_jobs])
                    candidate_details = f"Education: {', '.join(education)}; Skills: {', '.join(skills)}; Experience: {', '.join(experience)}"

                    # Check for selected LLM and use the appropriate class
                    llm = Ollama(model=llm_option, temperature=0.0)
                    rag_engine = RAGStringQueryEngine(
                        retriever=job_index.as_retriever(),
                        llm=analyzer.llm,
                        qa_prompt=PromptTemplate(
                            template="""
                            You are expert in analyzing resumes, based on the following candidate details and job descriptions:
                            Candidate Details:
                            ---------------------
                            {query_str}
                            ---------------------
                            Job Descriptions:
                            ---------------------
                            {context_str}
                            ---------------------
                            Provide a concise list of the matching jobs. For each matching job, mention job-related details such as 
                            company, brief job description, location, employment type, salary range, URL for each suggestion, and a 
                            brief explanation of why the job matches the candidate's profile. Be critical in matching profile with the
                            jobs. Thoroughly analyze education, skills and experience to match jobs. Do not explain why the candidate's 
                            profile does not match with the other jobs. Do not include any summary. Order the jobs based on their relevance.
                            Answer:
                            """
                        ),
                    )

                    llm_response = rag_engine.custom_query(
                        candidate_details=candidate_details,
                        retrieved_jobs=retrieved_context
                    )

                    # Display extraacted information
                    st.subheader("Extracted Information")
                    st.write(f"**Name:** {insights.name}")
                    st.write(f"**Email:** {insights.email}")
                    st.write(f"**Age:** {insights.age}")
                    display_education(insights.education or [])
                    with st.spinner("Extracting skills..."):
                        display_skills(insights.skills or [], analyzer)
                    display_education(insights.education or [])
                    st.subheader("Top Matching Jobs with Explanation")
                    st.markdown(llm_response)
                    print("Done.")

                except Exception as e:
                    st.error(f"Failed to analyze the resume: {str(e)}")


def display_skills(skills: list[str], analyzer):
    """
    Display skills with their computer scores as large golen stars with partial coverage
    """
    if not skills:
        st.warning("No skills found to display.")
        return 
    
    st.subheader("Skills")
    # Custom CSS for large golden stars
    st.markdown(
        """
        <style>
        .star-container {
            display: inline-block;
            position: relative;
            font-size: 1.5rem;
            color: lightgray;
        }
        .star-container .filled {
            position: absolute;
            top: 0;
            left: 0;
            color: gold;
            overflow: hidden;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Compute scores for all skills
    print(f"Computing skill scores")
    skill_scores = analyzer.compute_skill_scores(skills)
    # Display each skill with a star rating 
    print(f"Completed computing scores")
    for skill in skills:
        score = skill_scores.get(skill, 0)  # Get the raw score 
        max_score = max(skill_scores.values()) if skill_scores else 1 

        # Normalize the score to a 5-star scale 
        normalized_score = (score / max_score) * 5 if max_score > 0 else 0

        # Split into full stars and partial star percentage
        full_stars = int(normalized_score)
        if (normalized_score - full_stars) <= 0.40:
            partial_star_percentage = 0
        elif (normalized_score - full_stars) > 0.4 and (normalized_score - full_stars) <= 70:
            partial_star_percentage = 50
        else:
            partial_star_percentage = 100


        # Generate the star display 
        stars_html = ""
        for i in range(5):
            if i < full_stars:
                # Fully filled star
                stars_html += '<span class="star-container"><span class="filled">★</span>★</span>'
            elif i == full_stars:
                # Partially filled star
                stars_html += f'<span class="star-container"><span class="filled" style="width: {partial_star_percentage}%">★</span>★</span>'
            else:
                # Empty star
                stars_html += '<span class="star-container">★</span>'

        # Displau skill name and star rating 
        st.markdown(f"**{skill}**: {stars_html}", unsafe_allow_html=True)


def display_education(education_list):
    """
    Display a list of educational qualifications
    """
    if education_list:
        st.subheader("Education")
        for education in education_list:
            institution = education.institution if education.institution else "Not found"
            degree = education.degree if education.degree else "Not found"
            year = education.graduation_date if education.graduation_date else "Not found"
            details = education.details if education.details else []
            formatted_details = ". ".join(details) if details else "No additional details provided."
            st.markdown(f"**{degree}**, {institution} ({year})")
            st.markdown(f"_Details_: {formatted_details}")


def display_experience(experience_list):
    """
    Display a single-level bulleted list of experiences.
    """
    if experience_list:
        st.subheader("Experience")
        for experience in experience_list:
            job_title = experience.role if experience.role else "Not found"
            company_name = experience.company if experience.company else "Not found"
            location = experience.location if experience.location else "Not found"
            start_date = experience.start_date if experience.start_date else "Not found"
            end_date = experience.end_date if experience.end_date else "Not found"
            responsibilities = experience.responsibilities if experience.responsibilities else ["Not found"]
            brief_responsibilities = ", ".join(responsibilities)
            st.markdown(
                f"- Worked as **{job_title}** from {start_date} to {end_date} in *{company_name}*, {location}, "
                f"where responsibilities include {brief_responsibilities}."
            )

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main()
