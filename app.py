


import streamlit as st

st.set_page_config(
    page_title="Legislative Evidence Locator",
    layout="wide"
)

st.markdown("""
<style>
.main {
    background-color: #f5f7fb;
}
.block-container {
    padding-top: 2rem;
}
.header-box {
    background-color: #14213d;
    padding: 28px;
    border-radius: 12px;
    color: white;
    margin-bottom: 25px;
}
.header-box h1 {
    color: white;
}
.header-box p {
    color: #e0eaff;
    font-size: 16px;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.title("LEGISLATIVE EVIDENCE LOCATOR")

page = st.sidebar.radio(
    "MAIN MENU",
    [
        "Dashboard",
        "Document Processing",
        "Knowledge Base",
        "Evidence Retrieval",
        "AI Question Answering",
        "About Project"
    ]
)

st.sidebar.divider()
st.sidebar.write("PS 160 Hackathon Project")

if page == "Dashboard":

    st.markdown("""
    <div class="header-box">
        <h1>Legislative Transcript Search</h1>
        <p>AI-Powered Parliamentary Evidence Search and Locator</p>
        <p>Find parliamentary discussions, speakers and original sources easily.</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Project Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("PDF Documents", "2")
    col2.metric("Total Pages", "2615")
    col3.metric("Project Modules", "4")
    col4.metric("AI Assistant", "Planned")

    st.divider()

    st.subheader("Project Workflow")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 1. Document Processing")
        st.write("Extracts and cleans text from Lok Sabha PDF documents.")

        st.markdown("### 2. Knowledge Base")
        st.write("Creates text chunks, embeddings and a FAISS database.")

    with col2:
        st.markdown("### 3. Evidence Retrieval")
        st.write("Finds relevant discussions and checks source information.")

        st.markdown("### 4. AI Question Answering")
        st.write("Uses retrieved evidence to generate clear answers.")

    st.divider()

    st.subheader("Quick Navigation")
    st.write("Select a module from the sidebar to explore the application.")

elif page == "Document Processing":

    st.title("Document Processing")
    st.subheader("Member 1: Document Ingestion and Preprocessing")

    st.write("Upload Lok Sabha parliamentary debate documents.")

    files = st.file_uploader(
        "Choose PDF documents",
        type=["pdf"],
        accept_multiple_files=True
    )

    if files:
        st.success(str(len(files)) + " PDF documents uploaded.")

        for file in files:
            st.write(file.name)

        st.info("PDF extraction and preprocessing will be connected here.")

    st.divider()

    st.subheader("Preprocessing Tasks")

    st.write("Text Extraction")
    st.write("Text Cleaning")
    st.write("Speaker and Date Identification")
    st.write("Page Number Preservation")
    st.write("Text Chunking")

elif page == "Knowledge Base":

    st.title("Knowledge Base")
    st.subheader("Member 2: Chunking, Embeddings and Vector Database")

    col1, col2, col3 = st.columns(3)

    col1.metric("Text Chunks", "Pending")
    col2.metric("Embeddings", "Pending")
    col3.metric("FAISS Database", "Not Connected")

    st.divider()

    st.subheader("Knowledge Base Process")

    st.write("1. Divide extracted text into smaller sections.")
    st.write("2. Convert text sections into numerical embeddings.")
    st.write("3. Store embeddings in the FAISS vector database.")
    st.write("4. Make the information searchable.")

    st.warning("Waiting for Member 2 integration.")

elif page == "Evidence Retrieval":

    st.title("Evidence Retrieval")
    st.subheader("Member 3: RAG Retrieval and Evidence Validation")

    question = st.text_input(
        "Enter a question to search parliamentary records"
    )

    if st.button("Search Evidence"):

        if question.strip():
            st.info("Member 3 retrieval module will be connected here.")
        else:
            st.warning("Please enter a question.")

    st.divider()

    st.subheader("Retrieved Evidence")

    st.write("Original Statement: Waiting for retrieval")
    st.write("Speaker: Waiting for verification")
    st.write("Discussion Date: Not available")
    st.write("Document: Not available")
    st.write("Page Number: Not available")

elif page == "AI Question Answering":

    st.title("AI Question Answering")
    st.subheader("Member 4: Frontend and Gemini Integration")

    question = st.text_area(
        "Enter your question",
        placeholder="Example: Who raised the issue of generic medicines?"
    )

    if st.button("Generate Answer"):

        if question.strip():
            st.info("Gemini AI integration will be connected here.")
        else:
            st.warning("Please enter a question.")

    st.divider()

    st.subheader("Answer and Source Information")

    st.write("AI Answer: Waiting for Gemini integration")
    st.write("Speaker Name: Waiting for evidence")
    st.write("Discussion Date: Waiting for evidence")
    st.write("Original Evidence: Waiting for retrieval")
    st.write("Document Name: Waiting for evidence")
    st.write("Page Number: Waiting for evidence")

elif page == "About Project":

    st.title("About Our Project")

    st.subheader("Problem Statement")

    st.write("""
    Lok Sabha parliamentary proceedings contain extensive discussions,
    questions, answers and ministerial statements. Finding specific
    information in lengthy PDF documents is difficult.

    Our project aims to provide a search system that retrieves relevant
    parliamentary evidence and displays the original statement,
    speaker name, discussion date and page number.
    """)

    st.subheader("Technologies Used")

    st.write("Python: Main programming language")
    st.write("Streamlit: Frontend dashboard")
    st.write("PyMuPDF: PDF text extraction")
    st.write("Sentence Transformers: Text embeddings")
    st.write("FAISS: Similarity search")
    st.write("RAG: Evidence-based retrieval and generation")
    st.write("Gemini API: AI answer generation")

st.divider()

st.caption("PS 160 | Legislative Transcript Search and Evidence Locator")
