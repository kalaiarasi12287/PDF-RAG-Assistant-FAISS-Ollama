# PDF RAG Assistant using FAISS and Ollama

## Project Overview

This project implements a Retrieval-Augmented Generation (RAG) system that answers questions from a PDF document using semantic retrieval and a locally hosted Large Language Model (LLM).

The application extracts text from a PDF, splits the document into overlapping chunks, generates vector embeddings, and indexes the embeddings using FAISS. When a user asks a question, the system retrieves relevant chunks and supplies them to an LLM to generate a context-grounded answer.

The project also includes a RAG evaluation workflow that measures evidence retrieval, examines answer faithfulness using an LLM judge, supports manual answer review, and exports evaluation results to a CSV report.

## Key Features

- **PDF text extraction:** Extracts text from PDF pages using PyPDF.
- **Document chunking:** Uses recursive character-based splitting with configurable chunk size and overlap.
- **Semantic embeddings:** Generates vector embeddings using Sentence Transformers.
- **Vector search:** Uses a FAISS `IndexFlatL2` index to retrieve the nearest document chunks.
- **Context-grounded generation:** Uses Ollama to run a local LLM that answers questions using retrieved context.
- **Fallback responses:** Returns a predefined response when retrieval produces no results or when the supplied context does not support an answer.
- **Retrieval evaluation:** Evaluates whether expected evidence appears in the top three retrieved chunks.
- **Faithfulness evaluation:** Uses an LLM judge to assess whether generated answers are supported by retrieved context.
- **Manual answer review:** Records manually reviewed answers in the evaluation workflow.
- **Evaluation reporting:** Exports question-level evaluation details to a CSV file.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Google Colab | Development and experimentation environment |
| PyPDF | PDF text extraction |
| LangChain Text Splitters | Document chunking |
| Sentence Transformers | Text embedding generation |
| `all-MiniLM-L6-v2` | Embedding model |
| FAISS | Vector indexing and similarity search |
| Ollama | Local LLM serving |
| Gemma 3 4B | Answer generation and LLM-based judging |
| NumPy | Numerical operations and embedding arrays |
| Pandas | Evaluation data processing |
| Requests | Checking Ollama API availability |
| JSON and Regular Expressions | Structured output parsing and text normalization |

## System Architecture

```text
                PDF Document
                      |
                      v
             Extract PDF Text
                      |
                      v
              Split into Chunks
          (700 characters, 150 overlap)
                      |
                      v
           Generate Text Embeddings
             (all-MiniLM-L6-v2)
                      |
                      v
                FAISS Index
                      |
                      |
User Question --------+
       |
       v
Generate Question Embedding
       |
       v
Retrieve Top 3 Relevant Chunks
       |
       v
Combine Retrieved Context
       |
       v
Prompt Local LLM through Ollama
             (Gemma 3 4B)
       |
       v
     Final Answer
       |
       v
   RAG Evaluation
       |
       +--> Evidence Recall@3
       +--> LLM-Judge Faithfulness
       +--> Manual Answer Review
       +--> Abstention Accuracy
       |
       v
  CSV Evaluation Report
```

## How the RAG Pipeline Works

### 1. PDF Text Extraction

The system uses `PdfReader` to read the input PDF page by page. Extracted text is labelled with its page number and combined into a single document string.

Pages without extractable text are skipped. Scanned PDFs may require an OCR step before they can be processed.

### 2. Document Chunking

The extracted text is split using `RecursiveCharacterTextSplitter`.

- Chunk size: 700 characters
- Chunk overlap: 150 characters
- Separators: Paragraph breaks, line breaks, sentence boundaries, spaces, and individual characters

The overlap helps preserve context when information spans adjacent chunks.

### 3. Embedding Generation

The `all-MiniLM-L6-v2` Sentence Transformer converts each chunk into a numerical vector representation.

The embeddings are converted to NumPy `float32` arrays before being added to the FAISS index.

The implementation also checks that the embedding values are finite.

### 4. FAISS Vector Indexing

The project uses:

`faiss.IndexFlatL2(dimension)`

This index performs exact nearest-neighbour search using squared L2 distance. The system stores the generated embeddings in the index and retrieves the three nearest chunks for each question by default.

Lower L2 distance indicates a closer vector match. This implementation uses L2 distance rather than cosine similarity.

### 5. Context Retrieval

For every question, the system:

1. Generates an embedding for the question.
2. Searches the FAISS index.
3. Retrieves up to three relevant chunks.
4. Combines their text into a context string.

The retrieved chunks and their distances can also be inspected during evaluation.

### 6. LLM Answer Generation

The retrieved context and user question are inserted into a prompt and sent to the local Ollama model `gemma3:4b`.

The prompt instructs the model to:

- Answer using only the supplied context.
- Avoid unsupported names, dates, numbers, and facts.
- Return a predefined fallback answer when the information is not supported.
- Keep answers concise.
- Treat retrieved text as source material rather than instructions.

The generation temperature is set to zero to encourage more consistent outputs.

**Fallback response:**

> I could not find this information in the document.

This prompt-based restriction helps reduce unsupported answers but does not guarantee that hallucinations will never occur.

## Evaluation Framework

The project includes ten sample question-and-answer test cases covering information present in the PDF and a question whose answer is not expected to be present.

### 1. Evidence Recall@3

The retrieval evaluation checks whether the expected evidence text appears in any of the top three retrieved chunks for each answerable question.

The reported metric is calculated as:

`Evidence Recall@3 = Evidence hits / Number of answerable test cases`

For this project, the evidence check uses normalized text matching. It is a useful initial retrieval check, but it is not a general semantic-relevance metric and depends on the expected evidence wording.

The notebook also prints retrieved chunks and their L2 distances for inspection.

### 2. LLM-Judge Faithfulness

An LLM judge receives the question, retrieved context, and generated answer. It returns JSON containing:

- `supported`: A Boolean indicating whether the answer is supported by the context.
- `reason`: A short explanation of the judgement.

The code validates the Boolean field and records invalid judge outputs as missing judgements rather than treating them as valid results.

The resulting rate is an **LLM-judge-estimated faithfulness rate**, not an independently verified ground-truth score.

### 3. Manual Answer Review

The notebook contains a manually assigned review dictionary for the ten test questions. These labels are then used to calculate the manually reviewed answer accuracy.

These labels must be checked against the actual generated answers before reporting the metric as a validated result.

### 4. Abstention Accuracy

The abstention evaluation checks whether the model returns the exact predefined fallback response for unanswerable test cases.

The notebook reports the proportion of unanswerable cases for which the expected fallback response was generated.

### 5. CSV Evaluation Report

The notebook saves an evaluation report as:

`rag_evaluation_report.csv`

The report includes information such as:

- Test question and expected answer
- Whether the question is answerable
- Retrieved context
- Generated answer
- LLM-judge faithfulness result and explanation
- Manual answer-review label
- Abstention status
- Retrieved chunk details and L2 distances

The CSV is intended for reviewing and comparing test results.

## Project Setup

### Prerequisites

- A Google account with access to Google Colab
- A GitHub account for version control
- A PDF document that you are authorized to process
- Sufficient Colab runtime resources to install and run Ollama and the selected model

### Step 1: Install Dependencies

Run the following commands in a Google Colab cell:

```bash
!apt-get update -qq
!apt-get install -y zstd
!curl -fsSL https://ollama.com/install.sh | sh
```

Install the Python libraries:

```python
!pip install -q sentence-transformers faiss-cpu pypdf ollama \
    langchain-text-splitters pandas numpy requests
```

### Step 2: Start Ollama

The notebook checks whether the Ollama API is already running and starts the server if necessary.

The API is checked at:

`http://127.0.0.1:11434/api/tags`

The notebook pulls the `gemma3:4b` model used for answer generation and faithfulness judging. It also pulls `llama3.2`, although the shown generation and judging functions use `gemma3:4b`.

Ollama and the model must be available in the active runtime before the LLM calls can succeed.

### Step 3: Provide a PDF

Update the PDF path in the notebook:

```python
PDF_PATH = "/content/sample_data/your_document.pdf"
```

Upload your own PDF to Colab or change this path to the location of your file.

The notebook raises an error if no extractable text is found.

### Step 4: Run the Notebook

Execute the notebook cells in order:

1. Install dependencies and start Ollama.
2. Load the PDF and extract text.
3. Split the document into chunks.
4. Generate embeddings and build the FAISS index.
5. Ask questions and generate answers.
6. Run retrieval evaluation.
7. Run answer generation across the test cases.
8. Run LLM-judge faithfulness evaluation.
9. Review answers and calculate abstention behaviour.
10. Export and download the CSV evaluation report.

## Repository Structure

A recommended repository structure is:

```text
PDF-RAG-Assistant-FAISS-Ollama/
├── README.md
├── rag_project.ipynb
├── requirements.txt
├── .gitignore
└── reports/
    └── .gitkeep
```

Replace `rag_project.ipynb` with your actual notebook filename.

The `reports/` directory can hold evaluation reports that do not contain confidential information. The notebook currently writes the CSV to `/content/rag_evaluation_report.csv`; update the path if you want it saved inside the repository.

## Important Implementation Notes and Limitations

- **Document-specific answers:** Answers depend on the content extracted from the supplied PDF.
- **PDF extraction:** Complex layouts, tables, and scanned documents may not extract cleanly.
- **Retrieval limitations:** The current implementation uses a fixed top-three retrieval strategy without a relevance threshold.
- **Index persistence:** The FAISS index and embeddings are created in memory; the current notebook does not persist and reload the index.
- **Prompt-based grounding:** Instructions to use only retrieved context reduce the risk of unsupported answers but are not a guarantee.
- **Evaluation scope:** The evaluation uses ten manually defined test cases, so results should not be generalized to other documents or larger question sets.
- **LLM-judge limitations:** Judge outputs can be inconsistent or incorrect. Manual review is important.
- **Runtime requirements:** Running Ollama and a 4B-parameter model in Colab depends on available memory and runtime resources.
- **Privacy:** Do not commit private PDFs, personal resumes, confidential documents, credentials, or sensitive evaluation outputs. Use a synthetic or public sample document for demonstrations.

## Future Enhancements

- Build a Streamlit interface for interactive document question answering.
- Support multiple PDF uploads.
- Persist and reload the FAISS index and chunk metadata.
- Add page-level source citations to generated answers.
- Add a configurable retrieval count and relevance threshold.
- Improve evaluation with semantic retrieval metrics and more diverse test questions.
- Compare embedding models and chunking strategies.
- Add automated tests for retrieval, fallback behaviour, and output validation.
- Add a reproducible environment file and a cleaner application entry point.

## Learning Outcomes

This project demonstrates practical implementation of:

- Retrieval-Augmented Generation (RAG)
- PDF ingestion and text preprocessing
- Document chunking and overlapping windows
- Sentence embeddings and vector search
- FAISS nearest-neighbour retrieval
- Local LLM integration through Ollama
- Context-constrained prompting
- Retrieval evaluation and LLM-as-a-judge evaluation
- Manual validation and CSV-based reporting

## Author

**Kalaiarasi Thangamani**

AI/ML learning portfolio project.

## Disclaimer

This is an educational RAG project intended for experimentation and portfolio demonstration. Generated answers and evaluation metrics should be validated before use in real-world applications.

