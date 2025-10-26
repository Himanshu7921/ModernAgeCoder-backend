# ModernAgeCoders AI Chatbot Backend

**ModernAgeCoders AI Chatbot** is a personal project designed to explore the use of **Retrieval-Augmented Generation (RAG)** in delivering instant answers about online classes, courses, schedules, and instructors. This backend demonstrates how an AI agent can intelligently answer queries related to ModernAgeCoders’ courses and services using a custom RAG framework.

The AI agent can read company/course/teacher PDFs, process them with **RAG**, and provide **contextually accurate and friendly answers** to students’ questions. This project is primarily implemented as a learning exercise while also serving as a real backend for the chatbot.

---

## Features

* **RAG-Powered AI Agent**: Answers questions about classes, teachers, pricing, schedules, and batches using the Retrieval-Augmented Generation approach.
* **Custom RAG Framework**: Built from scratch, published as **[RetrievalMind](https://github.com/Himanshu7921/RetrievalMind)** on PyPI.

  ```bash
  pip install RetrievalMind==0.1.1
  ```

  Anyone can use RetrievalMind locally to build their own RAG-powered applications.
* **Supports PDFs as Knowledge Source**: Handles ModernAgeCoders’ course lists, teacher info, schedules, and FAQs.
* **Lightweight Web Frontend**: Simple HTML interface to interact with the AI agent.

---

## Project Structure

```
ModernAgeCoders-Chatbot-Backend/
├── __pycache__/
├── data/
│   ├── faqs/
│   │   ├── Courses_faqs.pdf
│   │   ├── Teacher_faqs.pdf
│   │   └── Pricing_faqs.pdf
│   ├── documents/
│   │   └── CompanyDetails.pdf
│   └── vector_store/
│       └── embeddings/
├── index.html
├── main.py
├── server.py
├── serve_frontend.py
```

**Folder Details:**

* **data/faqs**: Contains PDF files with frequently asked questions.
* **data/documents**: Contains PDF files with company details and course info.
* **data/vector_store**: Stores vectorized embeddings of documents for RAG retrieval.
* **main.py**: Entry point for the RAG-based AI agent.
* **server.py**: Backend server handling AI queries.
* **serve_frontend.py**: Serves the HTML frontend for interaction.
* **index.html**: Web interface to chat with ModernAgeCoders AI Chatbot.

---

## Usage

1. **Install RetrievalMind (custom RAG framework):**

   ```bash
   pip install RetrievalMind==0.1.1
   ```

2. **Clone this repository:**

   ```bash
   git clone https://github.com/yourusername/ModernAgeCoders-Chatbot-Backend
   cd ModernAgeCoders-Chatbot-Backend
   ```

3. **Run the backend server:**

   ```bash
   python server.py
   ```

4. **Serve the frontend (optional if using index.html locally):**

   ```bash
   python serve_frontend.py
   ```

5. **Open `index.html`** in your browser and start chatting with the AI agent.

---

## About RetrievalMind

**RetrievalMind** is a custom RAG framework that simplifies creating AI agents capable of answering queries from structured documents. It allows you to:

* Ingest documents (PDFs, text files)
* Generate embeddings
* Store embeddings in a vector store
* Retrieve relevant context and generate AI responses

GitHub: [https://github.com/Himanshu7921/RetrievalMind](https://github.com/Himanshu7921/RetrievalMind)

PyPI:

```bash
pip install RetrievalMind==0.1.1
```

This framework powers the **ModernAgeCoders AI Chatbot**, enabling accurate responses from course, teacher, and schedule documents.

---

## Future Enhancements

* Multi-user support via web interface
* Integration with additional file formats like DOCX or CSV
* Advanced ranking and filtering for precise answers
* Cloud deployment for enterprise or large-scale usage

---

## License

This project is **for learning, demonstration, and ModernAgeCoders internal use**. For details about the RetrievalMind framework license, check its [GitHub repository](https://github.com/Himanshu7921/RetrievalMind).
