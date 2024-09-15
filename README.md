# PDF Chatbot
PDF Chatbot is an AI-powered application that allows users to upload PDF files and ask questions based on the content of those files. 
It supports multiple chat sessions, file management, and API key customization. 

## Key Features
- **AI Models**: Powered by Google's `gemini-1.5-flash` for the chatbot model,
with Cohere's `embed-english-light-v3.0` used for embeddings.
- **Multiple Chats**: Create and manage multiple chat sessions.
- **Upload Files**: Upload PDFs in the `Upload Files` section.
- **Delete Files**: Manage the database by deleting unnecessary PDF files in the `Delete Files` section.
- **Custom API Keys**: Save and update API keys easily in the `API Keys` section.

## Getting Started

### Local Setup

To run the PDF Chatbot locally, follow these steps:
1. **Clone the repository**
   ```bash
   git clone https://github.com/andrewiva99/pdf-chatbot
   cd pdf-chatbot
   ```

3. **Create Conda Environment**:
    ```bash
    conda create --name <env_name> python=3.10
    conda activate <env_name>
    ```

4. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

5. **Run the Application**:
    ```bash
    streamlit run pdf_chatbot.py
    ```

### Docker Setup

Alternatively, you can use Docker to run the application:

```bash
docker run -p 8501:8501 andreybg/pdf-chatbot
```
