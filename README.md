# Creating Multimodal AI Agent with Llama 3.2

## Overview

This app is a fork of [Multimodal RAG](https://github.com/NVIDIA/GenerativeAIExamples/tree/main/community/llm_video_series/video_2_multimodal-rag) that leverages the latest **Llama-3.2-3B**, a small language model and **Llama-3.2-11B-Vision**, a Vision Language Model from [Meta](https://www.llama.com/) to extract and index information from these documents including text files, PDFs, PowerPoint presentations, and images, allowing users to query the processed data through an interactive chat interface through streamlit.

The system utilizes [LlamaIndex](https://github.com/run-llama/llama_index) for efficient indexing and retrieval of information for orchestration, Hugging Face integration for LlamaIndex for generating inference output from Llama 3.2 VLM and SLM, [NIM microservices](https://www.nvidia.com/en-us/ai/) for high-performance inference on [Google DePlot](https://build.nvidia.com/google/google-deplot), and [Milvus](https://github.com/milvus-io/milvus) as a vector database for efficient storage and retrieval of embedding vectors. This combination of technologies enables the application to handle complex multimodal data, perform advanced queries, and deliver rapid, context-aware responses to user inquiries.

The Llama 3.2 language and vision models with NIM microservices will be integrated in this reference app soon.

## Features

- **Multi-format Document Processing**: Handles text files, PDFs, PowerPoint presentations, and images.
- **Advanced Text Extraction**: Extracts text from PDFs and PowerPoint slides, including tables and embedded images.
- **Image Analysis**: Uses a VLM (Llama-3.2-11B-Vision) running on Hugging Face transformers to describe images and Google's DePlot for processing graphs/charts on NIM microservices.
- **Vector Store Indexing**: Creates a searchable index of processed documents using Milvus vector store. This folder is auto generated on execution.
- **Interactive Chat Interface**: Allows users to query the processed information through a chat-like interface.

## Setup

1. Clone the repository:
```
git clone https://github.com/jayrodge/Multimodal-RAG-with-Llama-3.2.git
cd Multimodal-RAG-with-Llama-3.2/
```

2. (Optional) Create a conda environment or a virtual environment:

   - Using conda:
     ```
     conda create --name multimodal-rag python=3.10
     conda activate multimodal-rag
     ```

   - Using venv:
     ```
     python -m venv venv
     source venv/bin/activate

3. Install the required packages:
```
pip install -r requirements.txt
```

4. Set up your NVIDIA API key as an environment variable or define it in `initialize_settings()` in `app.py`:
```
export NVIDIA_API_KEY="your-api-key-here"
```
Generate the NVIDIA API key on [build.nvidia.com](https://build.nvidia.com)

5. Refer this [tutorial](https://milvus.io/docs/install_standalone-docker-compose-gpu.md) to install and start the GPU-accelerated Milvus container:

```
sudo docker compose up -d
```


## Usage

1. Ensure the Milvus container is running:

```bash
docker ps
```

2. Run the Streamlit app:
```
streamlit run app.py
```

3. Open the provided URL in your web browser.

4. Choose between uploading files or specifying a directory path containing your documents.

5. Process the files by clicking the "Process Files" or "Process Directory" button.

6. Once processing is complete, use the chat interface to query your documents.

## File Structure

- `app.py`: Main Streamlit application
- `utils.py`: Utility functions for image processing and API interactions
- `document_processors.py`: Functions for processing various document types
- `requirements.txt`: List of Python dependencies
- `vectorstore/` : Repository to store information from pdfs and ppt, created automatically


## GPU Acceleration for Vector Search
To utilize GPU acceleration in the vector database, ensure that:
1. Your system has a compatible NVIDIA GPU.
2. You're using the GPU-enabled version of Milvus (as shown in the setup instructions).
3. There are enough concurrent requests to justify GPU usage. GPU acceleration typically shows significant benefits under high load conditions.

It's important to note that GPU acceleration will only be used when the incoming requests are extremely high. For more detailed information on GPU indexing and search in Milvus, refer to the [official Milvus GPU Index documentation](https://milvus.io/docs/gpu_index.md).

To connect the GPU-accelerated Milvus with LlamaIndex, update the MilvusVectorStore configuration in app.py:
```
vector_store = MilvusVectorStore(
    host="127.0.0.1",
    port=19530,
    dim=1024,
    collection_name="your_collection_name",
    gpu_id=0  # Specify the GPU ID to use
)
```
