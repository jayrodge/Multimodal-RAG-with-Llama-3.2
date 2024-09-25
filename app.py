# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import torch
import streamlit as st
from llama_index.core import Settings
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.embeddings.nvidia import NVIDIAEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM
from document_processors import load_multimodal_data, load_data_from_directory

# Initialize session state variables for managing chat history and document index
if 'history' not in st.session_state:
    st.session_state['history'] = []
if 'index' not in st.session_state:
    st.session_state['index'] = None

# Set up the page configuration
st.set_page_config(layout="wide")

# Initialize the Llama with HuggingFaceLLM integration
@st.cache_resource
def initialize_llm():
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    llm = HuggingFaceLLM(
        model_name=model_id,
        tokenizer_name=model_id,
        device_map="auto",
        model_kwargs={"torch_dtype": torch.bfloat16},
        # generate_kwargs={"pad_token_id": tokenizer.eos_token_id},
        tokenizer_kwargs={"padding_side": "left"},
        context_window=2048,
        max_new_tokens=50,
    )
    return llm

@st.cache_resource
def initialize_settings():
    # os.environ["NVIDIA_API_KEY"] = "" #set API key here
    Settings.embed_model = NVIDIAEmbedding(model="nvidia/nv-embedqa-e5-v5", truncate="END")
    Settings.llm = initialize_llm()
    Settings.text_splitter = SentenceSplitter(chunk_size=600)

# Create index from documents
def create_index(documents):
    vector_store = MilvusVectorStore(
            host = "127.0.0.1",
            port = 19530,
            dim = 1024
    )
    # vector_store = MilvusVectorStore(uri="./milvus_demo.db", dim=1024, overwrite=True) #For CPU only vector store
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_documents(documents, storage_context=storage_context)

def main():
    initialize_settings()
    llm = initialize_llm()

    col1, col2 = st.columns([1, 2])

    with col1:
        st.title("Multimodal RAG")

        input_method = st.radio("Choose input method:", ("Upload Files", "Enter Directory Path"))

        if input_method == "Upload Files":
            uploaded_files = st.file_uploader("Drag and drop files here", accept_multiple_files=True)
            if uploaded_files and st.button("Process Files"):
                with st.spinner("Processing files..."):
                    documents = load_multimodal_data(uploaded_files, llm)
                    st.session_state['index'] = create_index(documents)
                    st.success("Files processed and index created!")
        else:
            directory_path = st.text_input("Enter directory path:")
            if directory_path and st.button("Process Directory"):
                if os.path.isdir(directory_path):
                    with st.spinner("Processing directory..."):
                        documents = load_data_from_directory(directory_path, llm)
                        st.session_state['index'] = create_index(documents)
                        st.success("Directory processed and index created!")
                else:
                    st.error("Invalid directory path. Please enter a valid path.")

    with col2:
        if st.session_state['index'] is not None:
            st.title("Chat")

            query_engine = st.session_state['index'].as_query_engine(similarity_top_k=20, streaming=False)

            # Display chat messages
            chat_container = st.container()
            with chat_container:
                for message in st.session_state['history']:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])

            user_input = st.chat_input("Enter your query:")

            if user_input:
                with st.chat_message("user"):
                    st.markdown(user_input)
                st.session_state['history'].append({"role": "user", "content": user_input})

                with st.chat_message("assistant"):
                    with st.spinner("Generating response..."):
                        response = query_engine.query(user_input)
                        full_response = response.response
                        st.markdown(full_response)
                st.session_state['history'].append({"role": "assistant", "content": full_response})
                st.rerun()  # Force a rerun to update the chat display

            # Add a clear button
            if st.button("Clear Chat"):
                st.session_state['history'] = []
                st.rerun()


if __name__ == "__main__":
    main()