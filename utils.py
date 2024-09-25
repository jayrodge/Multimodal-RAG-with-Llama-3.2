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
import base64
import streamlit as st
import fitz
import torch
from io import BytesIO
from PIL import Image
import requests
from transformers import MllamaForConditionalGeneration, AutoProcessor

@st.cache_resource
def initialize_vlm():
    """Initialize and load the Vision-Language Model (VLM) for image description from a specified model ID."""
    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    vlm_model = MllamaForConditionalGeneration.from_pretrained(model_id, device_map="auto", torch_dtype=torch.float16)
    vlm_processor = AutoProcessor.from_pretrained(model_id)
    return vlm_model, vlm_processor

def get_b64_image_from_content(image_content):
    """Convert image content to base64 encoded string."""
    img = Image.open(BytesIO(image_content))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def is_graph(image_content):
    """Determine if an image is a graph, plot, chart, or table."""
    res = describe_image(image_content)
    return any(keyword in res.lower() for keyword in ["graph", "plot", "chart", "table"])

def process_graph(image_content, llm):
    """Process a graph image and generate a description."""
    deplot_description = process_graph_deplot(image_content)
    response = llm.complete("Your responsibility is to explain charts. You are an expert in describing the responses of linearized tables into plain English text for LLMs to use. Explain the following linearized table. " + deplot_description)
    return response.text

def describe_image(image_content):
    """Generate a description of an image using the multimodal LLM."""
    vlm_model, vlm_processor = initialize_vlm()
    image = Image.open(BytesIO(image_content))
    messages = [
    {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe what you see in this image"}
        ]
    }
    ]
    text = vlm_processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = vlm_processor(text=text, images=image, return_tensors="pt").to(vlm_model.device)
    output = vlm_model.generate(**inputs, max_new_tokens=1024)
    text = vlm_processor.decode(output[0], skip_special_tokens=True)
    return text

def process_graph_deplot(image_content):
    """Process a graph image using NVIDIA's Deplot API."""
    invoke_url = "https://ai.api.nvidia.com/v1/vlm/google/deplot"
    image_b64 = get_b64_image_from_content(image_content)
    api_key = os.getenv("NVIDIA_API_KEY")
    
    if not api_key:
        raise ValueError("NVIDIA API Key is not set. Please set the NVIDIA_API_KEY environment variable.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json"
    }

    payload = {
        "messages": [
            {
                "role": "user",
                "content": f'Generate underlying data table of the figure below: <img src="data:image/png;base64,{image_b64}" />'
            }
        ],
        "max_tokens": 1024,
        "temperature": 0.20,
        "top_p": 0.20,
        "stream": False
    }

    response = requests.post(invoke_url, headers=headers, json=payload)
    return response.json()["choices"][0]['message']['content']

def extract_text_around_item(text_blocks, bbox, page_height, threshold_percentage=0.1):
    """Extract text above and below a given bounding box on a page."""
    before_text, after_text = "", ""
    vertical_threshold_distance = page_height * threshold_percentage
    horizontal_threshold_distance = bbox.width * threshold_percentage

    for block in text_blocks:
        block_bbox = fitz.Rect(block[:4])
        vertical_distance = min(abs(block_bbox.y1 - bbox.y0), abs(block_bbox.y0 - bbox.y1))
        horizontal_overlap = max(0, min(block_bbox.x1, bbox.x1) - max(block_bbox.x0, bbox.x0))

        if vertical_distance <= vertical_threshold_distance and horizontal_overlap >= -horizontal_threshold_distance:
            if block_bbox.y1 < bbox.y0 and not before_text:
                before_text = block[4]
            elif block_bbox.y0 > bbox.y1 and not after_text:
                after_text = block[4]
                break

    return before_text, after_text

def process_text_blocks(text_blocks, char_count_threshold=500):
    """Group text blocks based on a character count threshold."""
    current_group = []
    grouped_blocks = []
    current_char_count = 0

    for block in text_blocks:
        if block[-1] == 0:  # Check if the block is of text type
            block_text = block[4]
            block_char_count = len(block_text)

            if current_char_count + block_char_count <= char_count_threshold:
                current_group.append(block)
                current_char_count += block_char_count
            else:
                if current_group:
                    grouped_content = "\n".join([b[4] for b in current_group])
                    grouped_blocks.append((current_group[0], grouped_content))
                current_group = [block]
                current_char_count = block_char_count

    # Append the last group
    if current_group:
        grouped_content = "\n".join([b[4] for b in current_group])
        grouped_blocks.append((current_group[0], grouped_content))

    return grouped_blocks

def save_uploaded_file(uploaded_file):
    """Save an uploaded file to a temporary directory."""
    temp_dir = os.path.join(os.getcwd(), "vectorstore", "ppt_references", "tmp")
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, uploaded_file.name)
    
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(uploaded_file.read())
    
    return temp_file_path