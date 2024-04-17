# 🦙 Llama-2-GGML-CSV-Chatbot 🤖

## Reference
1. **This project is inspired/referenced from the below specified repository.It was taken from hugging face:**
   ```bash
   git clone https://github.com/ThisIs-Developer/Llama-2-GGML-CSV-Chatbot.git
   ```
2. The repository contents were thoroughly studied and modified as required for the **Desi Rail Chatbot Project**

## Overview
The **Llama-2-GGML-CSV-Chatbot** is a conversational tool powered by a fine-tuned large language model (LLM) known as *Llama-2 7B*. This chatbot utilizes CSV retrieval capabilities, enabling users to engage in multi-turn interactions based on uploaded CSV data.

<img width="2000" src="assets/workflow_1.jpg">

## 🚀 Features

- **CSV Data Interaction:** Allows users to engage in conversations based on CSV data.
- **Multi-turn Interaction:** Supports seamless multi-turn interactions for a better conversational experience.

## Development Specs
- Utilizes [Llama-2 7B](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main) and [Sentence Transformers](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) for robust functionality.
- Developed using [Langchain](https://github.com/langchain-ai/langchain) and [Streamlit](https://github.com/streamlit/streamlit) technologies for enhanced performance.
- Cross-platform compatibility with Linux, macOS, or Windows OS.

## 🛠️ Installation
1. **Clone This Repository:**
 ```bash
   git clone https://github.com/Kav1n-Lal/desi_rail_chatbot_using_a_dataset.git
   ```
2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Download the Llama 2 Model:

Download the Llama 2 model file named `llama-2-7b-chat.ggmlv3.q4_0.bin` from the following link:

[🔗Download Llama 2 Model](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main)

### Llama 2 Model Information

| Name                           | Quant method | Bits | Size    | Max RAM required |
|--------------------------------|--------------|------|---------|------------------|
| llama-2-7b-chat.ggmlv3.q4_0.bin | q4_0         | 4    | 3.79 GB | 6.29 GB          |

**Note:** After downloading the model, add the model file to the `models` directory. The file should be located at `models\llama-2-7b-chat.ggmlv3.q4_0.bin`, in order to run the code.

## 📝 Usage

1. **Run the Application:**
   ```bash
   streamlit run app.py
   ```
2. **Access the Application:**
   - Once the application is running, access it through the provided URL.
   - 
## System Requirements
- **CPU:** Intel® Core™ i5 or equivalent.
- **RAM:** 8 GB.
- **Disk Space:** 7 GB.
- **Hardware:** Operates on CPU; no GPU required.

## 🤖 How to Use
- Copy the cloned repository path and at the end add "\\chat.csv" on line 41 in app.py before running the code.
- Eg.uploaded_file='C:\\Users\\Kavin\\Desktop\\desi_rail_chatbot\\chat.csv'
- Upon running the application, you'll be presented with a sidebar providing information about the chatbot and a selectbox to specify the source station i.e where to start the journey from.
- If you get import pwd error just refresh the streamlit app.
- Then another selectbox box appears to specify the destination station.
- Then click on 'Chat' button.
- Enter your query or prompt in the input field provided.
- The chatbot will process your query and generate a response based on the railway CSV data and the Llama-2-7B-Chat-GGML model.


## 📖 ChatBot Conversation

### ⚡Streamlit ver. on [#v2.0.2.dev20240102](https://github.com/ThisIs-Developer/Llama-2-GGML-CSV-Chatbot/releases/tag/v2.0.2.dev20240102)
![Screenshot (1)](https://github.com/Kav1n-Lal/desi_rail_chatbot_using_a_dataset/assets/116146011/dbd36ea3-32b5-4ab7-b68b-0250c2d6fb88)




