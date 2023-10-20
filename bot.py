import gradio as gr
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
import os


persist_directory = 'faq_db'
os.environ['OPENAI_API_KEY'] = "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"

# Initialize the BGE Embedding Model
# model_name = "BAAI/bge-large-en"
# model_kwargs = {'device': 'cpu'}
# encode_kwargs = {'normalize_embeddings': True}

# embeddings = HuggingFaceBgeEmbeddings(
#     model_name=model_name,
#     model_kwargs=model_kwargs,
#     encode_kwargs=encode_kwargs
# )

embeddings = OpenAIEmbeddings()

# Load the vector database from the persisted directory
vectordb = db = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings,
)

# Step 1: Prepare your query
# query = "How to describe customer service skills?"
# get_result(query)

# Step 2: Search the database
# Assuming that Chroma has a search method


def get_result(query):
    results = vectordb.similarity_search_with_score(query, k=1)
    try:
        final_results = " ".join(d[0].metadata['answer'] for d in results if d[1] < 0.15)
    except Exception as e:
        final_results = "Sorry, I can answer questions about Swinburne Online only"
    return final_results


with gr.Blocks() as demo:
    chatbot = gr.Chatbot([], elem_id="chatbot").style(height=500)
    msg = gr.Textbox()
    clear = gr.Button('Clear')

    def user(user_message, history):
        return "", history + [[user_message, None]]

    def bot(history):
        bot_message = get_result(history[-1][0])
        if bot_message == "":
            bot_message = "Sorry, I can answer questions about Swinburne Online only."
        history[-1][-1] = bot_message
        return history

    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(bot, chatbot, chatbot)
    clear.click(lambda: None, None, chatbot, queue=False)

demo.launch(share=True)
