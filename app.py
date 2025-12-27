import os
import gradio as gr
from huggingface_hub import login
from smolagents import DuckDuckGoSearchTool, InferenceClientModel, CodeAgent
from tools import best_city, ClassifierTool

web_search_tool = DuckDuckGoSearchTool()
classifier_tool = ClassifierTool()

hf_token = os.environ.get('HF_TOKEN')
if hf_token:
    login(token=hf_token) 

model = InferenceClientModel(model_id='Qwen/Qwen3-4B-Instruct-2507', token=hf_token)

tools = [
    web_search_tool,
    classifier_tool,
    best_city
]

my_aiagent = CodeAgent(
    tools=tools,
    # For the purpose of this tutorial, just have tools you integrated.
    # Also by default when teh add_base_tools is set to true, it will integrate DuckDuckGo Search.
    add_base_tools=False, 
    model=model
)

def respond(
    message,
    history: list[dict[str, str]],
    system_message
):
    full_prompt = f"{system_message}\n\nChat history:\n{history}\n\nUser: {message}"
    response = my_aiagent.run(
        full_prompt,
        max_steps=5, 
        stream=False,
    )

    yield response

"""
For information on how to customize the ChatInterface, peruse the gradio docs: https://www.gradio.app/docs/chatinterface
"""
chatbot = gr.ChatInterface(
    respond,
    type="messages",
    additional_inputs=[],
)

with gr.Blocks() as demo:
    chatbot.render()


if __name__ == "__main__":
    demo.launch()
