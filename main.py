import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser


_ = load_dotenv()



model = init_chat_model("llama3-8b-8192", model_provider="groq")

parser = StrOutputParser()


chain = model | parser


while True:
    messages = [
        HumanMessage(input("Human: "))
    ]

    print(f"Machine: {chain.invoke(messages)}")