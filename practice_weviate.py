# # ============================================================================
# # IMPORTS
# # ============================================================================
# import os
# import json
# import torch
# import tiktoken
# import weaviate
# from sentence_transformers import SentenceTransformer, util
# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.document_loaders import TextLoader
# from langchain.chains import create_retrieval_chain
# from langchain.chains.combine_documents import create_stuff_documents_chain
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_community.callbacks.manager import get_openai_callback
# from langchain_weaviate import WeaviateVectorStore 
# from dotenv import load_dotenv

# load_dotenv()


# # ============================================================================
# # CONFIGURATION
# # ============================================================================
# base_intents_path = "intents"
# type_bot = "advanced"
# embedding = OpenAIEmbeddings()
# weaviate_client = weaviate.connect_to_local()
# device = "cuda" if torch.cuda.is_available() else "cpu"
# model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
# data = {}


# # ============================================================================
# # UTILITY FUNCTIONS
# # ============================================================================
# def count_tokens(text):
#     tokenizer = tiktoken.get_encoding("cl100k_base")
#     return len(tokenizer.encode(text))


# def estimate_embedding_cost(token_count, cost_per_million_tokens=0.10):
#     return (token_count / 1_000_000) * cost_per_million_tokens


# # ============================================================================
# # DATA MANAGEMENT
# # ============================================================================
# def update_user_project_data(user_id, project_id, bot_type):
#     key = f"{user_id}_{project_id}"
#     intents_file = os.path.join(base_intents_path, bot_type, user_id, project_id, "data.json")
#     knowledgebase_file = os.path.join(base_intents_path, bot_type, user_id, project_id, "data.txt")
    
#     if not os.path.exists(intents_file):
#         print(f"No intents file found for {key}")
#         return
    
#     with open(intents_file, "r", encoding="utf-8") as file:
#         intents_data = json.load(file)
    
#     data[key] = {
#         "intents": intents_data,
#         "intent_embeddings": None,
#         "knowledgebase_embeddings": None
#     }
    
#     if bot_type == "advanced" and os.path.exists(knowledgebase_file):
#         print(f"Loading knowledge base for {key} from {knowledgebase_file}")
#         loader = TextLoader(knowledgebase_file, encoding="utf-8")
#         documents = loader.load()
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=300,
#             chunk_overlap=20,
#             length_function=len,
#             is_separator_regex=False
#         )
#         chunks = text_splitter.split_documents(documents)
#         index_name = f"KB{user_id}{project_id}"
#         kb_embeddings = WeaviateVectorStore.from_documents(
#             documents=chunks,
#             embedding=embedding,
#             client=weaviate_client,
#             index_name=index_name
#         )
#         data[key]["knowledgebase_embeddings"] = kb_embeddings


# # ============================================================================
# # BOT LOGIC
# # ============================================================================
# def basic_bot(user_id, project_id, query):
#     key = f"{user_id}_{project_id}"
#     intents_data = data.get(key, {}).get("intents", {})
    
#     if not intents_data:
#         return "fallback"
    
#     patterns, tags = [], []
#     for tag, val in intents_data.items():
#         for pattern in val.get("patterns", []):
#             patterns.append(pattern)
#             tags.append(tag)
    
#     if not patterns:
#         return "fallback"
    
#     pattern_emb = model.encode(patterns, convert_to_tensor=True, device=device)
#     query_emb = model.encode(query, convert_to_tensor=True, device=device)
#     scores = util.cos_sim(query_emb, pattern_emb)[0]
#     max_score, max_idx = scores.max().item(), scores.argmax().item()
    
#     return tags[max_idx] if max_score >= 0.6 else "fallback"


# def fallback_bot(chat_history, user_id, project_id):
#     key = f"{user_id}_{project_id}"
#     kb_embeddings = data[key]["knowledgebase_embeddings"]
    
#     if not kb_embeddings:
#         return "Sorry, no fallback knowledge base available."
    
#     system_prompt = """
#     You are a friendly and helpful chatbot for the company.
#     Answer the users questions only using the context provided.
#     Always maintain conversational tone.
#     Keep your answers small, simple, direct.
#     Avoid fabricating the answer.
#     Strictly do not answer the question if it is out of the context, reply politely as you can not provide answer for that query, if needed ask them to contact us.
#     "\n\n"
#     "{context}"""
    
#     prompt_messages = [("system", system_prompt)]
#     for msg in chat_history:
#         if "human" in msg:
#             prompt_messages.append(("human", msg["human"]))
#         elif "ai" in msg:
#             prompt_messages.append(("ai", msg["ai"]))
    
#     prompt = ChatPromptTemplate.from_messages(prompt_messages)
#     llm = ChatOpenAI(model="gpt-4.1-nano-2025-04-14")
#     document_chain = create_stuff_documents_chain(llm, prompt)
#     retriever = kb_embeddings.as_retriever()
#     chain = create_retrieval_chain(retriever, document_chain)
    
#     last_user_msg = next(reversed([m["human"] for m in chat_history if "human" in m]), "")
#     retrieved_docs = retriever.invoke(last_user_msg)
    
#     print("==== Retrieved Context =================")
#     for doc in retrieved_docs:
#         print(doc.page_content)
#         print()
#     print("===========================")
    
#     with get_openai_callback() as cb:
#         result = chain.invoke({"input": last_user_msg})
    
#     return result["answer"]


# def advd_bot(user_id, project_id, chat_history):
#     key = f"{user_id}_{project_id}"
    
#     if key not in data:
#         return "No data found for this bot."
    
#     intents = data[key]["intents"]
#     trimmed_history = chat_history[-10:] if len(chat_history) > 10 else chat_history
#     last_user_msg = next(reversed([m["human"] for m in trimmed_history if "human" in m]), "")
    
#     tag = basic_bot(user_id, project_id, last_user_msg)
    
#     if tag == "fallback":
#         answer = fallback_bot(trimmed_history, user_id, project_id)
#         return {
#             "patterns": [],
#             "responses": [answer],
#             "questions": [],
#             "custom_payloads": [],
#             "form_status": False,
#             "form_details": [],
#             "api_call": False,
#             "api_details": {"url": "", "method": "POST"},
#         }
    
#     return intents.get(tag, intents.get("fallback", {}))


# # ============================================================================
# # MAIN EXECUTION
# # ============================================================================
# if __name__ == "__main__":
#     user_id = "9"
#     project_id = "1"
#     type_bot = "advanced"
    
#     update_user_project_data(user_id, project_id, type_bot)
    
#     if not weaviate_client.is_ready():
#         print("Weaviate is not ready. Exiting.")
#         exit(1)
    
#     try:
#         chat_history = []
#         while True:
#             user_input = input("You: ")
#             chat_history.append({"human": user_input})
#             response = advd_bot(user_id, project_id, chat_history)
#             print("Bot:", response, "/n/n")
#             chat_history.append({"ai": response["responses"][0]})
#     finally:
#         weaviate_client.close()
#         print("Connection to Weaviate closed.")  










# from pydantic_ai import Agent, RunContext
# from dotenv import load_dotenv 
# load_dotenv() 



# roulette_agent = Agent(  
#     'openai:gpt-4o',
#     deps_type=int,
#     output_type=bool,
#     system_prompt=(
#         'Use the `roulette_wheel` function to see if the '
#         'customer has won based on the number they provide.'
#     ),
# )


# @roulette_agent.tool
# async def roulette_wheel(ctx: RunContext[int], square: int) -> str:  
#     """check if the square is a winner"""
#     return 'winner' if square == ctx.deps else 'loser'


# # Run the agent
# success_number = 18  
# result = roulette_agent.run_sync('Put my money on square eighteen', deps=success_number)
# print(result)
# print(result.output)  
# #> True

# result = roulette_agent.run_sync('I bet five is the winner', deps=success_number)
# print(result.output)
# #> False 


import os
import json
import torch
import tiktoken
import weaviate 
from sentence_transformers import SentenceTransformer, util
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.callbacks.manager import get_openai_callback
from langchain_weaviate import WeaviateVectorStore 
from dotenv import load_dotenv

load_dotenv()

txt_path = "website.txt" 
embedding = OpenAIEmbeddings()
weaviate_client = weaviate.connect_to_local()

def create_docs_embeddings(txt_path):
    loader = TextLoader(txt_path, encoding='utf-8') 
    docs = loader.load() 
    
    text_splitter = RecursiveCharacterTextSplitter( 
        chunk_size = 200,
        chunk_overlap =50, 
    ) 

    chunks = text_splitter.split_documents(docs)
    index_name = "WebsiteData" 
    web_embeddings = WeaviateVectorStore.from_documents(
        documents= chunks,
        embedding=embedding,
        client=weaviate_client,
        index_name=index_name
    )

    return web_embeddings  

def retrival_docs(vectorstore, query):
    retriever = vectorstore.as_retriever()
    docs = retriever.invoke(query)
    return docs


def load_existing_vectorstore():
    index_name = "WebsiteData"

    vectorstore = WeaviateVectorStore(
        client=weaviate_client,
        index_name=index_name,
        text_key="text",
        embedding=embedding,
    )

    return vectorstore


if __name__ == "__main__": 

    txt_path = "website.txt"  
    if not weaviate_client.is_ready():
        print("Weaviate is not ready. Exiting.")
        exit(1)

    # web_embeddings = create_docs_embeddings(txt_path)
    # print("Embeddings created and stored in Weaviate.")  

    web_embeddings = load_existing_vectorstore()

    docs = retrival_docs(web_embeddings, "Contact Us ")
    for doc in docs:
        print(doc.page_content)

    weaviate_client.close()
    print("Connection to Weaviate closed.")  


