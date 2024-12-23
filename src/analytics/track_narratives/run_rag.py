import numpy as np
import faiss
import h5py
from typing import List, Tuple
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
import argparse
import os


# Load the merged embeddings
def load_embeddings(file_path: str) -> Tuple[np.ndarray, List[str]]:
    with h5py.File(file_path, "r") as f:
        embeddings = f["dense_embeddings"][:]
        filepaths = [path.decode("utf-8") for path in f["filepaths"][:]]
    return embeddings, filepaths


# Initialize FAISS index
def init_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    faiss.normalize_L2(embeddings)  # Normalize vectors before adding to the index
    index.add(embeddings)
    return index


# Custom retriever class
class CustomRetriever:
    def __init__(
        self,
        index: faiss.IndexFlatIP,
        embeddings: np.ndarray,
        filepaths: List[str],
        k: int = 4,
    ):
        self.index = index
        self.embeddings = embeddings
        self.filepaths = filepaths
        self.k = k
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
        self.model = AutoModel.from_pretrained("BAAI/bge-m3").to(self.device)

    def get_relevant_documents(self, query: str) -> List[str]:
        encoded_input = self.tokenizer(
            query, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            embedding = model_output.last_hidden_state[:, 0, :].cpu().numpy()
        faiss.normalize_L2(embedding)
        distances, indices = self.index.search(embedding, self.k)
        return [self.filepaths[i] for i in indices[0]]


# Load embeddings and initialize FAISS index
embeddings, filepaths = load_embeddings("./merged_temp.h5")
index = init_faiss_index(embeddings)

# Initialize custom retriever
retriever = CustomRetriever(index, embeddings, filepaths, k=250)

# Initialize a more powerful language model
model_name = "meta-llama/Llama-3.1-8B-Instruct"
root_path = "/vast/gm2724/transcripts_summarized"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
model.to("cuda")

# Create a pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=1024,
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
    device="cuda",
)

# Wrap the pipeline in a LangChain LLM
llm = HuggingFacePipeline(pipeline=pipe)

# Define RAG prompt template
RAG_TEMPLATE = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

<context>
{context}
</context>

Question: {question}

Answer:"""

rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)


# Define document formatting function
def format_docs(docs: List[str]) -> str:
    print("Docs used:", docs)
    docs = [
        doc + ":\n\n" + open(f"{root_path}/{doc.split('_')[1]}/{doc}").read()
        for doc in docs
    ]
    return "\n\n".join(docs)


# New function to dump retrieved documents
def dump_retrieved_docs(query: str, docs: List[str], root_path: str):
    dump_dir = "dumped_docs"
    os.makedirs(dump_dir, exist_ok=True)

    file_name = f"{dump_dir}/{query[:50].replace(' ', '_')}.txt"

    with open(file_name, "w", encoding="utf-8") as f:
        f.write(f"Query: {query}\n\n")
        for doc in docs:
            f.write(f"Document: {doc}\n")
            with open(
                f"{root_path}/{doc.split('_')[1]}/{doc}", "r", encoding="utf-8"
            ) as doc_file:
                content = doc_file.read()
            f.write("Content:\n")
            f.write(content)
            f.write("\n\n" + "-" * 50 + "\n\n")

    print(f"Dumped retrieved documents to {file_name}")


def get_relevant_docs(query: str) -> List[str]:
    docs = retriever.get_relevant_documents(query)
    dump_retrieved_docs(query, docs, root_path)
    return docs


rag_chain = (
    {
        "context": RunnablePassthrough()
        | RunnableLambda(get_relevant_docs)
        | RunnableLambda(format_docs),
        "question": RunnablePassthrough(),
    }
    | rag_prompt
    | llm
    | StrOutputParser()
)


# Function to chat with the RAG system
def chat_with_rag(k: int):
    global retriever
    retriever.k = k  # Update the k value

    print(
        f"Welcome to the RAG chat system. Using k={k}. Type 'exit' to end the conversation."
    )
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        response = rag_chain.invoke(user_input)
        print(f"\nAssistant: {response}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Chat System")
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of documents to retrieve (default: 5)",
    )
    args = parser.parse_args()

    # Initialize custom retriever with the specified k
    retriever = CustomRetriever(index, embeddings, filepaths, k=args.k)

    chat_with_rag(args.k)
