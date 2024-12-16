from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import MergerRetriever
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain_community.document_transformers import EmbeddingsRedundantFilter

from colorama import Fore, Style
from langgraph.graph import END, StateGraph, START
from abc import ABC
from typing_extensions import TypedDict, List
from typing_extensions import Self
from pprint import pprint


class Vehicle:

    def __init__(self, description: str):
        self.description = description

    def __str__(self):
        return self.description

    @staticmethod
    def load_vehicles(vehicles_path: str) -> list[Self]:
        vehicles = []
        with open(vehicles_path, "r") as fp:
            for des in fp:
                des = des.rstrip("\n")
                vehicles.append(Vehicle(des))

        return vehicles


class LLM(ABC):
    template = PromptTemplate(template="""You are an assistant for question-answering tasks at a car dealership. Your job is to answer client's inquiries. Use the following car descriptions to answer the question. Use three sentences maximum and keep the answer concise.
Cars:

{vehicles}


Client's question:


{user_prompt}


Answer:""", input_variables=[
                              "vehicles", "user_prompt"])


class NaiveLLM(LLM):
    def __init__(self, vehicles: list[Vehicle], model_name: str = "llama3"):
        self.vehicles = vehicles
        llm_model = ChatOllama(model=model_name, temperature=0)
        self.model = self.template | llm_model

    def prompt(self, prompt: str) -> str:
        vehicle_str = "\n".join([str(vehicle) for vehicle in self.vehicles])
        return self.model.invoke(
            {"vehicles": vehicle_str, "user_prompt": prompt}).content


class SimpleRAG(LLM):

    class State(TypedDict):
        question: str
        context: List[str]
        answer: str
        confident: bool

    def __init__(self, vehicles: list[Vehicle], model_name: str = "llama3",
                 embedding_model_name: str = "llama3"):
        # vehicles as strings
        vehicles_str = [str(vehicle) for vehicle in vehicles]
        llm_model = ChatOllama(model=model_name, temperature=0.)

        number_instances_to_retrieve = 10
        # vectore store
        embedding_model = OllamaEmbeddings(model=embedding_model_name)
        retriever_vec = Chroma.from_texts(
            texts=vehicles_str, collection_name="rag_vehicles",
            embedding=embedding_model).as_retriever(
            search_kwargs={"k": number_instances_to_retrieve},
        )

        # keyword store
        retriever_key = BM25Retriever.from_texts(
            texts=vehicles_str, k=number_instances_to_retrieve)

        # retriever on different types of chains.
        lotr = MergerRetriever(retrievers=[retriever_vec, retriever_key])
        # filtering
        filter = LLMChainFilter.from_llm(llm_model)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=filter, base_retriever=lotr
        )

        # generations

        # model that filters the documents even further

        # model that will generate the answer
        gen_model = self.template | llm_model

        # nodes

        def retrieve(state: self.State):
            docs = compression_retriever.invoke(state["question"])
            # print("-"*50)
            # pprint(docs)
            # print("-"*50)
            return {"context": [doc.page_content for doc in docs]}

        def generate(state: self.State):
            input = {"vehicles": "\n".join(
                state["context"]), "user_prompt": state["question"]}
            return {"answer": gen_model.invoke(input).content}

        # graph
        workflow = StateGraph(self.State)
        workflow.add_node("retrieve", retrieve)
        workflow.add_node("gen", generate)

        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "gen")
        workflow.add_edge("gen", END)

        self.graph = workflow.compile()

    def prompt(self, prompt: str) -> str:
        return self.graph.invoke({"question": prompt})["answer"]


def main() -> None:
    vehicles = Vehicle.load_vehicles("vehicles.txt")
    # naive_llm = SimpleRAG(vehicles[:1])
    smart_llm = SimpleRAG(vehicles)
    naive_llm = NaiveLLM(vehicles)
    print(Fore.RED + "In red is the RAG model" + Style.RESET_ALL)
    print(Fore.GREEN + "In green is the naive model" + Style.RESET_ALL)
    while True:
        prompt = input("> ")
        if prompt == "q":
            break
        print(Fore.RED + smart_llm.prompt(prompt) + Style.RESET_ALL)
        print(Fore.GREEN + naive_llm.prompt(prompt) + Style.RESET_ALL)


if __name__ == "__main__":
    main()
