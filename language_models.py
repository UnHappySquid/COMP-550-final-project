from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing_extensions import Self
from langchain.prompts import PromptTemplate
from abc import ABC
from typing_extensions import TypedDict, List
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
    template = PromptTemplate(template="You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {user_prompt}
Context: {vehicles}
Answer:", input_variables=[
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

    def __init__(self, vehicles: list[Vehicle], model_name: str = "llama3",
                 embedding_model_name: str = "llama3"):
        self.vehicles = vehicles
        vehicles_str = [str(vehicle) for vehicle in vehicles]
        embedding_model = OllamaEmbeddings(model=embedding_model_name)
        self.retriever = Chroma.from_texts(
            texts=vehicles_str, collection_name="rag_vehicles",
            embedding=embedding_model).as_retriever()
        llm_model = ChatOllama(model=model_name, temperature=0.)
        self.model = self.template | llm_model

    def prompt(self, prompt: str) -> str:
        docs = self.retriever.get_relevant_documents(prompt)
        pprint(docs)
        print(len(docs))
        input = {"vehicles": "\n".join([
            doc.page_content for doc in docs]), "user_prompt": prompt}
        pprint(self.template.invoke(input))
        return self.model.invoke(input).content


def main() -> None:
    vehicles = Vehicle.load_vehicles("vehicles.txt")
    naive_llm = SimpleRAG(vehicles)
    while True:
        prompt = input("> ")
        if prompt == "q":
            break
        print(naive_llm.prompt(prompt))


if __name__ == "__main__":
    main()
