from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing_extensions import Self
from langchain.prompts import PromptTemplate
from abc import ABC


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
                vehicles.append(Vehicle(des))

        return vehicles


class LLM(ABC):
    template = PromptTemplate(template="Prompt:\nYou are an AI chat assistant designed to assist customers with their inquiries about the cars available at a specific dealership. You are given a list of the cars and the user's prompt.\n\nCars: \n{vehicles}\n\nUser prompt: {user_prompt}\n\nYour Response :", input_variables=[
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
        print(docs)
        print(len(docs))
        input = {"vehicles": [
            doc.page_content for doc in docs], "user_prompt": prompt}
        print(self.template.invoke(input))
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
