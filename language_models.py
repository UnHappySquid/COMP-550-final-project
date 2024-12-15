from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing_extensions import Self
from langchain.prompts import PromptTemplate


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


class NaiveLLM:
    def __init__(self, vehicles: list[Vehicle]):
        self.vehicles = vehicles
        template = PromptTemplate(template="Prompt:\nYou are an AI chat assistant designed to assist customers with their inquiries about the cars available at a specific dealership. You are given a list of the cars and the user's prompt.\n\nCars: \n{vehicles}\n\nUser prompt: {user_prompt}\n\nYour Response :", input_variables=[
                                  "vehicles", "user_prompt"])
        llm_model = ChatOllama(model="llama3", temperature=0)
        self.model = template | llm_model

    def prompt(self, prompt: str) -> str:
        vehicle_str = "\n".join([str(vehicle) for vehicle in self.vehicles])
        return self.model.invoke(
            {"vehicles": vehicle_str, "user_prompt": prompt}).content


class RAG:

    def __init__(self, vehicles: list[Vehicle]):
        pass

    def prompt(self, prompt: str) -> str:
        pass


def main() -> None:
    vehicles = Vehicle.load_vehicles("vehicles.txt")
    naive_llm = NaiveLLM(vehicles)
    while True:
        prompt = input("> ")
        if prompt == "q":
            break
        print(naive_llm.prompt(prompt))


if __name__ == "__main__":
    main()
