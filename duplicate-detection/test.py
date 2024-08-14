import os
from dotenv import load_dotenv
from langchain_openai.chat_models import AzureChatOpenAI

load_dotenv(dotenv_path="concept-summary/given-text/.env")

model = AzureChatOpenAI(
    azure_deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT"),
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION")
)

ai_msg = model.invoke("""Formuliere den folgenden Absatz in anderen Worten, ohne die Bedeutung zu verändern:
             Aufgrund unserer Organisation sind kurze Wege und schnelle Reaktionszeiten garantiert. Im Bedarfsfall kann auf weitere Experten-Netzwerke der Westnetz oder Regionetz und auf das Know-how weiterer regionaler Netzservicestellen zugreifen können.
In diesem Netzwerk findet zusätzlich ein regelmäßiger Austausch der technischen Mitarbeiter, Meister und Ingenieure statt, so dass in der Praxis schnell Lösungen für sämtliche Herausfor-derungen gefunden werden können.""")

print(ai_msg)