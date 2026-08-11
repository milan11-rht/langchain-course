from dotenv import load_dotenv
load_dotenv()
from pydantic import BaseModel,Field
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_deepseek import ChatDeepSeek
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI
import os

class Source(BaseModel):
    
    url: str = Field(..., description="The URL of the source")
    
class AgentResponse(BaseModel):
    answer: str = Field(..., description="The answer to the user's query")
    sources: list[Source] = Field(..., description="A list of sources used to generate the answer")


@tool
def search(query: str) -> str:
    """
    Tool that searches over Internet
    Args:
        query : The query to search for
    Returns:
        The search results
    """
    print(f"Searching for: {query}")
    # return "Tokyo weather is sunny and warm, with a high of 25°C and a low of 18°C."
    return tavily.search(query=query)


# llm = ChatDeepSeek(
#     api_key=os.getenv("DEEPSEEK_API_KEY"),
#     model="deepseek-v4-pro",
#     temperature=0,
# )

llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-5.4-mini",
    temperature=0,
)
tools = [TavilySearch(api_key=os.getenv("TAVILY_API_KEY"))]
agent = create_agent(model=llm, tools=tools)

def main():
    print("hello")
    result = agent.invoke({"messages": HumanMessage(content="find the jobs related to senior data engineer hybrid in india with 6 years of experience and salary 30 lpa - 50 lpa")})
    print(result)


if __name__ == "__main__":
    main()