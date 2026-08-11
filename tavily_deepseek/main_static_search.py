from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_deepseek import ChatDeepSeek
from langchain_ollama import ChatOllama
import os

load_dotenv()   




def main():
    print(os.getenv("DEEPSEEK_API_KEY"))
    print("Hello from langchain-course!")

    information = """
    William Henry Gates III (born October 28, 1955) is an American businessman and 
    philanthropist. A pioneer of the microcomputer revolution of the 1970s and 
    1980s, he co-founded the software company Microsoft in 1975 with his childhood 
    friend Paul Allen. Following Microsoft's initial public offering in 1986 and 
    the subsequent increase in its stock price, Gates became the world's 
    then-youngest billionaire in 1987, at age 31. Forbes magazine ranked 
    him as the world's wealthiest person in their The World's Billionaires list for 
    18 out of 24 years between 1995 and 2017, including 13 years consecutively from 
    1995 to 2007. Gates became the first centibillionaire in 1999, when his net worth briefly surpassed US$100 billion. According to Forbes, as of July 2026, his net worth stood at US$106.5 billion, making him the 19th-wealthiest individual in the world.
    """
    summary_template = """
    given the information {information}, summarize it in 3 bullet points and 3 interesting facts in bullet points.
    """

    summary_prompt_template = PromptTemplate(
        template=summary_template, input_variables=["information"]
    )

    llm = ChatDeepSeek(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        model="deepseek-v4-pro",temperature=0,
    )

    # llm = ChatOllama(
    #     model="gpt-oss:latest",
    #     temperature=0,
    # )
    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information": information})

    print(response.content)


if __name__ == "__main__":
    main()
