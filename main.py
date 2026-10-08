import os
import certifi
import requests

from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.tools import tool
from langchain_tavily import TavilySearch
from langchain.agents import create_agent


# ==========================================
# LOAD ENV VARIABLES
# ==========================================

os.environ["SSL_CERT_FILE"] = certifi.where()
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")


# ==========================================
# SEARCH TOOL
# ==========================================

search_tool = TavilySearch(
    max_results=2
)


# ==========================================
# WEATHER TOOL
# ==========================================

@tool
def get_weather_data(city: str) -> str:
    """
    Fetch current weather information for a city.
    """

    url = (
        f"https://api.weatherstack.com/current?"
        f"access_key={WEATHERSTACK_API_KEY}&query={city}"
    )

    response = requests.get(url)

    data = response.json()

    if "current" not in data:
        return f"Could not fetch weather data for {city}"

    return (
        f"City: {city}\n"
        f"Temperature: {data['current']['temperature']}°C\n"
        f"Weather: {data['current']['weather_descriptions'][0]}\n"
        f"Humidity: {data['current']['humidity']}%"
    )
    

# ==========================================
# Calculator Tool
# ==========================================

import ast
import operator


# Supported mathematical operations
OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


def evaluate_expression(node):
    """Safely evaluate a mathematical expression."""

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Only numbers are allowed.")

    if isinstance(node, ast.BinOp):
        left = evaluate_expression(node.left)
        right = evaluate_expression(node.right)

        operation = OPERATORS.get(type(node.op))

        if operation is None:
            raise ValueError("Unsupported operator.")

        return operation(left, right)

    if isinstance(node, ast.UnaryOp):
        operand = evaluate_expression(node.operand)

        operation = OPERATORS.get(type(node.op))

        if operation is None:
            raise ValueError("Unsupported operator.")

        return operation(operand)

    raise ValueError("Invalid mathematical expression.")


@tool
def calculator(expression: str) -> str:
    """
    Calculate mathematical expressions.

    Examples:
    - 10 + 20
    - 100 * 1.18
    - (500 + 200) / 2
    - 2 ** 10

    Use this tool whenever an accurate numerical calculation is required.
    """

    try:
        tree = ast.parse(expression, mode="eval")

        result = evaluate_expression(tree.body)

        return str(result)

    except ZeroDivisionError:
        return "Error: Cannot divide by zero."

    except Exception as e:
        return f"Calculation error: {str(e)}"


# ==========================================
# LLM
# ==========================================

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=GROQ_API_KEY
)


# ==========================================
# TOOLS
# ==========================================

tools = [
    search_tool,
    get_weather_data,
    calculator
]


# ==========================================
# CREATE AGENT
# ==========================================

agent = create_agent(
    model=llm,
    tools=tools
)


# ==========================================
# RUN
# ==========================================

response = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "What is the Population of country whose economical capital is Mumbai. Also add that number to the population of China and give final answer."
        }
    ]
})


# ==========================================
# OUTPUT
# ==========================================

print("\n========================")
print("FINAL OUTPUT")
print("========================\n")

print(response["messages"][-1].content)