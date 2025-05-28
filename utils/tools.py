from langchain.tools import tool
import datetime
import requests

@tool
def get_current_date():
    """Get the current date and time."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def calculate(expression: str):
    """Evaluate a mathematical expression."""
    try:
        return eval(expression)
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"
@tool
def get_products_list():
    """Get list of products from automationexercise API."""
    try:
        url = "https://automationexercise.com/api/productsList"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            return f"Error: API returned status code {response.status_code}"
    except Exception as e:
        return f"Error calling API: {str(e)}"

def get_custom_tools():
    """Return a list of custom tools to use with the agent."""
    return [get_current_date, calculate, get_products_list]