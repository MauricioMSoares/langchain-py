import os
import json
os.environ["OPENAI_API_KEY"] = ""
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai.chat_models import ChatOpenAI


def perform_arithmetic(first_num: int, second_num: int, operation: str) -> int:
    """Perform arithmetic on two numbers.

    Args:
        first_num: first number in the arithmetic operation. If dividing, this would be the numerator.
        second_num: second number in the arithmetic operation. If dividing, this would be the denominator.
        operation: one of (add, subtract, multiply, divide)
    """

    if operation == "add":
        return first_num + second_num
    elif operation == "subtract":
        return first_num - second_num
    elif operation == "multiply":
        return first_num * second_num
    elif operation == "divide":
        return first_num / second_num
    else:
        return f"The following operation is not allowed: {operation}"
    
arithmetic_function = convert_to_openai_tool(perform_arithmetic)
print(json.dumps(arithmetic_function, indent=4))

regular_model = ChatOpenAI(model="gpt-3.5-turbo")
regular_model.invoke("What is 910751 * 81301589?") # Returns an incorrect answer

model_with_tool = regular_model.bind(
    tools=[arithmetic_function],
)

result = model_with_tool.invoke("What is 910751 * 81301589?")
print(result)

args = result.additional_kwargs["tool_calls"][0]["function"]["arguments"]
function_name = result.additional_kwargs["tool_calls"][0]["function"]["name"]

print("Arguments: ", args)
print("Function name: ", function_name)

refined_args = json.loads(args)
answer = perform_arithmetic(**refined_args)
print(f"ANSWER: {answer}")
