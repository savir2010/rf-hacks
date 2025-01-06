import openai
import tempfile
import playsound

openai.api_key = "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"




def quiz_response(user_input):
    prompt = f"""
You are a quiz generator. Given a topic, you will generate one question and four multiple-choice answers. The correct answer should be placed in parentheses, and the answer choices should be separated by a hyphen ("-"). Output the question and choices only, with no additional text.

Example:

Input: Variables
Output: What is a variable? (Storage)-Computer-Math-Data

    """

    # Using messages to structure the conversation
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",  # Or "gpt-4" depending on your access
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_input}
        ],
        max_tokens=200
    )
    
    str_response = response['choices'][0]['message']['content']
    question_part, options_part = str_response.split("?", 1)

    question = question_part.strip() + "?"
    options = options_part.split("-")
    correct_answer = options[0].strip(" ()")  # Removing parentheses around the correct answer
    choice1 = options[1].strip()
    choice2 = options[2].strip()
    choice3 = options[3].strip()
    quiz_lst = [question,correct_answer,choice1,choice2,choice3]
    return quiz_lst



print(quiz_response("World War 2"))