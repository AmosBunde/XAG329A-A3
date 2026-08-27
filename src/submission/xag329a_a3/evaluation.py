import re
from typing import Literal, Optional
from xag329a_a3.utils import generate_llm

MATH = [ 
    ("A random variable X has probability density function proportional to cos(x)/sqrt(x) from 1 to 2. What is the normalization constant should this be multiplied by to make this a valid PDF? Round your answer to the nearest integer.", 13),
    ("Find the real solution to x^3 = x^5 + 100. Give your answer to the nearest hundredth.", -2.59),
    ("Calculate the area of a rectangle with length 123.45 cm and width 67.89 cm, then convert the result to square inches to the nearest integer.", 1299)
]

GOOGLE = [
    ("What 2024 Beyoncé album broke streaming records with its blend of country-pop and Americana?", "Cowboy Carter"),
    ("What is the title of the next Avatar film, scheduled for release on December 19, 2025?", "Fire and Ash"),
    ("How many restaurants in San Francisco have three Michelin stars in 2025?.", "Three"),
    ("What is the name of the restaurant in Palo Alto with a Michelin star in 2025?", "Protégé"),
    ("When does Zareen's (Palo Alto) open on Saturday? Format your answer as HH:MM AM/PM", "11:00 AM"),
    
    ("What nationality is the player who won the 2025 men's Wimbledon championship?", "Italian"),
]

WEATHER = [
    ("Were there any thunderstorms in Seattle, WA at noon on June 1st, 2025?", "No"),
    ("Which city was windier on August 1, 2025: Chicago or Boston?", "Boston"),
    ("Over the first three days of August 2025, did the morning (9AM) temperature in Miami ever drop below 70 Fahrenheit?", "No"),
]

STOCK = [
    ("Did Tesla (TSLA) stock go up or down on March 5, 2025 compared to its opening price on that day?", "up"),
    ("From 4 Nov 2024 to 6 Nov 2024 inclusive (the two days around the U.S. election), what was the change in Tesla closing stock price, rounded to the nearest percent?", 19),
    ("If Tesla's closing price on Aug 27, 2025 increased by 11.2%, what would that be, to the nearest cent?", 388.76),
]

def prepare_dataset(
    max_rows: Optional[int] = None,
    debug_mode: bool = False,
    question_type: Literal["math", "google", "weather", "stock", "all"] = "all"
):
    if question_type == "math":
        dataset = MATH[:2] if debug_mode else MATH
    elif question_type == "google":
        dataset = GOOGLE[:2] if debug_mode else GOOGLE
    elif question_type == "weather":
        dataset = WEATHER[:2] if debug_mode else WEATHER
    elif question_type == "stock":
        dataset = STOCK[:2] if debug_mode else STOCK
    else:
        dataset = WEATHER[:2] + STOCK[:2] + MATH[:2] + GOOGLE[:2] if debug_mode else WEATHER + STOCK + MATH + GOOGLE

    if max_rows:
        dataset = dataset[:max_rows]

    return {"query": [item[0] for item in dataset], "answer": [item[1] for item in dataset]}


# The judge is grading short answers against short ground truths, so it has to be
# told that a longer official title or an explanation around the answer still
# counts, while an unresolved answer does not.
JUDGE_SYSTEM_PROMPT = (
    "You are grading a question-answering system. You are given a question, a response, "
    "and the ground truth answer. Decide whether the response conveys the ground truth answer.\n"
    "Grade it 'correct' when the response contains the ground truth answer, even if it also "
    "restates the question, explains its reasoning, adds units, or uses a longer official name "
    "or title that contains the ground truth answer.\n"
    "Grade it 'incorrect' when the response gives a different value, contradicts the ground "
    "truth, omits the answer, or leaves the answer unresolved, such as returning an unevaluated "
    "expression or reporting that it could not determine the answer.\n"
    "Reply with only 'correct' or 'incorrect'."
)


def _judge_verdict(content: str) -> Optional[bool]:
    """Reads the judge's verdict, or None when it did not state one.

    Reasoning models can wrap the verdict in a sentence, so the whole reply is
    searched rather than only its first word. "incorrect" is checked first
    because it contains "correct".
    """
    verdict = content.strip().lower()
    if re.search(r"\bincorrect\b", verdict):
        return False
    if re.search(r"\bcorrect\b", verdict):
        return True
    return None


def evaluate_qa(query: str, response: str, answer: str, model: str | None = None):
    """
    Evaluate QA performance by checking if the response is correct.
    
    If model is not provided, the response will be checked using a simple word-by-word comparison.
    
    Args:
        queries (List[str]): List of query strings
        responses (List[str]): List of model response strings
        answers (List[str]): List of ground truth answer strings
        model (str | None): The model to use for evaluation (default is None)
    
    Returns: 
        bool: True if the response is correct, False otherwise
    """
    if model:
        judgement = generate_llm(
            model=model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": f"Question: {query}\nResponse: {response}\nGround Truth Answer: {answer}."}
            ]
        )
        if judgement:
            verdict = _judge_verdict(judgement)
            if verdict is not None:
                return verdict
            print(f"Judge model {model} gave an unreadable verdict; using word comparison.")
            print(f"\tVerdict: {judgement!r}")
        else:
            # generate_llm returns None when retries are exhausted. Scoring this
            # incorrect would let a rate limit understate accuracy, so fall through
            # to the word-overlap comparison below.
            print(f"Judge model {model} did not respond; using word comparison.")

    answer_words = set(word.strip('.,!?()[]{}":;') for word in str(answer).lower().split())
    haystack = str(response).lower()
    # Match whole words only: a substring test scores "No" correct against "not"
    # and 13 correct against 130. \b cannot be used because answers may begin
    # with a non-word character, as in -2.59.
    return all(
        re.search(rf"(?<!\w){re.escape(word)}(?!\w)", haystack)
        for word in answer_words
        if word
    )
    
    
