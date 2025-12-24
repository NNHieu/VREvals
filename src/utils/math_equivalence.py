import re
from math_verify import parse, verify

def extract_boxed_answer(output, mode='gen'):
    if "</think>" in output:
        output = output.split("</think>")[1]
    extracted_text = ''

    # Existing extraction logic for 'gen' and 'choose' modes
    pattern = r'\\boxed\{(.*)\}'
    matches = re.findall(pattern, output)
    if matches:
        extracted_text = matches[-1]  # Take the last match
        if mode in ['choose', 'qa']:
            # Handle 'choose' mode
            # inner_pattern = r'\\text\{(.*)\}'
            # inner_matches = re.findall(inner_pattern, extracted_text)
            # if inner_matches:
            #     extracted_text = inner_matches[-1]  # Take the last match
            # extracted_text = extracted_text.strip("()")
            raise ValueError
    return extracted_text

def is_math_equiv(gold, answer):
    gold = parse(gold)
    answer = parse(answer)
    return verify(gold, answer)