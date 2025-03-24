# Standard library imports
import os
import sys
import json
from itertools import combinations

from tqdm import tqdm
from openai import OpenAI  

from .prompts import * 

def create_prompt(anchor, query1, query2, reasoning=False, query_to_dict=None, prompt_template=PROMPT_TEMPLATE):
    prompt = prompt_template.format(anchor=query_to_dict[anchor], query1=query_to_dict[query1], query2=query_to_dict[query2])

    if reasoning:
        prompt += "Reasoning: [provide a detailed explanation for your answer, discussing the semantic similarities and differences between the queries]"
    return prompt

def call_chatgpt(client, prompt, system_message=SYSTEM_MESSSAGE):
    messages = [
        {
            "role": "system", 
            "content": system_message},
        {
            "role": "user", 
            "content": [ 
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ] 

    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=messages,
        temperature=0.5,
        max_tokens=2000
    )

    return response
 
def calculate_cost(completion_tokens, prompt_tokens):
    completion_cost = 7.5 / 1e6
    prompt_cost = 2.5 / 1e6
    cost = round(completion_cost * completion_tokens + prompt_cost * prompt_tokens, 4)
    return cost

def handle_response(completion, reasoning=False, query1=None, query2=None):
    if len(completion.choices) > 0:
        content = completion.choices[0].message.content

        answer = None
        cost = calculate_cost(completion.usage.completion_tokens, completion.usage.prompt_tokens)

        # print('cost', cost)
        try:
            answer = int(content.split("\n")[0][8:][-1])
            answer = query1 if answer == 1 else query2

            try:
                reason = ' '.join(content.split("\n")[1:])[11:] # if reasoning else None
            except:
                reason = None
            
            return {'similarity': answer, 'reasoning': reason, "cost": cost}
        except ValueError:
            print('invalid gpt answer', content.split("\n")[0][8:][-1])
            return None
    else:
        return None
    
def get_llm_similarity(anchor, query1, query2, reasoning=False, query_to_dict=None, prompt_template=PROMPT_TEMPLATE):
    client = OpenAI(api_key=os.environ['OPENAI_API_KEY']) 

    prompt = create_prompt(anchor, query1, query2, reasoning, query_to_dict, prompt_template)
    response = None

    num_tries = 0
    while response == None and num_tries < 10:
        completion = call_chatgpt(client, prompt)
        response = handle_response(completion, reasoning, query1, query2)
        num_tries += 1
    
    return response

def compute_llm_similarities(queries, reasoning=False, filename=None, dataset='calvin', query_to_dict=None, prompt_template=PROMPT_TEMPLATE):
    filename = 'llm_similarities.json' if filename is None else filename
    pathname = f'/vision/u/emilyjin/abstractions/end_to_end/similarities/{dataset}/{filename}'
    os.makedirs(f'/vision/u/emilyjin/abstractions/end_to_end/similarities/{dataset}', exist_ok=True)

    if os.path.exists(pathname):
        similarities = json.load(open(pathname, 'r'))
    else:
        similarities = {}
 
    for anchor in queries:
        if anchor not in similarities.keys():
            similarities[anchor] = {}

    for anchor in queries:
        print('anchor', anchor)
        other_queries = [query for query in queries if query != anchor] 

        cost = 0
        count = 0

        for query1, query2 in tqdm(combinations(other_queries, 2), total=len(other_queries) * (len(other_queries)) // 2):
            pair = sorted([query1, query2])

            if not anchor in similarities.keys() or '//'.join(pair) not in similarities[anchor].keys():
                response = get_llm_similarity(anchor, pair[0], pair[1], reasoning, query_to_dict, prompt_template)
                similarities[anchor]['//'.join(pair)] = response

                if response is not None:
                    cost += response['cost']
                    count += 1

                if count % 5 == 0:
                    print('cost so far after', count, ': ', cost)
                    json.dump(similarities, open(pathname, 'w'), indent=4)

    json.dump(similarities, open(pathname, 'w'), indent=4)
    return similarities

