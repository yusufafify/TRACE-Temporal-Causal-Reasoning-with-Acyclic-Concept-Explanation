from mistralai import Mistral
import openai
import os
from env import OPENAI_API_KEY

class llm_client:
    def __init__(self, LLM='gpt-4o', temperature=0, max_tries=1):
        
        self.LLM = LLM
        self.temperature = temperature
        self.max_tries = max_tries
        
        if 'gpt' in self.LLM:     
            openai.api_key = OPENAI_API_KEY
        elif 'mistral' in self.LLM or 'mixtral' in self.LLM:
            api_key = os.environ.get("MISTRALAIKEY")
            self.client = Mistral(api_key=api_key)
        else:                     
            raise ValueError(f"model ({self.LLM}) still not implemented")
                                  
    def invoke(self, question, temperature=0):
        if 'gpt' in self.LLM:
            response = openai.chat.completions.create(
                model=self.LLM,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": question},
                ],
                temperature=temperature
            )
            answer = response.choices[0].message.content
        elif 'mistral' in self.LLM or 'mixtral' in self.LLM:
            chat_response = self.client.chat.complete(
                model = self.LLM,
                messages = [
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": question},
                ],
                temperature=temperature
            )
            answer = chat_response.choices[0].message.content
        else:
            raise ValueError("model still not implemented")
        return answer
    
    def parse_answer(self, answer):
        start_tag = "<answer>"
        end_tag = "</answer>"
        start_index = answer.find(start_tag) + len(start_tag)
        end_index = answer.find(end_tag)
        if start_index != -1 and end_index != -1:
            answer = answer[start_index:end_index].strip()
        return answer

    # This function is intended to be used only for asking the model to provide the causal relationship between two concepts
    def invoke_with_retry(self, question):
        tries = 0
        answers = {}
        while tries < self.max_tries:
            answer = self.invoke(question, self.temperature)
            answer = self.parse_answer(answer)
            if answer in answers.keys():
                answers[answer] += 1
            else:
                answers[answer] = 1
            tries += 1
        return max(answers, key=answers.get)