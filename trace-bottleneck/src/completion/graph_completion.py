from tqdm import tqdm
import pandas as pd

import re
from collections import Counter
from tqdm import tqdm
import pandas as pd

import re
from collections import Counter
# from langchain_huggingface import HuggingFacePipeline
# from langchain.prompts import PromptTemplate
from transformers import pipeline
# from langchain_core.output_parsers import StrOutputParser

prompt_template = """
Role: You are an expert in causal inference and logical analysis. I will provide you with two concepts and you have to infer the causal relationship between them.
**Concept 1:** {concept_1} - {concept_1_description}
**Concept 2:** {concept_2} - {concept_2_description}

Then, use only your knowledge and, possibly, the context provided, if available, to determine which of the following options is most likely true:
(A) changing {concept_1} to certain values result in a change in {concept_2};
(B) changing {concept_2} to certain values result in a change in {concept_1};
(C) there is no direct causal or logical relationship between {concept_1} and {concept_2}.

The following information are extracted from recent and reliable sources:
{context}

The answer has to be enclosed within <answer> tags, and do not include any additional text or explanation (e.g. <answer>A</answer>).
"""

prompt_template_cot2 = """
Role: You are an expert in causal inference and logical analysis. I will provide you with two concepts and you have to infer the causal relationship between them.
**Concept 1:** {concept_1} - {concept_1_description}
**Concept 2:** {concept_2} - {concept_2_description}

Now, use your knowledge and, if available, the context provided, to determine which of the following options is the correct one:
(A) there is a causal relationship between {concept_1} and {concept_2}, i,e, {concept_1} has an influence on {concept_2}; alternatively, changing {concept_1} will lead to a change in {concept_2};
(B) there is a causal relationship between {concept_2} and {concept_1}, i,e, {concept_2} has an influence on {concept_1}; alternatively, changing {concept_2} will lead to a change in {concept_1};
(C) there is no causal relationship or reciprocal influeance between {concept_1} and {concept_2}.

The following information are extracted from recent and reliable sources:
{context}

Hint: Information could be scattered across the context. So, if you find at least one evidence that supports a causal relationship, then prioritize that.

The answer has to be enclosed within <answer> tags (e.g. <answer>A</answer>).
Analyze the problem step-by-step to ensure the final conclusion is accurate.
"""

prompt_template_cot = """
Role: You are an expert in causal inference and logical analysis. I will provide you with two concepts and you have to infer the causal relationship between them.
**Concept 1:** {concept_1} - {concept_1_description}
**Concept 2:** {concept_2} - {concept_2_description}

Now, use your knowledge and, if available, the context provided, to determine which of the following options is the correct one:
(A) changing {concept_1} to certain values result in a change in {concept_2};
(B) changing {concept_2} to certain values result in a change in {concept_1};
(C) there is no causal relationship or reciprocal influeance between {concept_1} and {concept_2}.

The following information are extracted from recent and reliable sources:
{context}

The answer has to be enclosed within <answer> tags (e.g. <answer>A</answer>).
Analyze the situation step-by-step to ensure the final conclusion is accurate.
"""

def parse_answer(answer):
    start_tag = "<answer>"
    end_tag = "</answer>"
    start_index = answer.find(start_tag) + len(start_tag)
    end_index = answer.find(end_tag)
    if start_index != -1 and end_index != -1:
        answer = answer[start_index:end_index].strip()
        return answer
    else:
        return None

def complete_graph(adj_matrix_pd,
                   label_descriptions,
                   llm_model,
                   rag,
                   cfg):
    """
    TODO: What if the LLM is not able to provide a valid answer for a pair of concepts? The arc remains uncertain.
    
    This function enhances the adjacency matrix of a causal graph produced by a causal structural learning algorithm,
    using a large language model (LLM) and retrieval-augmented generation techniques.
    Currently, the only "uncertain" concept pairs are those with entries [i, j] = -1 and [j, i] = -1 in the adjacency matrix.

    Args:
        _adj_matrix (pd.Dataframe): A dictionary containing the adjacency matrix of the causal graph, node labels, and node label descriptions.
        llm (torch.nn.Module): The large language model (LLM) to use for completion.
        llm_tokenizer (transformers.PreTrainedTokenizer): The tokenizer for the LLM.
        rag (RagRetriever): The retrieval-augmented generation model to use for context retrieval.
        cfg (dict): The configuration dictionary containing the settings for the completion process.

    Returns:
        ...
    """
    adj_matrix = adj_matrix_pd.values
    node_labels = list(adj_matrix_pd.index)

    print(f"old graph \n {adj_matrix}")

    # Initialize the pipeline
    '''
    llm_pipeline_sp = pipeline("text-generation", 
                               model=llm,
                               tokenizer=llm_tokenizer,
                               max_new_tokens=512,
                               return_full_text=False)
    llm_pipeline = HuggingFacePipeline(pipeline=llm_pipeline_sp)
    '''

    # define the output parser
    # output_parser = StrOutputParser()

    # create the llm chain
    # llm_chain = prompt | llm_pipeline | output_parser

    # find pairs of concepts to analyze, i.e., concepts c1 and c2 such that c1 <-> c2 in the incomplete_graph
    problematic_conceptPairs = []
    n = len(adj_matrix)
    for i in range(n):
        for j in range(i + 1, n):
            if adj_matrix[i,j] == -1 and adj_matrix[j,i] == -1:
                problematic_conceptPairs.append([node_labels[i], node_labels[j]])
            elif adj_matrix[i,j] == 1 and adj_matrix[j,i] == 1:
                raise ValueError(f"Invalid adjacency matrix: bidirected edge in graph.")
    print('problematic_concepts:', problematic_conceptPairs)
    ### TODO: To get more reliable results, we could also repeat the analysis with
    ### the concepts reversed and only accept the results if they match the first set of findings.
    ### If they don’t match, we can ask the LLM to reconsider and choose the best option.

    if problematic_conceptPairs:
        for [c1,c2] in problematic_conceptPairs: # change to some clever procedure

            # Get descriptions for the concepts
            if label_descriptions is not None:
                c1_description = label_descriptions.get(c1, "No description available.")
                c2_description = label_descriptions.get(c2, "No description available.")
            else:
                c1_description, c2_description = "", ""
            #prompt = prompt_template2(input_variables=["context", "concept_1", "concept_2", "concept_1_description", "concept_2_description"],
            #                          template=prompt_template2)
            if rag is not None:
                #context_query = f"changing {c1} to certain values can result in a change in {c2}?"
                context_query = f"""Is there a causal relationship between {c1} and {c2}?
                1. {c1} - {c1_description}
                2. {c2} - {c2_description}
                """
                retrieved_context = str(rag.invoke(context_query))
                
            else:
                retrieved_context = "No context available."

            prompt = prompt_template_cot.format(context=retrieved_context, 
                                                concept_1=c1, 
                                                concept_2=c2, 
                                                concept_1_description=c1_description, 
                                                concept_2_description=c2_description)

            response = llm_model.invoke_with_retry(prompt)
            
            if rag is not None:
                if rag.verbose:
                    print("PROMPT:", prompt, '\n')
                    print("Response:", response)

            # retrieve the final answer (NC,IC,C) from the whole text answer of the large language model (LLM)
            # causal_edge = parse_answer(response)

            '''
            if causal_edge!=None:
                singlePair_responses.append((causal_edge, str(response)))
            else:
                print(f"No valid answer could be identified.")
                continue

            pattern = re.compile(r'"(A|B|C|D)"')
            match = pattern.search(response)
            if match:
                option = match.group(1).upper()
                singlePair_responses.append((option, str(response)))
            else:
                print(f"No valid answer could be identified.")
                continue
    
            # Extract options from singlePair_responses
            options = [option for option, _ in singlePair_responses]
            print(f"c1:{c1}, c2:{c2}, options: {options}")

            # Find the most common option from singlePair_responses
            most_common_option = Counter(options).most_common(1)[0][0]

            print("most common option", most_common_option)

            # Append to total_responses using the most common option
            total_responses.append(c1 + ", " + c2 + ", " + next((text for option, text in singlePair_responses
                                                if option == most_common_option), None))

            '''

            # use the answer to direct the edges in the graph
            c1_index = node_labels.index(c1)
            c2_index = node_labels.index(c2)

            if response == "A":
                adj_matrix[c1_index,c2_index]=1
                adj_matrix[c2_index,c1_index]=0
            elif response == "B":
                adj_matrix[c1_index,c2_index]=0
                adj_matrix[c2_index,c1_index]=1
            elif response == "C":
                adj_matrix[c1_index,c2_index]=0
                adj_matrix[c2_index,c1_index]=0
            else:
                raise ValueError(f"Invalid option: {response}")
            '''
            else:
                if cfg.selection_bias:
                    adj_matrix[c1_index,c2_index]=0
                    adj_matrix[c2_index,c1_index]=0
                else:
                    print(f"There is a mismatch between the skeleton found from the structural learning algorithm and the LLM response for the pair of concepts {c1} - {c2}. The skeleton will be retained.")

            progress_bar.update(1)
            '''
        
    print(f"new graph \n {adj_matrix}")
    return pd.DataFrame(adj_matrix, index=node_labels, columns=node_labels)