#from metapub import PubMedFetcher
import json
import time
import os
import re
import torch
from sklearn import metrics
from tqdm import tqdm
import numpy as np
from random import sample
from torch.utils.data import DataLoader

from sentence_transformers import SentenceTransformer
from src.completion.llm.llm_client import llm_client
from src.completion.context_generator import Context_generator
from src.data.labelfree_preprocessing import encode_text, encode_image



#nltk.download('punkt_tab')

CONCEPTS_GENERATION_PROMPT = """You are an expert of {context}. \
You need to list the most important features to recognize {class_label} from {input}. \
List also the variables that are most likely to be associated with {class_label} as well as
the variables that are most likely to be associated with the absence of {class_label}. \
You need also to give a list of superclasses for the word {class_label}.
Combine all the lists in a single one and separate the single terms with a comma. \
If a term is composed by more than one word, use an underscore to separate the words.\
"""

def parse_string_to_list(input_string):
    # Split the input string by commas and strip any extra whitespace
    output_string = [word.strip() for word in input_string.split(',')]
    output_string = [word.replace('_',' ') for word in output_string]
    output_string = ["there is "+word for word in output_string]
    return output_string

def concepts_generation(llm_model: str = "gpt-4o",
                        rag_model: str = "standard",
                        class_labels: list = ["Pneumothorax"],
                        input: str = "chest x-ray images", 
                        context: str = "medical imaging",
                        temperature = 0.):
    '''
    Generate concepts for recognizing a class label from an input in a given context.
    Args:
        llm_model: the LLM model to be used for generating the concepts
        rag_model: the RAG model to be used for generating the concepts
        class_label: the class label
        input: the input
        context: the context
        '''
    concepts = []
    llm_model = llm_client(LLM = llm_model, 
                           temperature = temperature)

    answer = llm_model.invoke(CONCEPTS_GENERATION_PROMPT.format(context = context,
                                                                class_label = class_labels[0],
                                                                input = input))
    concepts = parse_string_to_list(answer)
    return concepts

def filtering_concepts_from_llm(concepts, 
                                class_labels,
                                training_data,
                                clip_model,
                                clip_tokenizer,
                                ckpt_config,
                                device,
                                n_char_threshold = 50,
                                class_similarity_threshold = 0.9,
                                concept_similarity_threshold = 0.9,
                                presence_training_data_threshold = 0.2):

    # modify class labels
    class_label = class_labels[0]
    class_labels = [class_label, "No "+class_label]

    # filter out concepts that are too long
    filtered_concepts = [c for c in concepts if len(c) < n_char_threshold]
    print("Concepts too long:" ,[c for c in concepts if c  not in filtered_concepts])
    concepts = filtered_concepts

    # filter out concepts that are too similar to the class labels
    concepts_embeddings = encode_text(clip_model, clip_tokenizer, ckpt_config, concepts, device)
    class_labels_embeddings = encode_text(clip_model, clip_tokenizer, ckpt_config, class_labels, device)
    class_similarities = metrics.pairwise.cosine_similarity(concepts_embeddings, class_labels_embeddings)
    filtered_concepts = [c for i, c in enumerate(concepts) if (class_similarities[i,0] < class_similarity_threshold) & (class_similarities[i,1] < class_similarity_threshold)]
    print("Concepts too similar to ", class_labels[0], ":",[c for c in concepts if c  not in filtered_concepts])
    concepts_embeddings = concepts_embeddings[[i for i, c in enumerate(concepts) if c in filtered_concepts]]
    concepts = filtered_concepts
    
    # filter out concepts that are too similar to each other
    concepts_similarities = metrics.pairwise.cosine_similarity(concepts_embeddings)
    concepts_index_to_remove = []
    concepts_too_similar = []
    for i, c in enumerate(concepts[:-1]):
        if max(concepts_similarities[i, i+1:]) > concept_similarity_threshold:
            concepts_index_to_remove.append(i)
            index_argmax = np.argmax(concepts_similarities[i, i+1:])+ i+1
            concepts_too_similar.append(concepts[index_argmax])
            print("Concept ", concepts[i], " is too similar to ", concepts_too_similar[-1])
    
    filtered_concepts = [c for i, c in enumerate(concepts) if i not in concepts_index_to_remove]
    concepts_embeddings = concepts_embeddings[[i for i, c in enumerate(concepts) if i not in concepts_index_to_remove]]
    concepts = filtered_concepts

    # filter out concepts that are not present in training data
    # sample training data
    #training_data.df = training_data.df.iloc[sample(range(len(training_data.df)), 1000), :].reset_index(drop=True)

    # create dataloader
    dataloader = DataLoader(training_data, 
                            #batch_size=cfg.dataset.batch_size, 
                            batch_size = 64,
                            collate_fn = getattr(training_data, "collate_fn", None), 
                            num_workers = 1,
                            pin_memory = True,
                            shuffle = False,
                            drop_last = False)
    
    images_c_similarities = []
    for batch in tqdm(dataloader):
        # encode images to calculate similarity with concepts
        img_emb = encode_image(clip_model, batch["x"], device)
        img_c_similarities = metrics.pairwise.cosine_similarity(img_emb, concepts_embeddings)
        images_c_similarities.extend(img_c_similarities)

    images_c_similarities = torch.tensor(np.array(images_c_similarities))

    index_concepts_to_remove = []
    for i, c in enumerate(concepts):
        sorted_similarities = torch.sort(images_c_similarities[:,i], descending = True).values
        print("sorted similarities: ", sorted_similarities[:5])
        if all(sorted_similarities[:5] < presence_training_data_threshold): 
            print("Concept ", c, " is not higly represented in training.")
            index_concepts_to_remove.append(i)
    
    concepts = [c for i, c in enumerate(concepts) if i not in index_concepts_to_remove]

    return concepts