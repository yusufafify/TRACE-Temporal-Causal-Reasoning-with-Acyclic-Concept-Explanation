from src.completion.llm.llama_v3_2 import get_llm_access, get_pretrained_llm
from src.completion.graph_completion import complete_graph
from src.completion.context_generator import Context_generator
from src.completion.llm.llm_client import llm_client
from src.plots import maybe_plot_graph

def complete_graph_with_llm(cfg, predicted_graph, dataset_name):
    # Complete the causal graph with an LLM and a RAG
    if cfg.llm is not None:
        # get_llm_access()
        # llm_model, llm_tokenizer = get_pretrained_llm(cfg.llm)
        llm_model = llm_client(LLM=cfg.llm.get('name'), temperature=cfg.llm.get('temperature'), max_tries=cfg.llm.get('max_tries'))
        if cfg.rag is not None:
            rag_model = Context_generator(llm=llm_model, 
                                          embedder=cfg.rag.get('embedder'),
                                          query_strategy=cfg.rag.get('query_strategy'), 
                                          chunking_strategy=cfg.rag.get('chunking_strategy'),
                                          source=cfg.rag.get('source'), 
                                          n_documents_per_source=cfg.rag.get('n_documents_per_source'),
                                          context_length=cfg.rag.get('context_length'),
                                          dataset=dataset_name,
                                          verbose = cfg.rag.get('verbose'))
        else:
            rag_model = None
        node_label_description = cfg.dataset.get('label_descriptions', None)
        completed_graph = complete_graph(predicted_graph,
                                         node_label_description,
                                         llm_model, 
                                         rag_model, 
                                         cfg.completion)
        maybe_plot_graph(completed_graph, 'completed_graph')
        return completed_graph
    else:
        return predicted_graph