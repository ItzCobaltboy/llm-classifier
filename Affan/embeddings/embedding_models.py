from sentence_transformers import SentenceTransformer
from transformers import pipeline


##################### QWEN-3-Embedding-0.6B #####################
# Load the model
Qwen3_Embedding = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

# The queries and documents to embed
queries = [
    "What is the capital of China?",
    "Explain gravity",
]
documents = [
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
]

# Encode the queries and documents. Note that queries benefit from using a prompt
# Here we use the prompt called "query" stored under `model.prompts`, but you can
# also pass your own prompt via the `prompt` argument
query_embeddings = Qwen3_Embedding.encode(queries, prompt_name="query")
document_embeddings = Qwen3_Embedding.encode(documents)

# Compute the (cosine) similarity between the query and document embeddings
similarity = Qwen3_Embedding.similarity(query_embeddings, document_embeddings)
print(similarity)


###################### sentence-transformers/all-MiniLM-L6-v2 #####################
sentences = ["This is an example sentence", "Each sentence is converted"]

MiniLM = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = MiniLM.encode(sentences)
print(embeddings)

########################### google-bert/bert-base-uncased #####################

Google_bert = pipeline("fill-mask", model="google-bert/bert-base-uncased")
result = Google_bert("The capital of France is [MASK].")
print(result)

###################### sentence-transformers/all-mpnet-base-v2 #####################
sentences = ["This is an example sentence", "Each sentence is converted"]

MPNet = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
mpnet_embeddings = MPNet.encode(sentences)
print(mpnet_embeddings)


###################### intfloat/e5-base #####################
# E5 expects prefixes like "query:" or "passage:"
queries = [
    "query: What is the capital of China?",
    "query: Explain gravity",
]

documents = [
    "passage: The capital of China is Beijing.",
    "passage: Gravity is a force that attracts two bodies towards each other.",
]

E5 = SentenceTransformer("intfloat/e5-base")

query_embeddings = E5.encode(queries)
doc_embeddings = E5.encode(documents)

similarity = E5.similarity(query_embeddings, doc_embeddings)
print(similarity)   
