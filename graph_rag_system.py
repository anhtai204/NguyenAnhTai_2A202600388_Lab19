import os
import networkx as nx
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import JsonOutputParser
try:
    from langchain_core.pydantic_v1 import BaseModel, Field
except ImportError:
    from pydantic import BaseModel, Field
from typing import List, Tuple

# Load environment variables (API Key)
load_dotenv()

class Triple(BaseModel):
    subject: str = Field(description="The subject of the relationship")
    predicate: str = Field(description="The relationship between subject and object")
    obj: str = Field(description="The object of the relationship")

class Triples(BaseModel):
    triples: List[Triple]

class GraphRAGSystem:
    def __init__(self, corpus_path: str, provider: str = "openai"):
        self.corpus_path = corpus_path
        if provider == "openai":
            self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, timeout=60, max_retries=3)
            self.embeddings = OpenAIEmbeddings()
        elif provider == "google":
            self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, timeout=60)
            self.embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
        elif provider == "ollama":
            print("Using local Ollama (qwen2.5:7b)...")
            self.llm = ChatOllama(model="qwen2.5:7b", temperature=0)
            self.embeddings = OllamaEmbeddings(model="qwen2.5:7b")
        else:
            raise ValueError("Provider must be 'openai', 'google', or 'ollama'")
        
        self.graph = nx.MultiDiGraph()
        self.vectorstore = None
        
    def load_corpus(self):
        with open(self.corpus_path, "r", encoding="utf-8") as f:
            return f.read()

    def extract_triples(self, text: str) -> List[Tuple[str, str, str]]:
        """Step 1: Indexing - Extract entities and relations with strict atomic rules."""
        print(f"Extracting triples from section (length: {len(text)})...")
        
        SCHEMA = """
        Allowed Entity Types: ORGANIZATION, PERSON, PRODUCT, LOCATION, TECHNOLOGY, DATE
        Allowed Relation Types: FOUNDED_BY, DEVELOPED, ACQUIRED, PARTNERED_WITH, HEADQUARTERED_IN, COMPETES_WITH, INVESTED_IN, SUCCEEDED_BY, MEMBER_OF, USES, PROVIDES, RUNS_ON, CREATED
        """
        
        parser = JsonOutputParser(pydantic_object=Triples)
        prompt = ChatPromptTemplate.from_template(
            "You are a Knowledge Graph Engineer. Your task is to extract ATOMIC triples from the text.\n"
            "SCHEMA:\n{schema}\n\n"
            "CRITICAL RULES:\n"
            "1. NO ENTITY MERGING: If multiple entities are involved, create SEPARATE triples for each. \n"
            "   BAD: (OpenAI, FOUNDED_BY, 'Sam Altman, Elon Musk')\n"
            "   GOOD: (OpenAI, FOUNDED_BY, 'Sam Altman'), (OpenAI, FOUNDED_BY, 'Elon Musk')\n"
            "2. ATOMIC OBJECTS: Each triple must have exactly ONE subject and ONE object.\n"
            "3. ENTITY RESOLUTION: Use official names (e.g., 'Microsoft Azure' instead of 'Azure', 'Apple' instead of 'the tech giant').\n"
            "4. STRICT SCHEMA: Use only the Allowed Relation Types provided.\n"
            "5. NO NOISE: Do not extract 'unknown', 'various', or vague entities.\n\n"
            "Text: {text}\n"
            "{format_instructions}"
        )
        
        chain = prompt | self.llm | parser
        try:
            result = chain.invoke({
                "text": text, 
                "schema": SCHEMA,
                "format_instructions": parser.get_format_instructions()
            })
            
            raw_data = []
            if isinstance(result, dict) and 'triples' in result:
                raw_data = result['triples']
            elif isinstance(result, list):
                raw_data = result
            
            # Post-processing: Force split any remaining merged entities and filter noise
            processed_triples = []
            for t in raw_data:
                s_raw = str(t.get('subject', ''))
                p = str(t.get('predicate', '')).upper().replace(' ', '_')
                o_raw = str(t.get('obj', ''))
                
                # Split by comma or 'and' to catch LLM failures in following Rule #1
                subjects = [x.strip() for x in s_raw.replace(' and ', ',').split(',')]
                objects = [x.strip() for x in o_raw.replace(' and ', ',').split(',')]
                
                for s in subjects:
                    for o in objects:
                        if s and o and s.lower() not in ['unknown', 'none', 'various', 'null'] and o.lower() not in ['unknown', 'none', 'various', 'null']:
                            processed_triples.append((s, p, o))
            return processed_triples
        except Exception as e:
            print(f"Error extracting from chunk: {e}")
            return []

    def build_graph(self, triples: List[Tuple[str, str, str]]):
        """Step 2: Construction - Build the knowledge graph."""
        print("Building graph...")
        for s, p, o in triples:
            self.graph.add_edge(s, o, relation=p)
        
    def visualize_graph(self, output_path: str = "graph_viz.png"):
        """Visualize the Knowledge Graph with improved spacing and readability."""
        plt.figure(figsize=(20, 15))
        # Increase k to push nodes further apart
        pos = nx.spring_layout(self.graph, k=2.0, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(self.graph, pos, node_color='skyblue', node_size=3500, alpha=0.9)
        
        # Draw labels with balanced font size
        nx.draw_networkx_labels(self.graph, pos, font_size=9, font_weight='bold')
        
        # Draw curved edges to distinguish multiple relations between same nodes
        nx.draw_networkx_edges(
            self.graph, pos, 
            edgelist=self.graph.edges(), 
            edge_color='gray', 
            arrows=True, 
            arrowsize=15, 
            connectionstyle='arc3, rad=0.1'
        )
        
        # Draw edge labels
        edge_labels = nx.get_edge_attributes(self.graph, 'relation')
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_size=7)
        
        plt.title("Tech Company Knowledge Graph", fontsize=15)
        plt.axis('off')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Graph visualization saved to {output_path}")

    def build_flat_rag(self, text: str):
        """Build a baseline Flat RAG using ChromaDB."""
        print("Building Flat RAG baseline...")
        texts = [p for p in text.split("\n\n") if p.strip()]
        
        # Ensure texts is not empty
        if not texts:
            print("Warning: Corpus is empty. Flat RAG not built.")
            return

        import uuid
        print(f"Creating vector store with {len(texts)} chunks...")
        
        # Manual embedding loop to ensure length consistency
        embeddings_list = []
        for i, t in enumerate(texts):
            print(f"  Embedding chunk {i+1}/{len(texts)}...")
            emb = self.embeddings.embed_query(t)
            embeddings_list.append(emb)
            
        print(f"DEBUG: Number of texts: {len(texts)}, Number of embeddings: {len(embeddings_list)}")
            
        # Bypass LangChain's internal embedding call by using the collection directly
        self.vectorstore = Chroma(
            collection_name="tech_corpus",
            embedding_function=self.embeddings
        )
        self.vectorstore._collection.add(
            ids=[str(uuid.uuid4()) for _ in texts],
            embeddings=embeddings_list,
            documents=texts,
            metadatas=[{"source": "corpus"}] * len(texts)
        )
        print("Flat RAG baseline built successfully.")

    def query_processing(self, query: str) -> List[str]:
        """Pipeline Step 1: Extract entities from query."""
        prompt = ChatPromptTemplate.from_template(
            "Extract ONLY the core names of companies, products, or people mentioned in this query.\n"
            "Avoid extracting long phrases or descriptions. Output ONLY the names separated by commas.\n"
            "Example: 'Who founded the company that created ChatGPT?' -> 'ChatGPT'\n"
            "Query: {query}\n"
            "Entities:"
        )
        response = self.llm.invoke(prompt.format(query=query))
        entities = [e.strip() for e in response.content.split(",")]
        return entities

    def graph_traversal(self, entities: List[str], hops: int = 2) -> List[str]:
        """Pipeline Step 2 & 3: Seed node matching & Traversal."""
        context_triples = []
        visited_nodes = set()
        
        for entity in entities:
            # Fuzzy matching: find nodes that contain the entity string
            seeds = [node for node in self.graph.nodes if entity.lower() in node.lower()]
            
            for seed in seeds:
                # N-hop traversal
                nodes_to_explore = {seed}
                for _ in range(hops):
                    new_nodes = set()
                    for node in nodes_to_explore:
                        if node in visited_nodes: continue
                        visited_nodes.add(node)
                        
                        # Get all outgoing edges
                        for neighbor in self.graph.successors(node):
                            edge_data = self.graph.get_edge_data(node, neighbor)
                            for key in edge_data:
                                rel = edge_data[key]['relation']
                                context_triples.append(f"{node} --[{rel}]--> {neighbor}")
                            new_nodes.add(neighbor)
                        
                        # Get all incoming edges (Bi-directional traversal)
                        for predecessor in self.graph.predecessors(node):
                            edge_data = self.graph.get_edge_data(predecessor, node)
                            for key in edge_data:
                                rel = edge_data[key]['relation']
                                context_triples.append(f"{predecessor} --[{rel}]--> {node}")
                            new_nodes.add(predecessor)
                    nodes_to_explore = new_nodes
        
        return list(set(context_triples))

    def generate_response(self, query: str, context: str, mode: str = "GraphRAG"):
        """Pipeline Step 4 & 5: Textualization & Generation."""
        prompt = ChatPromptTemplate.from_template(
            "You are a helpful assistant. Answer the question based ONLY on the provided context.\n"
            "Context Type: {mode}\n"
            "Context:\n{context}\n\n"
            "Question: {query}\n"
            "Answer:"
        )
        chain = prompt | self.llm
        return chain.invoke({"query": query, "context": context, "mode": mode}).content

    def run_benchmark(self, questions: List[str]):
        results = []
        for q in questions:
            print(f"\nQuestion: {q}")
            
            # Flat RAG
            docs = self.vectorstore.similarity_search(q, k=2)
            flat_context = "\n".join([d.page_content for d in docs])
            flat_ans = self.generate_response(q, flat_context, "FlatRAG")
            
            # GraphRAG
            entities = self.query_processing(q)
            triples = self.graph_traversal(entities)
            graph_context = "\n".join(triples)
            graph_ans = self.generate_response(q, graph_context, "GraphRAG")
            
            results.append({
                "question": q,
                "flat_rag": flat_ans,
                "graph_rag": graph_ans,
                "entities_found": entities
            })
        return results

if __name__ == "__main__":
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    print(f"Using provider: {provider}")
    system = GraphRAGSystem("tech_corpus.txt", provider=provider)
    
    # 1. Indexing - Process Doc-by-Doc for better resolution
    corpus_text = system.load_corpus()
    doc_chunks = [chunk.strip() for chunk in corpus_text.split("[DOC") if chunk.strip()]
    
    all_triples = []
    for i, chunk in enumerate(doc_chunks):
        # Restore the [DOC tag for context if needed, or just process the text
        full_chunk = f"[DOC {chunk}" if not chunk.startswith(" ") else f"[DOC{chunk}"
        print(f"\nProcessing Document {i+1}/{len(doc_chunks)}...")
        triples = system.extract_triples(full_chunk)
        all_triples.extend(triples)
    
    # 2. Construction
    system.build_graph(all_triples)
    system.visualize_graph()
    system.build_flat_rag(corpus_text)
    
    test_questions = [
        "Who founded the company that developed ChatGPT?",
        "Which company's cloud service is the exclusive provider for OpenAI?",
        "Who became the CEO of Google in 2015?",
        "Which company founded in 1976 created the iPhone?",
        "Which company developed CUDA to accelerate AI development?",
        "Who were the co-founders of Apple along with Steve Jobs?",
        "Which company acquired YouTube and Android?",
        "Which company launched AWS in 2006?",
        "Who succeeded Steve Jobs as CEO of Apple?",
        "Which companies rely on NVIDIA's GPUs for their AI infrastructure?",
        "Which university did the founders of Google attend?",
        "What was the original business of Amazon when it was founded in 1994?",
        "Which company transitioned to a capped-profit structure called OpenAI LP?",
        "Which company competes with Apple's iOS through the Android operating system?",
        "Which hardware company's processors compete with Apple's M-series chips?",
        "Which organization was initially established as a non-profit to benefit humanity?",
        "Which cloud provider is the largest globally according to the text?",
        "Who are the founders of NVIDIA?",
        "Which company partnered with Anthropic?",
        "Which company developed image generation system DALL-E?"
    ]
    
    benchmark_results = system.run_benchmark(test_questions)
    
    # Save Results
    import pandas as pd
    df = pd.DataFrame(benchmark_results)
    df.to_csv("benchmark_results.csv", index=False)
    print("\nBenchmark completed. Results saved to benchmark_results.csv")
