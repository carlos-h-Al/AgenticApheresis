from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import chromadb

# Embedding function
class NomicEmbeddingFunction:
    def __init__(self):
        self.model = SentenceTransformer("nomic-ai/nomic-embed-text-v2-moe", trust_remote_code=True)
    
    def __call__(self, input):
        # Make sure input is always a list, even for a single string
        if isinstance(input, str):
            input = [input]
        
        # Generate embeddings
        embeddings = self.model.encode(input, prompt_name="passage")
        
        # Make sure the output has the right shape
        # If embeddings is a single vector, ensure it's returned properly
        if len(input) == 1 and len(embeddings.shape) == 1:
            # Convert 1D array to 2D array with one row
            return [embeddings]
        
        return embeddings.tolist()


def text_formatter(text):
    filter_one = text.replace('\n', '')
    filter_two = [word for word in filter_one.split() if word != '']
    filter_three = ' '.join(filter_two)
    return filter_three


def document_splitter(reader):
    pages = []
    metadata = []
    print('\nSplitting document into chunks...\n')
    for idx, page in enumerate(reader.pages):
        # Skip last page
        if idx == (len(reader.pages) - 1):
            pass
        else:
            metadata.append({
                'document': 'consent_leaflet',
                'page': idx + 1
            })
            pages.append(text_formatter(page.extract_text()))
    print('Document split completed ✅\n')
    return pages, metadata


def main():
    reader = PdfReader('plt_leaflet.pdf')

    pages, metadata = document_splitter(reader)

    chroma_database = 'embeddings/platelets_general'
    nomic = NomicEmbeddingFunction()
    client = chromadb.PersistentClient(path=chroma_database)
    platelets_general_knowledge = client.get_or_create_collection(
        name='general', 
        embedding_function=nomic
        )

    title_ids = [str(i) for i in range(len(pages))]
    print('Adding pages and metadata to vector database...\n')
    platelets_general_knowledge.add(
        documents=pages,
        metadatas=metadata,
        ids=title_ids,
    )
    print('Database setup complete ✅')


if __name__ == '__main__':
    main()
