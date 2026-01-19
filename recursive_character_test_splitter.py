from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

tesla_text = """Tesla's Q3 Results

Tesla reported record revenue of $25.2B in Q3 2024.

Model Y Performance

The Model Y became the best-selling vehicle globally, with 350,000 units sold.

Production Challenges

Supply chain issues caused a 12% increase in production costs.

This is one very long paragraph that definitely exceeds our 100 character limit and has no double newlines inside it whatsoever making it impossible to split properly."""

# recursive_splitter= RecursiveCharacterTextSplitter(
#     separators=["\n\n","\n",". "," ", ""], # different separator options to select from one not work
#     chunk_size = 100,
#     chunk_overlap= 0
# )

# chunks = recursive_splitter.split_text(tesla_text)

# print(f" chunks with Recursice Character Text Splitter: ")
# for i, chunk in enumerate(chunks):
#     print(f"Chunk {i}: ({len(chunk)} chars)")
#     print(f" {chunk}")
#     print("\n")
    
    
#     chunks with Recursice Character Text Splitter: 
# Chunk 0: (92 chars)
#  Tesla's Q3 Results

# Tesla reported record revenue of $25.2B in Q3 2024.

# Model Y Performance

# Chunk 1: (78 chars)
#  The Model Y became the best-selling vehicle globally, with 350,000 units sold.

# Chunk 2: (85 chars)
#  Production Challenges

# Supply chain issues caused a 12% increase in production costs.

# Chunk 3: (97 chars)
#  This is one very long paragraph that definitely exceeds our 100 character limit and has no double

# Chunk 4: (69 chars)
#  newlines inside it whatsoever making it impossible to split properly. 


splitter1 = CharacterTextSplitter(
    separator=" ",  # Default separator. Other options include ["\n\n", "\n", ". ", " ", ""]
    chunk_size=100,
    chunk_overlap=0
)

chunks1 = splitter1.split_text(tesla_text)
for i, chunk in enumerate(chunks1, 1):
    print(f"Chunk {i}: ({len(chunk)} chars)")
    print(f'"{chunk}"')
    print()
