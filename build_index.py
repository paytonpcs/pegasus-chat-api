import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

WP_BASE_URL = "https://pegasuscs.com/wp-json/wp/v2/"   
INDEX_PATH = "./wp_semantic_index"
EMBEDDING_MODEL = "mxbai-embed-large"

# Headers to fix 406 error
HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0 (compatible; SemanticIndexer/1.0)"
}

def clean_content(html):
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form", "button", "svg"]):
        tag.decompose()
    text = soup.get_text(separator="\n", strip=True)
    lines = [line.strip() for line in text.splitlines() if line.strip() and len(line.strip()) > 3]
    return " ".join(lines)

def fetch_all(endpoint):
    items = []
    page = 1
    while True:
        url = f"{WP_BASE_URL}{endpoint}?per_page=100&page={page}"
        print(f"Fetching {endpoint} page {page}...")
        resp = requests.get(url, headers=HEADERS)
        print(f"  Status: {resp.status_code}")
        if resp.status_code != 200:
            break
        try:
            data = resp.json()
        except:
            print("  Failed to parse JSON")
            break
        if not isinstance(data, list) or not data:
            break
        items.extend(data)
        page += 1
    return items

print("Starting fetch from https://pegasuscs.com ...")
all_items = fetch_all("pages") + fetch_all("posts")
print(f"\nTotal items fetched: {len(all_items)}")

documents = []
for item in all_items:
    title = item.get("title", {}).get("rendered", "Untitled")
    content_html = item.get("content", {}).get("rendered", "")
    url = item.get("link", "")
    
    clean_text = clean_content(content_html)
    if len(clean_text) < 40:
        continue
        
    documents.append(Document(
        page_content=clean_text,
        metadata={"title": title, "source_url": url}
    ))

print(f"Valid documents after cleaning: {len(documents)}")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
chunks = text_splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks.")

if len(chunks) == 0:
    print("❌ No usable content found.")
    exit()

print("Building semantic index... (this may take a few minutes)")
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=INDEX_PATH
)

print("✅ Semantic index created successfully!")
print(f"Index saved in: wp_semantic_index")
print(f"Total chunks indexed: {len(chunks)}")