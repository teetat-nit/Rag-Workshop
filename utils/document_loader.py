from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    CSVLoader,
    UnstructuredMarkdownLoader,
)
import os

def load_single_document(file_path):
    """Load a single document based on its file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".csv":
        loader = CSVLoader(file_path)
    elif ext == ".md":
        loader = UnstructuredMarkdownLoader(file_path)
    else:
        # Default to text loader
        loader = TextLoader(file_path)
        
    return loader.load()

def load_documents_from_directory(directory_path, glob_pattern="**/*.*"):
    """Load all documents from a directory matching the glob pattern."""
    from langchain_community.document_loaders import DirectoryLoader
    
    # Create a mapping of file extensions to loaders
    loaders = {
        ".pdf": PyPDFLoader,
        ".csv": CSVLoader,
        ".md": UnstructuredMarkdownLoader,
        ".txt": TextLoader
    }
    
    # Load all documents
    all_documents = []
    for ext, loader_cls in loaders.items():
        try:
            loader = DirectoryLoader(
                directory_path, 
                glob=f"**/*{ext}",
                loader_cls=loader_cls
            )
            documents = loader.load()
            all_documents.extend(documents)
            print(f"Loaded {len(documents)} {ext} files")
        except Exception as e:
            print(f"Error loading {ext} files: {e}")
    
    return all_documents