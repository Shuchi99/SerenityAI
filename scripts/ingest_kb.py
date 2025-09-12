import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from app.rag.ingest import main

if __name__ == "__main__":
    main()
