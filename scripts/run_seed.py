# scripts/run_seed.py

import os
import sys
from supabase import create_client, Client
from datetime import datetime
import hashlib

# Add project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from seed_headspace import SEED_DOCUMENTS

# --- Supabase Config ---
SUPABASE_URL = "https://pwxleudvclhcbksvjcho.supabase.co"
# It's better to use an environment variable for the service key
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")

if not SUPABASE_SERVICE_KEY:
    print("🔥 Error: SUPABASE_SERVICE_KEY environment variable not set.")
    print("Please set it to your Supabase project's service_role key.")
    sys.exit(1)

def seed_database():
    """Connects to Supabase and seeds the documents table."""
    print("🌱 Starting to seed the database...")
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
        print("✅ Connected to Supabase.")
    except Exception as e:
        print(f"🔥 Failed to connect to Supabase: {e}")
        return

    documents_to_upsert = []
    for seed_doc in SEED_DOCUMENTS:
        # Generate a consistent ID for each seed document
        doc_id = hashlib.md5(
            f"{seed_doc['title']}".encode()
        ).hexdigest()[:12]

        # The 'documents' table schema seems to be a flat structure based on index-debug.html
        # and data_models.py. Let's construct the object to insert.
        document_data = {
            "id": doc_id,
            "title": seed_doc["title"],
            "content": seed_doc["content"],
            "doc_type": seed_doc["doc_type"],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "metadata": seed_doc.get("metadata", {}),
            # UMAP coordinates will be null until calculated
            "umap_coordinates": None 
        }
        documents_to_upsert.append(document_data)
        print(f"  - Preparing document: {seed_doc['title']}")

    try:
        print("\n🔄 Upserting documents into Supabase...")
        # Using upsert to prevent duplicates if script is run again
        response = supabase.table("documents").upsert(documents_to_upsert).execute()
        
        if response.data:
            print(f"✅ Successfully upserted {len(response.data)} documents.")
        else:
            # The V1 library might not have a helpful response, check for error
            if hasattr(response, 'error') and response.error:
                 raise response.error
            print("✅ Upsert operation completed (check Supabase for details).")


    except Exception as e:
        print(f"🔥 An error occurred during upsert: {e}")

if __name__ == "__main__":
    seed_database()
