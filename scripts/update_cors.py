"""
Script to update CORS configuration on GCS bucket additively.
Adds new origins without overwriting existing ones.
"""

import json
from google.cloud import storage

# Configuration
BUCKET_NAME = "voiceagent-recording"
SERVICE_ACCOUNT_KEY = "secrets/salesmaya-yts-6b49f7694826.json"
NEW_ORIGIN = "https://app.mrlads.com"


def get_current_cors(bucket):
    """Get current CORS configuration from bucket."""
    cors = bucket.cors
    print(f"Current CORS configuration: {json.dumps(cors, indent=2)}")
    return cors


def update_cors_additive(bucket, new_origin):
    """Add new origin to CORS configuration without overwriting existing ones."""
    current_cors = bucket.cors or []
    
    # Check if we already have a CORS rule for GET/HEAD methods
    existing_rule = None
    for rule in current_cors:
        methods = rule.get("method", [])
        if "GET" in methods or "*" in methods:
            existing_rule = rule
            break
    
    if existing_rule:
        # Check if origin already exists
        origins = existing_rule.get("origin", [])
        if new_origin in origins:
            print(f"Origin '{new_origin}' already exists in CORS configuration.")
            return False
        
        # Add new origin to existing rule
        origins.append(new_origin)
        existing_rule["origin"] = origins
        print(f"Added '{new_origin}' to existing CORS rule.")
    else:
        # Create new CORS rule
        new_rule = {
            "origin": [new_origin],
            "method": ["GET", "HEAD", "OPTIONS"],
            "responseHeader": [
                "Content-Type",
                "Content-Length", 
                "Content-Range",
                "Accept-Ranges",
                "Content-Disposition"
            ],
            "maxAgeSeconds": 3600
        }
        current_cors.append(new_rule)
        print(f"Created new CORS rule for '{new_origin}'.")
    
    # Update bucket CORS
    bucket.cors = current_cors
    bucket.patch()
    
    print(f"\nUpdated CORS configuration: {json.dumps(bucket.cors, indent=2)}")
    return True


def main():
    # Initialize client with service account
    client = storage.Client.from_service_account_json(SERVICE_ACCOUNT_KEY)
    
    # Get bucket
    bucket = client.bucket(BUCKET_NAME)
    bucket.reload()  # Load current configuration
    
    print(f"Bucket: {BUCKET_NAME}")
    print("=" * 50)
    
    # Show current CORS
    get_current_cors(bucket)
    
    print("\n" + "=" * 50)
    print("Updating CORS configuration...")
    print("=" * 50 + "\n")
    
    # Update CORS additively
    if update_cors_additive(bucket, NEW_ORIGIN):
        print("\n✅ CORS configuration updated successfully!")
    else:
        print("\n⚠️ No changes needed - origin already configured.")


if __name__ == "__main__":
    main()
