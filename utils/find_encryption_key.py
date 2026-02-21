     encrypted = cipher.encrypt(plain.encode()).decode()
        encrypted_with_prefix = f"dev-s-t-{encrypted}"
        print(f"\n{name.upper()}:")
        print(f"  Plain: {plain}")
        print(f"  Encrypted: {encrypted_with_prefix}")
    
    print("\n" + "="*80)
    print("Update database with these new encrypted values")
    print("="*80)
")
    print("\nGenerating new key...")
    
    # Generate a new key
    new_key = Fernet.generate_key().decode()
    print(f"\n✓ New key generated: {new_key}")
    
    print("\n" + "="*80)
    print("Add this to your .env file:")
    print("="*80)
    print(f'LIVEKIT_SECRET_ENCRYPTION_KEY={new_key}')
    print("="*80)
    
    # Encrypt the secrets with the new key
    print("\nRe-encrypting secrets with new key...")
    cipher = Fernet(new_key.encode())
    
    for name, plain in plain_secrets.items():
           else:
            print(f"✗ Key doesn't match. Decrypted: {decrypted[:20]}...")
    except Exception as e:
        print(f"✗ Decryption failed: {e}")
else:
    print("\n✗ No LIVEKIT_SECRET_ENCRYPTION_KEY in environmenty in environment: {env_key[:20]}...")
    
    # Try to decrypt with this key
    try:
        cipher = Fernet(env_key.encode())
        encrypted_data = encrypted_secrets["uae"].replace("dev-s-t-", "")
        decrypted = cipher.decrypt(encrypted_data.encode()).decode()
        
        if decrypted == plain_secrets["uae"]:
            print("✓ Key works! Secrets can be decrypted.")
gAAAAABpkwqJYnGoRzbTZhTKJDPlipA6ktgxEykb81Rys7gdajnWc6hSXX3h0PiV44HXeYvZWUjkjRAGjPe3T9j0c9rTP4UzFMrcuzJy6BrOgfWJGzjKfaGBnmBpCrYtquHin9IBAV5P"
}

# The plain text secrets we know
plain_secrets = {
    "uae": "SEC43172b2431a470ae02f0b11151f43866023f60c2f872f91e",
    "india": "fdfbeSxBNPgjHQyTr2bEeou4AxWIZqfCQHne9gyjFPxN"
}

print("="*80)
print("Encryption Key Recovery")
print("="*80)

# Check if key exists in environment
env_key = os.getenv("LIVEKIT_SECRET_ENCRYPTION_KEY")
if env_key:
    print(f"\n✓ Found ke original key or generate a new one and re-encrypt.
"""

from cryptography.fernet import Fernet
import os

# The encrypted secrets from the database
encrypted_secrets = {
    "uae": "dev-s-t-gAAAAABpkvXrdJaiHZ6NQt0QT0fh4UbptQUtrXp4ke3VtuQb0R8YTFIfPow2lCSetE19xuIaEnCrUwySow1EaACwr1gvGC98b4WHfSFwsODrtYKn9rbE23e6X6t2k68whL3pfgyqqvR5d9_4eT8_Qs4aqVNZVLc0XA==",
    "india": "dev-s-t-ry to find or generate the encryption key.

The encrypted secrets start with "dev-s-t-" which means they were encrypted with a Fernet key.
We need to either find the"""
T