"""Generate new encryption key and re-encrypt secrets"""
from cryptography.fernet import Fernet

# The plain text secrets
plain_secrets = {
    "uae": "SEC43172b2431a470ae02f0b11151f43866023f60c2f872f91e",
    "india": "fdfbeSxBNPgjHQyTr2bEeou4AxWIZqfCQHne9gyjFPxN"
}

print("="*80)
print("Generating New Encryption Key")
print("="*80)

# Generate a new key
new_key = Fernet.generate_key().decode()
print(f"\nNew Encryption Key:")
print(f"{new_key}")

print("\n" + "="*80)
print("Add this line to your .env file:")
print("="*80)
print(f'LIVEKIT_SECRET_ENCRYPTION_KEY={new_key}')
print("="*80)

# Encrypt the secrets with the new key
print("\nEncrypted Secrets (with dev-s-t- prefix):")
print("="*80)
cipher = Fernet(new_key.encode())

encrypted_values = {}
for name, plain in plain_secrets.items():
    encrypted = cipher.encrypt(plain.encode()).decode()
    encrypted_with_prefix = f"dev-s-t-{encrypted}"
    encrypted_values[name] = encrypted_with_prefix
    print(f"\n{name.upper()}:")
    print(f"  {encrypted_with_prefix}")

print("\n" + "="*80)
print("SQL to update database:")
print("="*80)
print(f"""
-- UAE VM
UPDATE lad_dev.voice_agent_livekit
SET livekit_api_secret = '{encrypted_values['uae']}'
WHERE name = 'uae-vm-selfhosted';

-- India Cloud
UPDATE lad_dev.voice_agent_livekit
SET livekit_api_secret = '{encrypted_values['india']}'
WHERE name = 'india-techiemaya-cloud';
""")
print("="*80)
