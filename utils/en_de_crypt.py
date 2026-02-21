"""
Encryption/Decryption Utility for LiveKit Secrets

This module provides symmetric encryption/decryption using Fernet (cryptography library).
It auto-detects whether input is encrypted or plain text based on the "dev-s-t-" prefix.

Usage as module:
    from utils.en_de_crypt import encrypt_decrypt
    
    # Encrypt
    encrypted = encrypt_decrypt("my_secret_key")
    # Returns: "dev-s-t-gAAAAABh1234..."
    
    # Decrypt
    decrypted = encrypt_decrypt("dev-s-t-gAAAAABh1234...")
    # Returns: "my_secret_key"

Usage as CLI:
    # Encrypt
    uv run python utils/en_de_crypt.py "my_secret_key"
    
    # Decrypt
    uv run python utils/en_de_crypt.py "dev-s-t-gAAAAABh1234..."

Environment Variables:
    LIVEKIT_SECRET_ENCRYPTION_KEY: 32-byte base64 Fernet key (required)
    
Generate key:
    python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
"""

import os
import sys
from cryptography.fernet import Fernet, InvalidToken


# Prefix to identify encrypted strings
ENCRYPTED_PREFIX = "dev-s-t-"


def encrypt_decrypt(input_string: str) -> str:
    """
    Auto-detect and encrypt/decrypt a string.
    
    This function automatically determines whether to encrypt or decrypt based on
    the presence of the "dev-s-t-" prefix.
    
    Args:
        input_string: Plain text string or encrypted string with "dev-s-t-" prefix
        
    Returns:
        - If input starts with "dev-s-t-": Returns decrypted plain text
        - Otherwise: Returns encrypted string with "dev-s-t-" prefix
        
    Raises:
        ValueError: If LIVEKIT_SECRET_ENCRYPTION_KEY environment variable is not set
        cryptography.fernet.InvalidToken: If decryption fails (invalid encrypted string or wrong key)
        
    Examples:
        >>> # Encrypt
        >>> encrypted = encrypt_decrypt("my_secret")
        >>> encrypted.startswith("dev-s-t-")
        True
        
        >>> # Decrypt
        >>> decrypted = encrypt_decrypt(encrypted)
        >>> decrypted
        'my_secret'
    """
    if not input_string:
        raise ValueError("Input string cannot be empty")
    
    # Get encryption key from environment
    encryption_key = os.getenv("LIVEKIT_SECRET_ENCRYPTION_KEY")
    if not encryption_key:
        raise ValueError(
            "LIVEKIT_SECRET_ENCRYPTION_KEY environment variable not set. "
            "Generate with: python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\""
        )
    
    try:
        # Initialize Fernet cipher
        fernet = Fernet(encryption_key.encode())
    except Exception as e:
        raise ValueError(f"Invalid encryption key format: {e}. Key must be a 32-byte base64-encoded string.")
    
    # Check if input is encrypted (has prefix)
    if input_string.startswith(ENCRYPTED_PREFIX):
        # Decrypt
        encrypted_data = input_string[len(ENCRYPTED_PREFIX):]  # Remove prefix
        try:
            decrypted_bytes = fernet.decrypt(encrypted_data.encode())
            return decrypted_bytes.decode()
        except InvalidToken:
            raise InvalidToken(
                "Decryption failed. The encrypted string may be corrupted or encrypted with a different key."
            )
    else:
        # Encrypt
        encrypted_bytes = fernet.encrypt(input_string.encode())
        encrypted_string = encrypted_bytes.decode()
        return f"{ENCRYPTED_PREFIX}{encrypted_string}"


def main():
    """CLI interface for encryption/decryption."""
    if len(sys.argv) != 2:
        print("Usage: uv run python utils/en_de_crypt.py <string_to_encrypt_or_decrypt>")
        print()
        print("Examples:")
        print("  # Encrypt a secret")
        print('  uv run python utils/en_de_crypt.py "my_secret_key"')
        print()
        print("  # Decrypt an encrypted secret")
        print('  uv run python utils/en_de_crypt.py "dev-s-t-gAAAAABh1234..."')
        print()
        print("Environment Variables:")
        print("  LIVEKIT_SECRET_ENCRYPTION_KEY: Required. Generate with:")
        print('  python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"')
        sys.exit(1)
    
    input_string = sys.argv[1]
    
    try:
        result = encrypt_decrypt(input_string)
        
        # Determine operation
        if input_string.startswith(ENCRYPTED_PREFIX):
            print(f"Decrypted: {result}")
        else:
            print(f"Encrypted: {result}")
            print()
            print("Store this encrypted value in the database.")
            print("To decrypt later, run:")
            print(f'  uv run python utils/en_de_crypt.py "{result}"')
        
        sys.exit(0)
        
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except InvalidToken as e:
        print(f"Decryption Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
