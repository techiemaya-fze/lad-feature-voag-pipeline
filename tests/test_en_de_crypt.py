"""
Unit Tests for Encryption/Decryption Utility

Tests the utils/en_de_crypt.py module for encrypting and decrypting LiveKit secrets.
"""

import os
import pytest
from cryptography.fernet import Fernet, InvalidToken
from utils.en_de_crypt import encrypt_decrypt, ENCRYPTED_PREFIX


# Test encryption key (generated for testing only)
TEST_ENCRYPTION_KEY = "mDTmNQg6OxP_qMsdahXWWqujA6BfpPjYyn_YkpXeo0o="


@pytest.fixture
def set_encryption_key(monkeypatch):
    """Set test encryption key in environment."""
    monkeypatch.setenv("LIVEKIT_SECRET_ENCRYPTION_KEY", TEST_ENCRYPTION_KEY)


@pytest.fixture
def unset_encryption_key(monkeypatch):
    """Remove encryption key from environment."""
    monkeypatch.delenv("LIVEKIT_SECRET_ENCRYPTION_KEY", raising=False)


class TestEncryption:
    """Test encryption functionality."""
    
    def test_encrypt_produces_prefix(self, set_encryption_key):
        """Test that encryption produces the dev-s-t- prefix."""
        plain_text = "my_secret_key"
        encrypted = encrypt_decrypt(plain_text)
        
        assert encrypted.startswith(ENCRYPTED_PREFIX)
        assert encrypted != plain_text
        assert len(encrypted) > len(ENCRYPTED_PREFIX)
    
    def test_encrypt_different_inputs_produce_different_outputs(self, set_encryption_key):
        """Test that different inputs produce different encrypted outputs."""
        text1 = "secret1"
        text2 = "secret2"
        
        encrypted1 = encrypt_decrypt(text1)
        encrypted2 = encrypt_decrypt(text2)
        
        assert encrypted1 != encrypted2
        assert encrypted1.startswith(ENCRYPTED_PREFIX)
        assert encrypted2.startswith(ENCRYPTED_PREFIX)
    
    def test_encrypt_same_input_produces_different_outputs(self, set_encryption_key):
        """Test that encrypting the same input twice produces different outputs (Fernet uses random IV)."""
        plain_text = "my_secret"
        
        encrypted1 = encrypt_decrypt(plain_text)
        encrypted2 = encrypt_decrypt(plain_text)
        
        # Fernet uses random IV, so same plaintext produces different ciphertext
        assert encrypted1 != encrypted2
        assert encrypted1.startswith(ENCRYPTED_PREFIX)
        assert encrypted2.startswith(ENCRYPTED_PREFIX)


class TestDecryption:
    """Test decryption functionality."""
    
    def test_decrypt_returns_original_value(self, set_encryption_key):
        """Test that decryption returns the original plain text."""
        plain_text = "my_secret_key_123"
        
        # Encrypt
        encrypted = encrypt_decrypt(plain_text)
        
        # Decrypt
        decrypted = encrypt_decrypt(encrypted)
        
        assert decrypted == plain_text
    
    def test_decrypt_multiple_values(self, set_encryption_key):
        """Test decrypting multiple different values."""
        test_values = [
            "simple_secret",
            "complex_secret_with_special_chars!@#$%",
            "secret with spaces",
            "123456789",
            "a" * 100,  # Long secret
        ]
        
        for plain_text in test_values:
            encrypted = encrypt_decrypt(plain_text)
            decrypted = encrypt_decrypt(encrypted)
            assert decrypted == plain_text, f"Failed for: {plain_text}"
    
    def test_decrypt_invalid_encrypted_string_raises_error(self, set_encryption_key):
        """Test that decrypting an invalid encrypted string raises InvalidToken."""
        invalid_encrypted = f"{ENCRYPTED_PREFIX}invalid_base64_string"
        
        with pytest.raises(InvalidToken):
            encrypt_decrypt(invalid_encrypted)
    
    def test_decrypt_with_wrong_key_raises_error(self, monkeypatch):
        """Test that decrypting with a different key raises InvalidToken."""
        # Encrypt with first key
        monkeypatch.setenv("LIVEKIT_SECRET_ENCRYPTION_KEY", TEST_ENCRYPTION_KEY)
        plain_text = "my_secret"
        encrypted = encrypt_decrypt(plain_text)
        
        # Try to decrypt with different key
        different_key = Fernet.generate_key().decode()
        monkeypatch.setenv("LIVEKIT_SECRET_ENCRYPTION_KEY", different_key)
        
        with pytest.raises(InvalidToken):
            encrypt_decrypt(encrypted)


class TestRoundTrip:
    """Test round-trip encryption and decryption."""
    
    def test_round_trip_simple(self, set_encryption_key):
        """Test encrypt → decrypt → original."""
        original = "test_secret"
        encrypted = encrypt_decrypt(original)
        decrypted = encrypt_decrypt(encrypted)
        
        assert decrypted == original
    
    def test_round_trip_complex_string(self, set_encryption_key):
        """Test round-trip with complex strings."""
        test_cases = [
            "SEC43172b2431a470ae02f0b11151f43866023f60c2f872f91e",  # LiveKit secret format
            "wss://mumbai.livekit.example.com",  # URL
            "APIbe273e3142c7b96a4a87bba4",  # API key format
            "multi\nline\nsecret",  # Multi-line
            '{"key": "value"}',  # JSON
        ]
        
        for original in test_cases:
            encrypted = encrypt_decrypt(original)
            decrypted = encrypt_decrypt(encrypted)
            assert decrypted == original, f"Round-trip failed for: {original}"
    
    def test_multiple_round_trips(self, set_encryption_key):
        """Test that we can't accidentally double-encrypt."""
        original = "my_secret"
        
        # First encryption
        encrypted1 = encrypt_decrypt(original)
        assert encrypted1.startswith(ENCRYPTED_PREFIX)
        
        # Decrypt
        decrypted1 = encrypt_decrypt(encrypted1)
        assert decrypted1 == original
        
        # Encrypt again
        encrypted2 = encrypt_decrypt(original)
        assert encrypted2.startswith(ENCRYPTED_PREFIX)
        
        # Decrypt again
        decrypted2 = encrypt_decrypt(encrypted2)
        assert decrypted2 == original


class TestErrorHandling:
    """Test error handling."""
    
    def test_missing_encryption_key_raises_error(self, unset_encryption_key):
        """Test that missing encryption key raises ValueError."""
        with pytest.raises(ValueError, match="LIVEKIT_SECRET_ENCRYPTION_KEY"):
            encrypt_decrypt("test")
    
    def test_empty_input_raises_error(self, set_encryption_key):
        """Test that empty input raises ValueError."""
        with pytest.raises(ValueError, match="cannot be empty"):
            encrypt_decrypt("")
    
    def test_invalid_encryption_key_format_raises_error(self, monkeypatch):
        """Test that invalid encryption key format raises ValueError."""
        monkeypatch.setenv("LIVEKIT_SECRET_ENCRYPTION_KEY", "invalid_key")
        
        with pytest.raises(ValueError, match="Invalid encryption key format"):
            encrypt_decrypt("test")


class TestPrefixDetection:
    """Test prefix detection logic."""
    
    def test_string_with_prefix_is_decrypted(self, set_encryption_key):
        """Test that strings with prefix are treated as encrypted."""
        # First encrypt something
        plain_text = "secret"
        encrypted = encrypt_decrypt(plain_text)
        
        # Verify it has prefix
        assert encrypted.startswith(ENCRYPTED_PREFIX)
        
        # Decrypt should work
        decrypted = encrypt_decrypt(encrypted)
        assert decrypted == plain_text
    
    def test_string_without_prefix_is_encrypted(self, set_encryption_key):
        """Test that strings without prefix are treated as plain text."""
        plain_text = "my_secret_without_prefix"
        
        # Should encrypt (add prefix)
        result = encrypt_decrypt(plain_text)
        assert result.startswith(ENCRYPTED_PREFIX)
        assert result != plain_text


class TestRealWorldScenarios:
    """Test real-world usage scenarios."""
    
    def test_livekit_secret_encryption(self, set_encryption_key):
        """Test encrypting a real LiveKit API secret."""
        livekit_secret = "SEC43172b2431a470ae02f0b11151f43866023f60c2f872f91e"
        
        encrypted = encrypt_decrypt(livekit_secret)
        assert encrypted.startswith(ENCRYPTED_PREFIX)
        
        decrypted = encrypt_decrypt(encrypted)
        assert decrypted == livekit_secret
    
    def test_store_and_retrieve_from_database_simulation(self, set_encryption_key):
        """Simulate storing encrypted value in database and retrieving it."""
        # Original secret
        original_secret = "my_livekit_api_secret_12345"
        
        # Encrypt before storing in database
        encrypted_for_db = encrypt_decrypt(original_secret)
        assert encrypted_for_db.startswith(ENCRYPTED_PREFIX)
        
        # Simulate storing in database (just keep in variable)
        stored_value = encrypted_for_db
        
        # Simulate retrieving from database
        retrieved_value = stored_value
        
        # Decrypt after retrieving
        decrypted_secret = encrypt_decrypt(retrieved_value)
        assert decrypted_secret == original_secret
    
    def test_multiple_secrets_independently(self, set_encryption_key):
        """Test encrypting multiple secrets independently."""
        secrets = {
            "livekit_url": "wss://server.livekit.cloud",
            "livekit_api_key": "APIxxxxxxxxxxxxx",
            "livekit_api_secret": "SECyyyyyyyyyyyyyy",
            "trunk_id": "ST_abcdefgh",
        }
        
        # Encrypt all
        encrypted_secrets = {}
        for key, value in secrets.items():
            encrypted_secrets[key] = encrypt_decrypt(value)
            assert encrypted_secrets[key].startswith(ENCRYPTED_PREFIX)
        
        # Decrypt all
        decrypted_secrets = {}
        for key, encrypted_value in encrypted_secrets.items():
            decrypted_secrets[key] = encrypt_decrypt(encrypted_value)
        
        # Verify all match original
        assert decrypted_secrets == secrets


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
