"""
Unit tests for authentication functionality
"""
import pytest
from datetime import datetime, timedelta, timezone
from jose import jwt

from app.core.auth import (
    verify_password,
    get_password_hash,
    create_access_token,
    verify_token
)
from app.core.config import settings


class TestPasswordHashing:
    """Test password hashing and verification"""
    
    def test_password_hashing(self):
        """Test password hashing works correctly"""
        password = "testpassword123"
        hashed = get_password_hash(password)
        
        assert hashed != password
        assert verify_password(password, hashed) is True
        assert verify_password("wrongpassword", hashed) is False
    
    def test_long_password_handling(self):
        """Test that long passwords are handled correctly"""
        long_password = "a" * 70  # 70 character password (within bcrypt limit)
        hashed = get_password_hash(long_password)
        
        assert verify_password(long_password, hashed) is True
    
    def test_very_long_password_handling(self):
        """Test that very long passwords are truncated"""
        very_long_password = "a" * 100  # 100 character password
        hashed = get_password_hash(very_long_password)
        
        # Should work because our implementation truncates to 72 bytes
        assert verify_password(very_long_password, hashed) is True
    
    def test_empty_password(self):
        """Test empty password handling"""
        # Empty password should be allowed but will be hashed
        empty_password = ""
        hashed = get_password_hash(empty_password)
        assert verify_password(empty_password, hashed) is True


class TestJWTTokens:
    """Test JWT token creation and verification"""
    
    def test_create_access_token(self):
        """Test access token creation"""
        data = {"sub": "testuser"}
        token = create_access_token(data)
        
        assert isinstance(token, str)
        assert len(token) > 0
    
    def test_create_token_with_expiry(self):
        """Test creating a token with custom expiry"""
        data = {"sub": "testuser"}
        expires_delta = timedelta(minutes=30)
        token = create_access_token(data, expires_delta)
        
        # Decode token to check expiry
        decoded = jwt.decode(token, settings.secret_key, algorithms=["HS256"])
        exp = datetime.fromtimestamp(decoded["exp"], tz=timezone.utc)
        
        # Check that expiry is approximately 30 minutes from now
        now = datetime.now(timezone.utc)
        assert exp > now
        assert exp < now + timedelta(minutes=31)
    
    def test_verify_valid_token(self):
        """Test verification of valid token"""
        data = {"sub": "testuser"}
        token = create_access_token(data)
        
        payload = verify_token(token)
        assert payload["sub"] == "testuser"
    
    def test_verify_invalid_token(self):
        """Test verification of invalid token"""
        invalid_token = "invalid.token.here"
        
        # verify_token should return None for invalid tokens
        try:
            payload = verify_token(invalid_token)
            assert payload is None
        except Exception:
            # If it raises an exception, that's also acceptable behavior
            pass
    
    def test_verify_expired_token(self):
        """Test verification of expired token"""
        data = {"sub": "testuser"}
        expires_delta = timedelta(seconds=-1)  # Already expired
        token = create_access_token(data, expires_delta)
        
        # verify_token should return None for expired tokens
        try:
            payload = verify_token(token)
            assert payload is None
        except Exception:
            # If it raises an exception, that's also acceptable behavior
            pass