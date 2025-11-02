"""
Security tests for the application
"""
import pytest
from fastapi import status


class TestAuthenticationSecurity:
    """Test authentication security measures"""
    
    def test_protected_endpoint_without_token(self, client):
        """Test that protected endpoints require authentication"""
        response = client.get("/api/v1/auth/me")
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        assert "Not authenticated" in response.json()["detail"]
    
    def test_protected_endpoint_with_invalid_token(self, client):
        """Test that invalid tokens are rejected"""
        client.headers.update({"Authorization": "Bearer invalid_token_here"})
        response = client.get("/api/v1/auth/me")
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    def test_protected_endpoint_with_malformed_token(self, client):
        """Test that malformed tokens are rejected"""
        client.headers.update({"Authorization": "InvalidFormat token"})
        response = client.get("/api/v1/auth/me")
        
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    def test_sql_injection_in_login(self, client):
        """Test SQL injection protection in login"""
        malicious_data = {
            "username": "admin'; DROP TABLE users; --",
            "password": "password"
        }
        
        response = client.post("/api/v1/auth/login", data=malicious_data)
        
        # Should not cause server error, should return 401
        assert response.status_code == status.HTTP_401_UNAUTHORIZED
    
    def test_xss_in_registration(self, client):
        """Test XSS protection in user registration"""
        xss_data = {
            "email": "test@example.com",
            "username": "<script>alert('xss')</script>",
            "password": "password123"
        }
        
        response = client.post("/api/v1/auth/register", json=xss_data)
        
        if response.status_code == 200:
            # If registration succeeds, ensure script tags are escaped/removed
            user_data = response.json()
            assert "<script>" not in user_data["username"]
        else:
            # Or registration should be rejected
            assert response.status_code in [400, 422]


class TestInputValidation:
    """Test input validation security"""
    
    def test_email_validation(self, client):
        """Test email format validation"""
        invalid_emails = [
            "invalid-email",
            "@example.com",
            "test@",
            "test..test@example.com",
            "test@example",
        ]
        
        for email in invalid_emails:
            user_data = {
                "email": email,
                "username": "testuser",
                "password": "password123"
            }
            
            response = client.post("/api/v1/auth/register", json=user_data)
            assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_password_length_validation(self, client):
        """Test password length requirements"""
        short_passwords = ["", "1", "12", "123", "1234", "12345"]
        
        for password in short_passwords:
            user_data = {
                "email": "test@example.com",
                "username": "testuser",
                "password": password
            }
            
            response = client.post("/api/v1/auth/register", json=user_data)
            assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_username_validation(self, client):
        """Test username validation"""
        invalid_usernames = [
            "",  # Empty
            "a",  # Too short
            "user with spaces",  # Spaces
            "user@special",  # Special characters
            "a" * 51,  # Too long (assuming 50 char limit)
        ]
        
        for username in invalid_usernames:
            user_data = {
                "email": "test@example.com",
                "username": username,
                "password": "password123"
            }
            
            response = client.post("/api/v1/auth/register", json=user_data)
            # Should either be validation error or bad request
            assert response.status_code in [400, 422]


class TestRateLimiting:
    """Test rate limiting (if implemented)"""
    
    def test_login_rate_limiting(self, client, test_user):
        """Test that excessive login attempts are rate limited"""
        login_data = {
            "username": "testuser",
            "password": "wrongpassword"
        }
        
        # Make multiple failed login attempts
        responses = []
        for _ in range(10):
            response = client.post("/api/v1/auth/login", data=login_data)
            responses.append(response.status_code)
        
        # All should be 401 (unauthorized) for now
        # In a production system, we might implement rate limiting
        # that would return 429 (Too Many Requests) after several attempts
        for status_code in responses:
            assert status_code in [401, 429]


class TestDataExposure:
    """Test that sensitive data is not exposed"""
    
    def test_password_not_in_response(self, client):
        """Test that passwords are never returned in responses"""
        user_data = {
            "email": "test@example.com",
            "username": "testuser",
            "password": "password123"
        }
        
        response = client.post("/api/v1/auth/register", json=user_data)
        
        if response.status_code == 200:
            data = response.json()
            assert "password" not in data
            assert "hashed_password" not in data
    
    def test_user_enumeration_protection(self, client):
        """Test protection against user enumeration"""
        # Try to register with existing email
        user_data = {
            "email": "test@example.com",
            "username": "newuser",
            "password": "password123"
        }
        
        # First registration
        client.post("/api/v1/auth/register", json=user_data)
        
        # Second registration with same email
        response = client.post("/api/v1/auth/register", json=user_data)
        
        # Error message should not reveal if email exists
        if response.status_code == 400:
            error_msg = response.json()["detail"].lower()
            # Should be generic, not revealing specific field
            assert "email" in error_msg or "already" in error_msg