"""
Performance tests for API endpoints
"""
import time
import pytest
from concurrent.futures import ThreadPoolExecutor, as_completed


class TestAPIPerformance:
    """Test API endpoint performance"""
    
    def test_health_endpoint_response_time(self, client):
        """Test health endpoint response time"""
        start_time = time.time()
        response = client.get("/health")
        end_time = time.time()
        
        response_time = end_time - start_time
        
        assert response.status_code == 200
        assert response_time < 0.1  # Should respond within 100ms
    
    def test_indices_list_response_time(self, client):
        """Test indices list endpoint response time"""
        start_time = time.time()
        response = client.get("/api/v1/markets/indices")
        end_time = time.time()
        
        response_time = end_time - start_time
        
        assert response.status_code == 200
        assert response_time < 0.5  # Should respond within 500ms
    
    def test_concurrent_health_requests(self, client):
        """Test concurrent requests to health endpoint"""
        def make_request():
            start_time = time.time()
            response = client.get("/health")
            end_time = time.time()
            return response.status_code, end_time - start_time
        
        # Make 10 concurrent requests
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [future.result() for future in as_completed(futures)]
        
        # All requests should succeed
        for status_code, response_time in results:
            assert status_code == 200
            assert response_time < 1.0  # Each request should complete within 1 second
    
    def test_authentication_performance(self, client, test_user):
        """Test authentication endpoint performance"""
        login_data = {
            "username": "testuser",
            "password": "testpassword123"
        }
        
        start_time = time.time()
        response = client.post("/api/v1/auth/login", data=login_data)
        end_time = time.time()
        
        response_time = end_time - start_time
        
        assert response.status_code == 200
        assert response_time < 1.0  # Authentication should complete within 1 second
    
    @pytest.mark.slow
    def test_load_test_registration(self, client):
        """Load test user registration endpoint"""
        def register_user(user_id):
            user_data = {
                "email": f"user{user_id}@example.com",
                "username": f"user{user_id}",
                "password": "password123"
            }
            start_time = time.time()
            response = client.post("/api/v1/auth/register", json=user_data)
            end_time = time.time()
            return response.status_code, end_time - start_time
        
        # Register 20 users concurrently
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(register_user, i) for i in range(20)]
            results = [future.result() for future in as_completed(futures)]
        
        successful_registrations = 0
        total_time = 0
        
        for status_code, response_time in results:
            if status_code == 200:
                successful_registrations += 1
                total_time += response_time
        
        # At least 90% should succeed
        assert successful_registrations >= 18
        
        # Average response time should be reasonable
        if successful_registrations > 0:
            avg_response_time = total_time / successful_registrations
            assert avg_response_time < 2.0  # Average should be under 2 seconds