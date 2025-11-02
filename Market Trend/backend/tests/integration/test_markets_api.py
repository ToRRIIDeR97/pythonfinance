"""
Integration tests for market data API endpoints
"""
import pytest
from fastapi import status


class TestMarketIndices:
    """Test market indices endpoints"""
    
    def test_list_indices(self, client):
        """Test listing market indices"""
        response = client.get("/api/v1/markets/indices")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0
        
        # Check structure of first index
        first_index = data[0]
        assert "ticker" in first_index
        assert "name" in first_index
        assert "region" in first_index
        
        # Verify we have expected indices
        tickers = [index["ticker"] for index in data]
        assert "^GSPC" in tickers  # S&P 500
    
    def test_get_index_historical_default_params(self, client):
        """Test getting historical data with default parameters"""
        response = client.get("/api/v1/markets/indices/^GSPC/historical")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["ticker"] == "^GSPC"
        assert data["period"] == "1y"
        assert data["interval"] == "1d"
        assert "data" in data
    
    def test_get_index_historical_custom_params(self, client):
        """Test getting historical data with custom parameters"""
        response = client.get(
            "/api/v1/markets/indices/^GSPC/historical",
            params={"period": "5d", "interval": "1h"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["ticker"] == "^GSPC"
        assert data["period"] == "5d"
        assert data["interval"] == "1h"
    
    def test_get_invalid_index_historical(self, client):
        """Test getting historical data for invalid ticker"""
        response = client.get("/api/v1/markets/indices/INVALID/historical")
        
        # Should handle gracefully - either 404 or empty data
        assert response.status_code in [status.HTTP_404_NOT_FOUND, status.HTTP_200_OK]


class TestHealthEndpoint:
    """Test health check endpoint"""
    
    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        
        assert response.status_code == 200
        assert response.json()["status"] == "ok"