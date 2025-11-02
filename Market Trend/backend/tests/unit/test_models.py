"""
Unit tests for database models
"""
import pytest
from datetime import datetime
from sqlalchemy.exc import IntegrityError

from app.models.user import User
from app.models.historical import HistoricalPrice


class TestUserModel:
    """Test User model functionality"""
    
    def test_create_user(self, db_session):
        """Test creating a new user"""
        user = User(
            email="test@example.com",
            username="testuser",
            hashed_password="hashedpassword123"
        )
        db_session.add(user)
        db_session.commit()
        
        assert user.id is not None
        assert user.email == "test@example.com"
        assert user.username == "testuser"
        assert user.is_active is True
        assert user.is_superuser is False
        assert isinstance(user.created_at, datetime)
    
    def test_user_email_unique(self, db_session):
        """Test that user email must be unique"""
        user1 = User(
            email="test@example.com",
            username="user1",
            hashed_password="password1"
        )
        user2 = User(
            email="test@example.com",
            username="user2",
            hashed_password="password2"
        )
        
        db_session.add(user1)
        db_session.commit()
        
        db_session.add(user2)
        with pytest.raises(IntegrityError):
            db_session.commit()
    
    def test_user_username_unique(self, db_session):
        """Test that username must be unique"""
        user1 = User(
            email="test1@example.com",
            username="testuser",
            hashed_password="password1"
        )
        user2 = User(
            email="test2@example.com",
            username="testuser",
            hashed_password="password2"
        )
        
        db_session.add(user1)
        db_session.commit()
        
        db_session.add(user2)
        with pytest.raises(IntegrityError):
            db_session.commit()
    
    def test_user_repr(self, db_session):
        """Test user string representation"""
        user = User(
            email="test@example.com",
            username="testuser",
            hashed_password="hashedpassword123"
        )
        db_session.add(user)
        db_session.commit()
        
        assert str(user) == f"<User(id={user.id}, username='testuser')>"


class TestHistoricalPriceModel:
    """Test HistoricalPrice model functionality"""
    
    def test_create_historical_price(self, db_session):
        """Test creating a historical price record"""
        price = HistoricalPrice(
            ticker="AAPL",
            date=datetime(2023, 1, 1),
            open=150.0,
            high=155.0,
            low=149.0,
            close=154.0,
            volume=1000000
        )
        db_session.add(price)
        db_session.commit()
        
        assert price.id is not None
        assert price.ticker == "AAPL"
        assert price.open == 150.0
        assert price.close == 154.0
    
    def test_historical_price_unique_constraint(self, db_session):
        """Test that ticker+date+interval combination must be unique"""
        date = datetime(2023, 1, 1)
        
        price1 = HistoricalPrice(
            ticker="AAPL",
            date=date,
            interval="1d",
            open=150.0,
            high=155.0,
            low=149.0,
            close=154.0,
            volume=1000000
        )
        price2 = HistoricalPrice(
            ticker="AAPL",
            date=date,
            interval="1d",
            open=151.0,
            high=156.0,
            low=150.0,
            close=155.0,
            volume=1100000
        )
        
        db_session.add(price1)
        db_session.commit()
        
        db_session.add(price2)
        with pytest.raises(IntegrityError):
            db_session.commit()