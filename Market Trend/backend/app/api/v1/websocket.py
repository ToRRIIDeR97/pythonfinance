"""
WebSocket API for real-time market data streaming.
Provides live price updates, market events, and real-time notifications.
"""

import asyncio
import json
from typing import Dict, List, Set, Optional, Any
from datetime import datetime, timedelta
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException
from fastapi.websockets import WebSocketState
from sqlalchemy.orm import Session

from ...core.db import get_db
from ...core.logging import get_logger
from ...repository.data_repository import DataRepository
from ...data.tickers import GLOBAL_INDICES, US_SECTOR_ETFS

router = APIRouter()
logger = get_logger("websocket")


class ConnectionManager:
    """Manages WebSocket connections and broadcasting."""
    
    def __init__(self):
        # Active connections by connection ID
        self.active_connections: Dict[str, WebSocket] = {}
        # Subscriptions by ticker symbol
        self.subscriptions: Dict[str, Set[str]] = {}
        # Connection metadata
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
        
    async def connect(self, websocket: WebSocket, connection_id: str, metadata: Dict[str, Any] = None):
        """Accept a new WebSocket connection."""
        await websocket.accept()
        self.active_connections[connection_id] = websocket
        self.connection_metadata[connection_id] = metadata or {}
        
        logger.info(
            "WebSocket connection established",
            connection_id=connection_id,
            total_connections=len(self.active_connections)
        )
        
        # Send welcome message
        await self.send_personal_message({
            "type": "connection_established",
            "connection_id": connection_id,
            "timestamp": datetime.utcnow().isoformat(),
            # Tickers are defined as lists of dicts; extract symbols safely
            "available_tickers": [i.get("ticker") for i in GLOBAL_INDICES if i.get("ticker")] +
                                 [e.get("ticker") for e in US_SECTOR_ETFS if e.get("ticker")]
        }, connection_id)
    
    def disconnect(self, connection_id: str):
        """Remove a WebSocket connection."""
        if connection_id in self.active_connections:
            # Remove from all subscriptions
            for ticker in list(self.subscriptions.keys()):
                if connection_id in self.subscriptions[ticker]:
                    self.subscriptions[ticker].discard(connection_id)
                    if not self.subscriptions[ticker]:
                        del self.subscriptions[ticker]
            
            # Remove connection
            del self.active_connections[connection_id]
            if connection_id in self.connection_metadata:
                del self.connection_metadata[connection_id]
            
            logger.info(
                "WebSocket connection closed",
                connection_id=connection_id,
                total_connections=len(self.active_connections)
            )
    
    async def send_personal_message(self, message: Dict[str, Any], connection_id: str):
        """Send a message to a specific connection."""
        if connection_id in self.active_connections:
            websocket = self.active_connections[connection_id]
            try:
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_text(json.dumps(message))
                else:
                    self.disconnect(connection_id)
            except Exception as e:
                logger.error(
                    "Failed to send personal message",
                    connection_id=connection_id,
                    error=str(e)
                )
                self.disconnect(connection_id)
    
    async def broadcast_to_subscribers(self, ticker: str, message: Dict[str, Any]):
        """Broadcast a message to all subscribers of a ticker."""
        if ticker in self.subscriptions:
            disconnected_connections = []
            
            for connection_id in self.subscriptions[ticker].copy():
                websocket = self.active_connections.get(connection_id)
                if websocket and websocket.client_state == WebSocketState.CONNECTED:
                    try:
                        await websocket.send_text(json.dumps(message))
                    except Exception as e:
                        logger.error(
                            "Failed to broadcast message",
                            connection_id=connection_id,
                            ticker=ticker,
                            error=str(e)
                        )
                        disconnected_connections.append(connection_id)
                else:
                    disconnected_connections.append(connection_id)
            
            # Clean up disconnected connections
            for connection_id in disconnected_connections:
                self.disconnect(connection_id)
    
    def subscribe_to_ticker(self, connection_id: str, ticker: str):
        """Subscribe a connection to ticker updates."""
        if ticker not in self.subscriptions:
            self.subscriptions[ticker] = set()
        self.subscriptions[ticker].add(connection_id)
        
        logger.info(
            "Connection subscribed to ticker",
            connection_id=connection_id,
            ticker=ticker,
            subscribers_count=len(self.subscriptions[ticker])
        )
    
    def unsubscribe_from_ticker(self, connection_id: str, ticker: str):
        """Unsubscribe a connection from ticker updates."""
        if ticker in self.subscriptions:
            self.subscriptions[ticker].discard(connection_id)
            if not self.subscriptions[ticker]:
                del self.subscriptions[ticker]
        
        logger.info(
            "Connection unsubscribed from ticker",
            connection_id=connection_id,
            ticker=ticker
        )
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        return {
            "total_connections": len(self.active_connections),
            "total_subscriptions": sum(len(subs) for subs in self.subscriptions.values()),
            "subscribed_tickers": list(self.subscriptions.keys()),
            "ticker_subscriber_counts": {
                ticker: len(subs) for ticker, subs in self.subscriptions.items()
            }
        }


# Global connection manager instance
manager = ConnectionManager()


class RealTimeDataService:
    """Service for fetching and broadcasting real-time market data."""
    
    def __init__(self):
        self.data_repo = DataRepository()
        self.last_update_times: Dict[str, datetime] = {}
        self.update_interval = 30  # seconds
        self.is_running = False
        
    async def start_data_streaming(self, db: Session):
        """Start the real-time data streaming loop."""
        if self.is_running:
            return
            
        self.is_running = True
        logger.info("Starting real-time data streaming service")
        
        while self.is_running:
            try:
                # Get all subscribed tickers
                subscribed_tickers = list(manager.subscriptions.keys())
                
                if subscribed_tickers:
                    # Fetch latest data for subscribed tickers
                    for ticker in subscribed_tickers:
                        await self.update_ticker_data(db, ticker)
                
                # Wait before next update
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.error("Error in data streaming loop", error=str(e))
                await asyncio.sleep(5)  # Short delay before retry
    
    async def update_ticker_data(self, db: Session, ticker: str):
        """Update and broadcast data for a specific ticker."""
        try:
            # Check if we need to update this ticker
            now = datetime.utcnow()
            last_update = self.last_update_times.get(ticker)
            
            if last_update and (now - last_update).seconds < self.update_interval:
                return
            
            # Fetch latest data
            summary_data = self.data_repo.get_summary(db, ticker, period="1d", interval="1m")
            
            if summary_data:
                # Prepare real-time update message
                message = {
                    "type": "price_update",
                    "ticker": ticker,
                    "timestamp": now.isoformat(),
                    "data": {
                        # Align keys with frontend expectations while mapping from summary fields
                        "current_price": summary_data.get("price"),
                        "change": summary_data.get("change"),
                        "change_percent": summary_data.get("change_pct"),
                        "volume": summary_data.get("volume") or summary_data.get("volume_avg_20d"),
                        "market_cap": summary_data.get("market_cap"),
                        "high_52w": summary_data.get("high_52w"),
                        "low_52w": summary_data.get("low_52w")
                    }
                }
                
                # Broadcast to subscribers
                await manager.broadcast_to_subscribers(ticker, message)
                
                # Update last update time
                self.last_update_times[ticker] = now
                
        except Exception as e:
            logger.error(
                "Error updating ticker data",
                ticker=ticker,
                error=str(e)
            )
    
    def stop_data_streaming(self):
        """Stop the real-time data streaming service."""
        self.is_running = False
        logger.info("Stopping real-time data streaming service")


# Global data service instance
data_service = RealTimeDataService()


@router.websocket("/ws/{connection_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    connection_id: str,
    db: Session = Depends(get_db)
):
    """WebSocket endpoint for real-time market data."""
    await manager.connect(websocket, connection_id)
    
    # Start data streaming if not already running
    if not data_service.is_running:
        asyncio.create_task(data_service.start_data_streaming(db))
    
    try:
        while True:
            # Receive messages from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            message_type = message.get("type")
            
            if message_type == "subscribe":
                # Subscribe to ticker updates
                ticker = message.get("ticker", "").upper()
                # Build allowed ticker sets from lists of dicts
                index_tickers = {i.get("ticker") for i in GLOBAL_INDICES if i.get("ticker")}
                etf_tickers = {e.get("ticker") for e in US_SECTOR_ETFS if e.get("ticker")}
                if ticker in index_tickers or ticker in etf_tickers:
                    manager.subscribe_to_ticker(connection_id, ticker)
                    
                    # Send immediate data if available
                    await data_service.update_ticker_data(db, ticker)
                    
                    await manager.send_personal_message({
                        "type": "subscription_confirmed",
                        "ticker": ticker,
                        "timestamp": datetime.utcnow().isoformat()
                    }, connection_id)
                else:
                    await manager.send_personal_message({
                        "type": "error",
                        "message": f"Invalid ticker: {ticker}",
                        "timestamp": datetime.utcnow().isoformat()
                    }, connection_id)
            
            elif message_type == "unsubscribe":
                # Unsubscribe from ticker updates
                ticker = message.get("ticker", "").upper()
                manager.unsubscribe_from_ticker(connection_id, ticker)
                
                await manager.send_personal_message({
                    "type": "unsubscription_confirmed",
                    "ticker": ticker,
                    "timestamp": datetime.utcnow().isoformat()
                }, connection_id)
            
            elif message_type == "get_stats":
                # Send connection statistics
                stats = manager.get_connection_stats()
                await manager.send_personal_message({
                    "type": "stats",
                    "data": stats,
                    "timestamp": datetime.utcnow().isoformat()
                }, connection_id)
            
            elif message_type == "ping":
                # Respond to ping with pong
                await manager.send_personal_message({
                    "type": "pong",
                    "timestamp": datetime.utcnow().isoformat()
                }, connection_id)
            
    except WebSocketDisconnect:
        manager.disconnect(connection_id)
    except Exception as e:
        logger.error(
            "WebSocket error",
            connection_id=connection_id,
            error=str(e)
        )
        manager.disconnect(connection_id)


@router.get("/ws/stats")
async def get_websocket_stats():
    """Get WebSocket connection statistics."""
    return {
        "stats": manager.get_connection_stats(),
        "data_service_running": data_service.is_running,
        "update_interval_seconds": data_service.update_interval
    }


@router.post("/ws/broadcast")
async def broadcast_message(
    message: Dict[str, Any],
    ticker: Optional[str] = None
):
    """Broadcast a message to WebSocket connections."""
    if ticker:
        # Broadcast to specific ticker subscribers
        await manager.broadcast_to_subscribers(ticker.upper(), {
            "type": "broadcast",
            "message": message,
            "timestamp": datetime.utcnow().isoformat()
        })
    else:
        # Broadcast to all connections
        for connection_id in manager.active_connections:
            await manager.send_personal_message({
                "type": "broadcast",
                "message": message,
                "timestamp": datetime.utcnow().isoformat()
            }, connection_id)
    
    return {"status": "Message broadcasted successfully"}
