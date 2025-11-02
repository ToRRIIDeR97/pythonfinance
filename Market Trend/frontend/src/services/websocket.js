/**
 * WebSocket service for real-time market data streaming
 * Handles connections, subscriptions, and data broadcasting
 */

class WebSocketService {
  constructor() {
    this.ws = null;
    this.connectionId = null;
    this.isConnected = false;
    this.reconnectAttempts = 0;
    this.maxReconnectAttempts = 5;
    this.reconnectDelay = 1000; // Start with 1 second
    this.maxReconnectDelay = 30000; // Max 30 seconds
    this.subscriptions = new Set();
    this.listeners = new Map();
    this.messageQueue = [];
    this.heartbeatInterval = null;
    this.heartbeatTimeout = null;
    
    // Configuration
    this.config = {
      url: import.meta.env.VITE_WS_URL || 'ws://localhost:8000/api/v1/ws',
      heartbeatInterval: 30000, // 30 seconds
      heartbeatTimeout: 5000, // 5 seconds
    };
  }

  /**
   * Generate a unique connection ID
   */
  generateConnectionId() {
    return `conn_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }

  /**
   * Connect to WebSocket server
   */
  async connect() {
    if (this.isConnected || (this.ws && this.ws.readyState === WebSocket.CONNECTING)) {
      return Promise.resolve();
    }

    return new Promise((resolve, reject) => {
      try {
        this.connectionId = this.generateConnectionId();
        const wsUrl = `${this.config.url}/${this.connectionId}`;
        
        console.log(`[WebSocket] Connecting to ${wsUrl}`);
        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
          console.log('[WebSocket] Connected successfully');
          this.isConnected = true;
          this.reconnectAttempts = 0;
          this.reconnectDelay = 1000;
          
          // Process queued messages
          this.processMessageQueue();
          
          // Start heartbeat
          this.startHeartbeat();
          
          // Re-subscribe to previous subscriptions
          this.resubscribeAll();
          
          this.emit('connected', { connectionId: this.connectionId });
          resolve();
        };

        this.ws.onmessage = (event) => {
          try {
            const message = JSON.parse(event.data);
            this.handleMessage(message);
          } catch (error) {
            console.error('[WebSocket] Error parsing message:', error);
          }
        };

        this.ws.onclose = (event) => {
          console.log('[WebSocket] Connection closed:', event.code, event.reason);
          this.isConnected = false;
          this.stopHeartbeat();
          
          this.emit('disconnected', { 
            code: event.code, 
            reason: event.reason,
            wasClean: event.wasClean 
          });
          
          // Attempt reconnection if not a clean close
          if (!event.wasClean && this.reconnectAttempts < this.maxReconnectAttempts) {
            this.scheduleReconnect();
          }
        };

        this.ws.onerror = (error) => {
          console.error('[WebSocket] Connection error:', error);
          this.emit('error', error);
          reject(error);
        };

      } catch (error) {
        console.error('[WebSocket] Failed to create connection:', error);
        reject(error);
      }
    });
  }

  /**
   * Disconnect from WebSocket server
   */
  disconnect() {
    if (this.ws) {
      this.ws.close(1000, 'Client disconnect');
      this.ws = null;
    }
    this.isConnected = false;
    this.stopHeartbeat();
    this.subscriptions.clear();
    this.messageQueue = [];
  }

  /**
   * Schedule reconnection with exponential backoff
   */
  scheduleReconnect() {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('[WebSocket] Max reconnection attempts reached');
      this.emit('maxReconnectAttemptsReached');
      return;
    }

    this.reconnectAttempts++;
    const delay = Math.min(
      this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1),
      this.maxReconnectDelay
    );

    console.log(`[WebSocket] Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);
    
    setTimeout(() => {
      if (!this.isConnected) {
        this.connect().catch(error => {
          console.error('[WebSocket] Reconnection failed:', error);
        });
      }
    }, delay);
  }

  /**
   * Start heartbeat to keep connection alive
   */
  startHeartbeat() {
    this.stopHeartbeat();
    
    this.heartbeatInterval = setInterval(() => {
      if (this.isConnected) {
        this.send({ type: 'ping' });
        
        // Set timeout for pong response
        this.heartbeatTimeout = setTimeout(() => {
          console.warn('[WebSocket] Heartbeat timeout - connection may be lost');
          this.ws?.close();
        }, this.config.heartbeatTimeout);
      }
    }, this.config.heartbeatInterval);
  }

  /**
   * Stop heartbeat
   */
  stopHeartbeat() {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
    if (this.heartbeatTimeout) {
      clearTimeout(this.heartbeatTimeout);
      this.heartbeatTimeout = null;
    }
  }

  /**
   * Handle incoming WebSocket messages
   */
  handleMessage(message) {
    const { type } = message;

    switch (type) {
      case 'connection_established':
        console.log('[WebSocket] Connection established:', message);
        this.emit('connectionEstablished', message);
        break;

      case 'price_update':
        this.emit('priceUpdate', message);
        this.emit(`priceUpdate:${message.ticker}`, message);
        break;

      case 'subscription_confirmed':
        console.log('[WebSocket] Subscription confirmed:', message.ticker);
        this.emit('subscriptionConfirmed', message);
        break;

      case 'unsubscription_confirmed':
        console.log('[WebSocket] Unsubscription confirmed:', message.ticker);
        this.subscriptions.delete(message.ticker);
        this.emit('unsubscriptionConfirmed', message);
        break;

      case 'error':
        console.error('[WebSocket] Server error:', message.message);
        this.emit('serverError', message);
        break;

      case 'pong':
        // Clear heartbeat timeout on pong response
        if (this.heartbeatTimeout) {
          clearTimeout(this.heartbeatTimeout);
          this.heartbeatTimeout = null;
        }
        break;

      case 'stats':
        this.emit('stats', message.data);
        break;

      case 'broadcast':
        this.emit('broadcast', message);
        break;

      default:
        console.log('[WebSocket] Unknown message type:', type, message);
        this.emit('message', message);
    }
  }

  /**
   * Send message to WebSocket server
   */
  send(message) {
    if (this.isConnected && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    } else {
      // Queue message for later sending
      this.messageQueue.push(message);
    }
  }

  /**
   * Process queued messages
   */
  processMessageQueue() {
    while (this.messageQueue.length > 0) {
      const message = this.messageQueue.shift();
      this.send(message);
    }
  }

  /**
   * Subscribe to ticker updates
   */
  subscribe(ticker) {
    const upperTicker = ticker.toUpperCase();
    this.subscriptions.add(upperTicker);
    
    this.send({
      type: 'subscribe',
      ticker: upperTicker
    });

    console.log(`[WebSocket] Subscribed to ${upperTicker}`);
  }

  /**
   * Unsubscribe from ticker updates
   */
  unsubscribe(ticker) {
    const upperTicker = ticker.toUpperCase();
    this.subscriptions.delete(upperTicker);
    
    this.send({
      type: 'unsubscribe',
      ticker: upperTicker
    });

    console.log(`[WebSocket] Unsubscribed from ${upperTicker}`);
  }

  /**
   * Re-subscribe to all previous subscriptions
   */
  resubscribeAll() {
    for (const ticker of this.subscriptions) {
      this.send({
        type: 'subscribe',
        ticker: ticker
      });
    }
  }

  /**
   * Get connection statistics
   */
  getStats() {
    this.send({ type: 'get_stats' });
  }

  /**
   * Add event listener
   */
  on(event, callback) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event).add(callback);
  }

  /**
   * Remove event listener
   */
  off(event, callback) {
    if (this.listeners.has(event)) {
      this.listeners.get(event).delete(callback);
    }
  }

  /**
   * Emit event to listeners
   */
  emit(event, data) {
    if (this.listeners.has(event)) {
      for (const callback of this.listeners.get(event)) {
        try {
          callback(data);
        } catch (error) {
          console.error(`[WebSocket] Error in event listener for ${event}:`, error);
        }
      }
    }
  }

  /**
   * Get connection status
   */
  getStatus() {
    return {
      isConnected: this.isConnected,
      connectionId: this.connectionId,
      subscriptions: Array.from(this.subscriptions),
      reconnectAttempts: this.reconnectAttempts,
      readyState: this.ws?.readyState,
      queuedMessages: this.messageQueue.length
    };
  }
}

// Create singleton instance
const webSocketService = new WebSocketService();

// Auto-connect on page load
if (typeof window !== 'undefined') {
  // Connect when the page loads
  document.addEventListener('DOMContentLoaded', () => {
    webSocketService.connect().catch(error => {
      console.error('[WebSocket] Initial connection failed:', error);
    });
  });

  // Disconnect when page unloads
  window.addEventListener('beforeunload', () => {
    webSocketService.disconnect();
  });

  // Handle visibility changes (pause/resume when tab is hidden/visible)
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      console.log('[WebSocket] Page hidden - maintaining connection');
    } else {
      console.log('[WebSocket] Page visible - ensuring connection');
      if (!webSocketService.isConnected) {
        webSocketService.connect().catch(error => {
          console.error('[WebSocket] Reconnection on visibility change failed:', error);
        });
      }
    }
  });
}

export default webSocketService;