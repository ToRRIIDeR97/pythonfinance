/**
 * Vue composable for WebSocket real-time data integration
 * Provides reactive state and easy subscription management
 */

import { ref, reactive, onMounted, onUnmounted, watch } from 'vue';
import webSocketService from '../services/websocket.js';

export function useWebSocket() {
  // Reactive state
  const isConnected = ref(false);
  const connectionStatus = ref('disconnected');
  const lastError = ref(null);
  const stats = reactive({
    totalConnections: 0,
    totalSubscriptions: 0,
    subscribedTickers: [],
    tickerSubscriberCounts: {}
  });

  // Real-time price data
  const priceData = reactive(new Map());
  const subscriptions = reactive(new Set());

  /**
   * Initialize WebSocket connection and event listeners
   */
  const initialize = () => {
    // Connection status listeners
    webSocketService.on('connected', (data) => {
      isConnected.value = true;
      connectionStatus.value = 'connected';
      lastError.value = null;
      console.log('[useWebSocket] Connected:', data);
    });

    webSocketService.on('disconnected', (data) => {
      isConnected.value = false;
      connectionStatus.value = 'disconnected';
      console.log('[useWebSocket] Disconnected:', data);
    });

    webSocketService.on('error', (error) => {
      lastError.value = error;
      connectionStatus.value = 'error';
      console.error('[useWebSocket] Error:', error);
    });

    webSocketService.on('maxReconnectAttemptsReached', () => {
      connectionStatus.value = 'failed';
      console.error('[useWebSocket] Max reconnection attempts reached');
    });

    // Data listeners
    webSocketService.on('priceUpdate', (message) => {
      const { ticker, data, timestamp } = message;
      
      // Update price data reactively
      priceData.set(ticker, {
        ...data,
        timestamp,
        lastUpdate: new Date(timestamp)
      });
    });

    webSocketService.on('subscriptionConfirmed', (message) => {
      subscriptions.add(message.ticker);
    });

    webSocketService.on('unsubscriptionConfirmed', (message) => {
      subscriptions.delete(message.ticker);
      priceData.delete(message.ticker);
    });

    webSocketService.on('stats', (data) => {
      Object.assign(stats, data);
    });

    webSocketService.on('serverError', (message) => {
      lastError.value = new Error(message.message);
    });

    // Auto-connect if not already connected
    if (!webSocketService.isConnected) {
      connect();
    } else {
      isConnected.value = true;
      connectionStatus.value = 'connected';
    }
  };

  /**
   * Connect to WebSocket server
   */
  const connect = async () => {
    try {
      connectionStatus.value = 'connecting';
      await webSocketService.connect();
    } catch (error) {
      lastError.value = error;
      connectionStatus.value = 'error';
      throw error;
    }
  };

  /**
   * Disconnect from WebSocket server
   */
  const disconnect = () => {
    webSocketService.disconnect();
    isConnected.value = false;
    connectionStatus.value = 'disconnected';
    priceData.clear();
    subscriptions.clear();
  };

  /**
   * Subscribe to ticker updates
   */
  const subscribe = (ticker) => {
    if (!ticker) return;
    
    const upperTicker = ticker.toUpperCase();
    webSocketService.subscribe(upperTicker);
    subscriptions.add(upperTicker);
  };

  /**
   * Unsubscribe from ticker updates
   */
  const unsubscribe = (ticker) => {
    if (!ticker) return;
    
    const upperTicker = ticker.toUpperCase();
    webSocketService.unsubscribe(upperTicker);
    subscriptions.delete(upperTicker);
    priceData.delete(upperTicker);
  };

  /**
   * Subscribe to multiple tickers
   */
  const subscribeToMultiple = (tickers) => {
    if (!Array.isArray(tickers)) return;
    
    tickers.forEach(ticker => {
      if (ticker) subscribe(ticker);
    });
  };

  /**
   * Unsubscribe from multiple tickers
   */
  const unsubscribeFromMultiple = (tickers) => {
    if (!Array.isArray(tickers)) return;
    
    tickers.forEach(ticker => {
      if (ticker) unsubscribe(ticker);
    });
  };

  /**
   * Get real-time price data for a ticker
   */
  const getPriceData = (ticker) => {
    if (!ticker) return null;
    return priceData.get(ticker.toUpperCase()) || null;
  };

  /**
   * Check if subscribed to a ticker
   */
  const isSubscribed = (ticker) => {
    if (!ticker) return false;
    return subscriptions.has(ticker.toUpperCase());
  };

  /**
   * Get connection statistics
   */
  const getStats = () => {
    webSocketService.getStats();
  };

  /**
   * Add custom event listener
   */
  const addEventListener = (event, callback) => {
    webSocketService.on(event, callback);
  };

  /**
   * Remove custom event listener
   */
  const removeEventListener = (event, callback) => {
    webSocketService.off(event, callback);
  };

  /**
   * Get current WebSocket status
   */
  const getStatus = () => {
    return {
      ...webSocketService.getStatus(),
      connectionStatus: connectionStatus.value,
      lastError: lastError.value,
      subscriptionsCount: subscriptions.size,
      priceDataCount: priceData.size
    };
  };

  // Lifecycle hooks
  onMounted(() => {
    initialize();
  });

  onUnmounted(() => {
    // Clean up subscriptions but don't disconnect
    // (other components might still be using the connection)
    subscriptions.clear();
    priceData.clear();
  });

  return {
    // Reactive state
    isConnected,
    connectionStatus,
    lastError,
    stats,
    priceData,
    subscriptions,
    
    // Methods
    connect,
    disconnect,
    subscribe,
    unsubscribe,
    subscribeToMultiple,
    unsubscribeFromMultiple,
    getPriceData,
    isSubscribed,
    getStats,
    addEventListener,
    removeEventListener,
    getStatus
  };
}

/**
 * Composable for ticker-specific real-time data
 */
export function useTickerData(ticker) {
  const { 
    isConnected, 
    subscribe, 
    unsubscribe, 
    getPriceData, 
    isSubscribed,
    addEventListener,
    removeEventListener
  } = useWebSocket();

  const tickerData = ref(null);
  const isLoading = ref(false);
  const error = ref(null);

  // Update ticker data when price updates arrive
  const updateTickerData = () => {
    if (ticker) {
      const data = getPriceData(ticker);
      if (data) {
        tickerData.value = data;
        isLoading.value = false;
        error.value = null;
      }
    }
  };

  // Watch for ticker changes
  watch(() => ticker, (newTicker, oldTicker) => {
    if (oldTicker) {
      unsubscribe(oldTicker);
    }
    if (newTicker) {
      subscribe(newTicker);
      updateTickerData();
    }
  }, { immediate: true });

  // Listen for price updates for this specific ticker
  const handlePriceUpdate = (message) => {
    if (message.ticker === ticker?.toUpperCase()) {
      updateTickerData();
    }
  };

  onMounted(() => {
    addEventListener('priceUpdate', handlePriceUpdate);
    if (ticker) {
      isLoading.value = true;
      subscribe(ticker);
    }
  });

  onUnmounted(() => {
    removeEventListener('priceUpdate', handlePriceUpdate);
    if (ticker) {
      unsubscribe(ticker);
    }
  });

  return {
    tickerData,
    isLoading,
    error,
    isConnected,
    isSubscribed: () => isSubscribed(ticker),
    refresh: () => {
      if (ticker) {
        isLoading.value = true;
        subscribe(ticker);
      }
    }
  };
}

/**
 * Composable for multiple tickers real-time data
 */
export function useMultipleTickersData(tickers) {
  const { 
    isConnected, 
    subscribeToMultiple, 
    unsubscribeFromMultiple, 
    priceData,
    addEventListener,
    removeEventListener
  } = useWebSocket();

  const tickersData = reactive(new Map());
  const isLoading = ref(false);
  const error = ref(null);

  // Watch for tickers changes
  watch(() => tickers, (newTickers, oldTickers) => {
    if (oldTickers) {
      unsubscribeFromMultiple(oldTickers);
    }
    if (newTickers && newTickers.length > 0) {
      isLoading.value = true;
      subscribeToMultiple(newTickers);
    }
  }, { immediate: true, deep: true });

  // Update tickers data when price updates arrive
  const updateTickersData = () => {
    if (tickers && Array.isArray(tickers)) {
      tickers.forEach(ticker => {
        const data = priceData.get(ticker?.toUpperCase());
        if (data) {
          tickersData.set(ticker.toUpperCase(), data);
        }
      });
      isLoading.value = false;
    }
  };

  // Listen for price updates
  const handlePriceUpdate = (message) => {
    if (tickers && tickers.some(t => t?.toUpperCase() === message.ticker)) {
      updateTickersData();
    }
  };

  onMounted(() => {
    addEventListener('priceUpdate', handlePriceUpdate);
    if (tickers && tickers.length > 0) {
      isLoading.value = true;
      subscribeToMultiple(tickers);
    }
  });

  onUnmounted(() => {
    removeEventListener('priceUpdate', handlePriceUpdate);
    if (tickers && tickers.length > 0) {
      unsubscribeFromMultiple(tickers);
    }
  });

  return {
    tickersData,
    isLoading,
    error,
    isConnected,
    refresh: () => {
      if (tickers && tickers.length > 0) {
        isLoading.value = true;
        subscribeToMultiple(tickers);
      }
    }
  };
}