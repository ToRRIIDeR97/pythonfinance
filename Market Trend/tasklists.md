# Project Task Lists

Below are actionable, trackable tasks derived from the WBS, UML, and Architectural Report. Each item is a checklist entry; I will update status to completed when you confirm tasks as done.

## Architecture & Planning
- [ ] Finalize system architecture document
- [ ] Define API contract (OpenAPI specification)
- [ ] Define security architecture (AuthN/AuthZ, secret management)
- [ ] Define monitoring, logging, and alerting strategy
- [ ] Define data backfill strategy and reconciliation plan
- [ ] Finalize ticker universe for indices and sector ETFs
- [ ] Develop UI/UX style guide and wireframes (dark/light modes)

## Backend API (FastAPI)
- [ ] Scaffold FastAPI application and project structure
- [ ] Implement health check endpoints
- [ ] Implement dependency injection for shared services
- [ ] Implement Pydantic models and response schemas
- [ ] Implement market indices endpoints
- [ ] Implement sector ETFs endpoints
- [ ] Implement summary endpoint with graceful degradation (200 + null on no data)
- [ ] Implement historical endpoint with graceful degradation (200 + empty on failure)
- [ ] Define and implement consistent error-handling policy across endpoints
- [ ] Generate interactive API documentation (Swagger/OpenAPI)

## Data Layer
- [ ] Implement PostgreSQL schemas for historical time-series data
- [ ] Implement Redis configuration for hot data caching
- [ ] Implement data persistence and caching logic (cold store + hot store)
- [ ] Implement read-through caching strategy for historical data
- [ ] Implement TTL-based caching for summary data

## External Data Adapter (yfinance)
- [ ] Implement YFinanceAdapter behind IDataProvider interface
- [ ] Implement robust error handling and retries for data fetches
- [ ] Implement request rate limiting (token bucket or queue)
- [ ] Validate fetched data against Pydantic schemas
- [ ] Log adapter failures with structured logging

## Security
- [ ] Implement secret management (Vault/KMS or secure env handling)
- [ ] Implement user authentication (OAuth2/JWT)
- [ ] Implement role-based authorization (RBAC)
- [ ] Implement API security middleware (CORS and headers hardening)
- [ ] Harden infrastructure and container images

## Monitoring & Logging
- [ ] Configure centralized logging pipeline (ELK/Datadog or equivalent)
- [ ] Instrument application performance monitoring (APM)
- [ ] Implement request/response logging with request IDs
- [ ] Implement health status monitoring and alerts
- [ ] Define alerting thresholds and notification channels

## Frontend (Vue.js)
- [ ] Scaffold Vue.js application
- [ ] Configure Vue Router for navigation
- [ ] Configure Pinia stores for state management
- [ ] Implement base layout and theme switching (dark/light)
- [ ] Build reusable components (MarketSelector, TimeframeSelector, KeyStatisticsCard)
- [ ] Integrate Highcharts Stock and configure client-side indicators
- [ ] Build MainChartWidget component
- [ ] Implement API client service with axios
- [ ] Wire components to Pinia stores and API client
- [ ] Implement error and loading states across views

## Data Population & Backfill
- [ ] Develop script to seed ticker universe
- [ ] Develop backfill orchestration scripts and scheduling
- [ ] Execute global indices backfill
- [ ] Execute U.S. sector ETFs backfill
- [ ] Validate and reconcile backfill data

## Testing & QA
- [ ] Develop test cases and scenarios (unit, integration)
- [ ] Execute backend unit testing
- [ ] Execute frontend unit and component testing
- [ ] Execute full-stack integration testing
- [ ] Execute performance and load testing for API
- [ ] Execute UI/UX and cross-browser compatibility testing
- [ ] Execute security testing (SAST, DAST, vulnerability scans, pen test)

## Deployment
- [ ] Configure Docker and docker-compose for environments
- [ ] Provision dev, staging, and prod infrastructure
- [ ] Build and test production images
- [ ] Execute go-live deployment plan
- [ ] Provide post-launch hypercare support

## Operations
- [ ] Monitor system health and performance
- [ ] Review application and security logs regularly
- [ ] Respond to system alerts
- [ ] Run change control process for scope/schedule/cost

## Documentation & Handover
- [ ] Finalize backend technical documentation
- [ ] Finalize frontend technical documentation
- [ ] Finalize operations and monitoring guides
- [ ] Finalize user and admin guides
- [ ] Conduct handover and training for maintenance team

## Enhanced Dashboard Features (New Priority Tasks)
- [ ] Design and implement enhanced Dashboard with mini-charts for each market index/ETF
- [ ] Create Market Comparison section with multi-ticker chart overlay functionality  
- [ ] Build interactive Market Heatmap with sector performance visualization
- [ ] Implement real-time price updates and WebSocket integration for live data
- [ ] Create advanced filtering and search capabilities across all dashboard sections
- [x] Build responsive grid layouts with drag-and-drop customization
 - [x] Implement saved filters functionality with local storage and user preferences
- [ ] Add portfolio tracking integration with P&L calculations
- [ ] Implement dashboard personalization and saved layouts

## Advanced Dashboard Components
- [ ] Create MiniChart component for dashboard cards with sparkline functionality
- [ ] Build ComparisonChart component with multi-series overlay capabilities
- [ ] Implement HeatmapChart component with interactive sector/performance visualization
- [ ] Create MarketSummaryCard component with real-time price updates
- [ ] Build FilterPanel component with advanced search and category filtering
- [ ] Implement DashboardGrid component with drag-and-drop layout customization
- [ ] Create PerformanceMetrics component for portfolio tracking
- [ ] Build UserPreferences component for dashboard personalization

## Backend API Enhancements for Dashboard
- [ ] Implement bulk market data endpoint for dashboard efficiency
- [ ] Create comparison endpoint for multi-ticker data retrieval
- [ ] Build heatmap data endpoint with sector categorization
- [ ] Implement WebSocket endpoints for real-time price streaming
- [ ] Create user preferences API for dashboard customization
- [ ] Add portfolio tracking endpoints with P&L calculations
- [ ] Implement market summary statistics endpoint
- [ ] Create sector performance aggregation endpoint

## Current Defects & Follow-ups
- [x] Fix Vue SFC invalid end tag in `Dashboard.vue` (compiler error)
- [x] Remove duplicated CSS outside `<style>` block in `Dashboard.vue`
- [x] Restart frontend dev server and validate preview loads without errors
- [x] Fix dashboard prices and mini charts by correcting API call parameters
- [x] Align summary fetching to unified `/api/v1/summary/{ticker}` endpoint
- [x] Map MarketCard summary fields to backend response names (price, change_pct)
- [x] Fix Highcharts treemap missing module error (#17) by initializing modules
- [ ] Fix health_check response encoding error in `main.py` (dict encode AttributeError)
- [ ] Investigate intermittent 502 responses for `/api/v1/summary/{ticker}` requests
- [ ] Investigate intermittent 502 responses for `/api/v1/markets/*/historical` requests
- [ ] Review yfinance failure handling and cache fallback for summary endpoints
- [ ] Align frontend error messaging with backend graceful degradation (200 + null/empty)
