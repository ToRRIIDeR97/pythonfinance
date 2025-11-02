<template>
  <div class="loading-spinner-container" :class="containerClass">
    <div class="loading-spinner" :class="spinnerClass" :style="spinnerStyle">
      <div class="spinner-inner"></div>
    </div>
    <p v-if="message" class="loading-message" :class="messageClass">{{ message }}</p>
  </div>
</template>

<script>
export default {
  name: 'LoadingSpinner',
  props: {
    size: {
      type: String,
      default: 'medium',
      validator: value => ['small', 'medium', 'large', 'xlarge'].includes(value)
    },
    color: {
      type: String,
      default: 'primary'
    },
    message: {
      type: String,
      default: ''
    },
    centered: {
      type: Boolean,
      default: true
    },
    overlay: {
      type: Boolean,
      default: false
    }
  },
  computed: {
    containerClass() {
      return {
        'centered': this.centered,
        'overlay': this.overlay,
        [`size-${this.size}`]: true
      };
    },
    spinnerClass() {
      return {
        [`spinner-${this.size}`]: true,
        [`spinner-${this.color}`]: true
      };
    },
    messageClass() {
      return {
        [`message-${this.size}`]: true
      };
    },
    spinnerStyle() {
      // Allow custom color override
      if (this.color.startsWith('#') || this.color.startsWith('rgb')) {
        return {
          '--spinner-color': this.color
        };
      }
      return {};
    }
  }
}
</script>

<style scoped>
.loading-spinner-container {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: var(--space-4);
}

.loading-spinner-container.centered {
  justify-content: center;
  min-height: 200px;
}

.loading-spinner-container.overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: rgba(255, 255, 255, 0.8);
  z-index: var(--z-overlay);
  min-height: 100vh;
}

.dark .loading-spinner-container.overlay {
  background-color: rgba(0, 0, 0, 0.8);
}

.loading-spinner {
  border-radius: 50%;
  animation: spin 1s linear infinite;
  position: relative;
}

.spinner-inner {
  width: 100%;
  height: 100%;
  border-radius: 50%;
  border: 3px solid var(--color-border);
  border-top: 3px solid var(--spinner-color, var(--color-primary));
}

/* Size variants */
.spinner-small {
  width: 20px;
  height: 20px;
}

.spinner-small .spinner-inner {
  border-width: 2px;
  border-top-width: 2px;
}

.spinner-medium {
  width: 32px;
  height: 32px;
}

.spinner-large {
  width: 48px;
  height: 48px;
}

.spinner-large .spinner-inner {
  border-width: 4px;
  border-top-width: 4px;
}

.spinner-xlarge {
  width: 64px;
  height: 64px;
}

.spinner-xlarge .spinner-inner {
  border-width: 5px;
  border-top-width: 5px;
}

/* Color variants */
.spinner-primary .spinner-inner {
  --spinner-color: var(--color-primary);
}

.spinner-success .spinner-inner {
  --spinner-color: var(--color-success);
}

.spinner-warning .spinner-inner {
  --spinner-color: var(--color-warning);
}

.spinner-danger .spinner-inner {
  --spinner-color: var(--color-danger);
}

.spinner-secondary .spinner-inner {
  --spinner-color: var(--color-text-secondary);
}

/* Message styling */
.loading-message {
  color: var(--color-text-secondary);
  font-weight: var(--font-medium);
  text-align: center;
  margin: 0;
}

.message-small {
  font-size: var(--text-sm);
}

.message-medium {
  font-size: var(--text-base);
}

.message-large {
  font-size: var(--text-lg);
}

.message-xlarge {
  font-size: var(--text-xl);
}

/* Container size adjustments */
.size-small {
  gap: var(--space-2);
}

.size-medium {
  gap: var(--space-4);
}

.size-large {
  gap: var(--space-6);
}

.size-xlarge {
  gap: var(--space-8);
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

/* Reduced motion support */
@media (prefers-reduced-motion: reduce) {
  .loading-spinner {
    animation: pulse 2s ease-in-out infinite;
  }
  
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }
}
</style>