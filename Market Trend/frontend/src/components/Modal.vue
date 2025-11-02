<template>
  <Teleport to="body">
    <Transition name="modal" appear>
      <div v-if="modelValue" class="modal-overlay" @click="handleOverlayClick">
        <div 
          class="modal-container" 
          :class="modalClass"
          @click.stop
          role="dialog"
          :aria-labelledby="titleId"
          :aria-describedby="contentId"
          aria-modal="true"
        >
          <!-- Header -->
          <div v-if="showHeader" class="modal-header">
            <h2 v-if="title" :id="titleId" class="modal-title">{{ title }}</h2>
            <slot name="header" v-else></slot>
            <button 
              v-if="closable"
              @click="close"
              class="modal-close"
              aria-label="Close modal"
            >
              ✕
            </button>
          </div>

          <!-- Body -->
          <div :id="contentId" class="modal-body" :class="bodyClass">
            <slot></slot>
          </div>

          <!-- Footer -->
          <div v-if="showFooter" class="modal-footer">
            <slot name="footer">
              <div class="modal-actions">
                <button 
                  v-if="showCancel"
                  @click="cancel"
                  class="btn btn-secondary"
                  :disabled="loading"
                >
                  {{ cancelText }}
                </button>
                <button 
                  v-if="showConfirm"
                  @click="confirm"
                  :class="['btn', confirmVariant]"
                  :disabled="loading || confirmDisabled"
                >
                  <LoadingSpinner 
                    v-if="loading" 
                    size="small" 
                    color="white"
                    :centered="false"
                  />
                  {{ confirmText }}
                </button>
              </div>
            </slot>
          </div>
        </div>
      </div>
    </Transition>
  </Teleport>
</template>

<script>
import LoadingSpinner from './LoadingSpinner.vue';

export default {
  name: 'Modal',
  components: {
    LoadingSpinner
  },
  props: {
    modelValue: {
      type: Boolean,
      required: true
    },
    title: {
      type: String,
      default: ''
    },
    size: {
      type: String,
      default: 'medium',
      validator: value => ['small', 'medium', 'large', 'xlarge', 'fullscreen'].includes(value)
    },
    closable: {
      type: Boolean,
      default: true
    },
    closeOnOverlay: {
      type: Boolean,
      default: true
    },
    showHeader: {
      type: Boolean,
      default: true
    },
    showFooter: {
      type: Boolean,
      default: false
    },
    showCancel: {
      type: Boolean,
      default: true
    },
    showConfirm: {
      type: Boolean,
      default: true
    },
    cancelText: {
      type: String,
      default: 'Cancel'
    },
    confirmText: {
      type: String,
      default: 'Confirm'
    },
    confirmVariant: {
      type: String,
      default: 'btn-primary'
    },
    confirmDisabled: {
      type: Boolean,
      default: false
    },
    loading: {
      type: Boolean,
      default: false
    },
    scrollable: {
      type: Boolean,
      default: true
    },
    centered: {
      type: Boolean,
      default: true
    }
  },
  emits: ['update:modelValue', 'close', 'cancel', 'confirm'],
  computed: {
    modalClass() {
      return {
        [`modal-${this.size}`]: true,
        'modal-centered': this.centered,
        'modal-scrollable': this.scrollable
      };
    },
    bodyClass() {
      return {
        'scrollable': this.scrollable,
        'no-padding': this.$slots.default && this.$slots.default().some(node => 
          node.type?.name === 'form' || node.props?.class?.includes('no-padding')
        )
      };
    },
    titleId() {
      return `modal-title-${this.$.uid}`;
    },
    contentId() {
      return `modal-content-${this.$.uid}`;
    }
  },
  watch: {
    modelValue(newValue) {
      if (newValue) {
        this.handleOpen();
      } else {
        this.handleClose();
      }
    }
  },
  mounted() {
    if (this.modelValue) {
      this.handleOpen();
    }
  },
  beforeUnmount() {
    this.handleClose();
  },
  methods: {
    close() {
      this.$emit('update:modelValue', false);
      this.$emit('close');
    },
    cancel() {
      this.$emit('cancel');
      this.close();
    },
    confirm() {
      this.$emit('confirm');
    },
    handleOverlayClick() {
      if (this.closeOnOverlay && this.closable) {
        this.close();
      }
    },
    handleOpen() {
      document.body.style.overflow = 'hidden';
      document.addEventListener('keydown', this.handleEscape);
    },
    handleClose() {
      document.body.style.overflow = '';
      document.removeEventListener('keydown', this.handleEscape);
    },
    handleEscape(event) {
      if (event.key === 'Escape' && this.closable) {
        this.close();
      }
    }
  }
}
</script>

<style scoped>
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: var(--z-modal);
  padding: var(--space-4);
}

.modal-container {
  background-color: var(--color-surface);
  border-radius: var(--radius-lg);
  box-shadow: var(--shadow-xl);
  display: flex;
  flex-direction: column;
  max-height: 90vh;
  width: 100%;
  position: relative;
}

.modal-container.modal-centered {
  margin: auto;
}

/* Size variants */
.modal-small {
  max-width: 400px;
}

.modal-medium {
  max-width: 600px;
}

.modal-large {
  max-width: 800px;
}

.modal-xlarge {
  max-width: 1200px;
}

.modal-fullscreen {
  max-width: none;
  max-height: none;
  height: 100vh;
  width: 100vw;
  border-radius: 0;
  margin: 0;
}

/* Header */
.modal-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: var(--space-6);
  border-bottom: 1px solid var(--color-border);
  flex-shrink: 0;
}

.modal-title {
  font-size: var(--text-xl);
  font-weight: var(--font-semibold);
  color: var(--color-text-primary);
  margin: 0;
}

.modal-close {
  background: none;
  border: none;
  font-size: var(--text-xl);
  color: var(--color-text-secondary);
  cursor: pointer;
  padding: var(--space-2);
  border-radius: var(--radius-md);
  transition: all var(--transition-fast);
  display: flex;
  align-items: center;
  justify-content: center;
  width: 32px;
  height: 32px;
}

.modal-close:hover {
  background-color: var(--gray-100);
  color: var(--color-text-primary);
}

.dark .modal-close:hover {
  background-color: var(--gray-700);
}

.modal-close:focus {
  outline: 2px solid var(--color-primary);
  outline-offset: 2px;
}

/* Body */
.modal-body {
  padding: var(--space-6);
  flex: 1;
  overflow-y: auto;
}

.modal-body.scrollable {
  overflow-y: auto;
}

.modal-body.no-padding {
  padding: 0;
}

/* Footer */
.modal-footer {
  padding: var(--space-6);
  border-top: 1px solid var(--color-border);
  flex-shrink: 0;
}

.modal-actions {
  display: flex;
  gap: var(--space-3);
  justify-content: flex-end;
}

/* Responsive design */
@media (max-width: 768px) {
  .modal-overlay {
    padding: var(--space-2);
  }
  
  .modal-container {
    max-height: 95vh;
  }
  
  .modal-small,
  .modal-medium,
  .modal-large,
  .modal-xlarge {
    max-width: none;
    width: 100%;
  }
  
  .modal-header,
  .modal-body,
  .modal-footer {
    padding: var(--space-4);
  }
  
  .modal-actions {
    flex-direction: column-reverse;
  }
  
  .modal-actions .btn {
    width: 100%;
  }
}

/* Transitions */
.modal-enter-active,
.modal-leave-active {
  transition: opacity var(--transition-normal);
}

.modal-enter-active .modal-container,
.modal-leave-active .modal-container {
  transition: transform var(--transition-normal);
}

.modal-enter-from,
.modal-leave-to {
  opacity: 0;
}

.modal-enter-from .modal-container,
.modal-leave-to .modal-container {
  transform: scale(0.95) translateY(-20px);
}

/* Loading state */
.modal-container:has(.loading-spinner) .modal-body {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 200px;
}

/* Scrollbar styling */
.modal-body::-webkit-scrollbar {
  width: 6px;
}

.modal-body::-webkit-scrollbar-track {
  background: var(--gray-100);
  border-radius: var(--radius-full);
}

.modal-body::-webkit-scrollbar-thumb {
  background: var(--gray-300);
  border-radius: var(--radius-full);
}

.modal-body::-webkit-scrollbar-thumb:hover {
  background: var(--gray-400);
}

.dark .modal-body::-webkit-scrollbar-track {
  background: var(--gray-700);
}

.dark .modal-body::-webkit-scrollbar-thumb {
  background: var(--gray-600);
}

.dark .modal-body::-webkit-scrollbar-thumb:hover {
  background: var(--gray-500);
}

/* Accessibility improvements */
@media (prefers-reduced-motion: reduce) {
  .modal-enter-active,
  .modal-leave-active,
  .modal-enter-active .modal-container,
  .modal-leave-active .modal-container {
    transition: none;
  }
}

/* Focus trap styling */
.modal-container:focus {
  outline: none;
}
</style>