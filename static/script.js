// Global variables
let isLoading = false;
const MAX_CHARS = 500;

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    checkHealth();
    loadStats();
    autoResizeTextarea();
    
    // Setup char counter
    const input = document.getElementById('queryInput');
    input.addEventListener('input', updateCharCount);
});

// Check API health
async function checkHealth() {
    const statusDot = document.querySelector('.status-dot');
    const statusText = document.querySelector('.status-text');
    
    try {
        const response = await fetch('/health');
        const data = await response.json();
        
        if (data.status === 'healthy') {
            statusDot.classList.add('online');
            statusDot.classList.remove('offline');
            statusText.textContent = 'Online';
        } else {
            statusDot.classList.add('offline');
            statusDot.classList.remove('online');
            statusText.textContent = 'Error: ' + data.error;
        }
    } catch (error) {
        statusDot.classList.add('offline');
        statusDot.classList.remove('online');
        statusText.textContent = 'Offline';
    }
}

// Load system stats
async function loadStats() {
    try {
        const response = await fetch('/stats');
        const data = await response.json();
        
        document.getElementById('embeddingModel').textContent = data.embedding_model || '-';
        document.getElementById('llmModel').textContent = data.llm_model || '-';
        document.getElementById('indexName').textContent = data.index_name || '-';
    } catch (error) {
        console.error('Failed to load stats:', error);
    }
}

// Toggle stats panel
function toggleStats() {
    const statsContent = document.getElementById('statsContent');
    if (statsContent.style.display === 'none') {
        statsContent.style.display = 'block';
    } else {
        statsContent.style.display = 'none';
    }
}

// Update character count
function updateCharCount() {
    const input = document.getElementById('queryInput');
    const charCount = document.getElementById('charCount');
    const currentLength = input.value.length;
    
    charCount.textContent = `${currentLength} / ${MAX_CHARS}`;
    
    if (currentLength > MAX_CHARS) {
        charCount.style.color = 'var(--error)';
        input.value = input.value.substring(0, MAX_CHARS);
    } else {
        charCount.style.color = 'var(--text-secondary)';
    }
}

// Auto-resize textarea
function autoResizeTextarea() {
    const textarea = document.getElementById('queryInput');
    textarea.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = Math.min(this.scrollHeight, 150) + 'px';
    });
}

// Handle keyboard shortcuts
function handleKeyPress(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendQuery();
    }
}

// Send example query
function sendExample(query) {
    document.getElementById('queryInput').value = query;
    sendQuery();
}

// Send query to API
async function sendQuery() {
    if (isLoading) return;
    
    const input = document.getElementById('queryInput');
    const query = input.value.trim();
    
    if (!query) {
        alert('Please enter a query');
        return;
    }
    
    // Clear input
    input.value = '';
    input.style.height = 'auto';
    updateCharCount();
    
    // Hide welcome message if exists
    const welcomeMsg = document.querySelector('.welcome-message');
    if (welcomeMsg) {
        welcomeMsg.remove();
    }
    
    // Add user message
    addMessage('user', query);
    
    // Add loading message
    const loadingId = addLoadingMessage();
    
    // Disable send button
    isLoading = true;
    document.getElementById('sendBtn').disabled = true;
    
    try {
        const response = await fetch('/query', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query, top_k: 5 })
        });
        
        const data = await response.json();
        
        // Remove loading message
        removeLoadingMessage(loadingId);
        
        if (data.success) {
            addMessage('assistant', data.response, data.sources);
        } else {
            addMessage('assistant', 'Sorry, an error occurred.', [], data.error);
        }
    } catch (error) {
        removeLoadingMessage(loadingId);
        addMessage('assistant', 'Failed to connect to the server.', [], error.message);
    } finally {
        isLoading = false;
        document.getElementById('sendBtn').disabled = false;
        input.focus();
    }
}

// Parse response and create property cards with sources
function parseResponse(text, sources) {
    let html = '';
    const lines = text.split('\n');
    
    for (let line of lines) {
        line = line.trim();
        
        // Check if it's a bullet point with [SOURCE:N]
        const bulletMatch = line.match(/^[•\-\*]\s*(.+?)\s*\[SOURCE:(\d+)\]\s*$/);
        
        if (bulletMatch) {
            const content = bulletMatch[1];
            const sourceIndex = parseInt(bulletMatch[2]) - 1;
            const source = sources[sourceIndex];
            
            // Create property card
            html += '<div class="property-card">';
            html += '<div class="property-content">';
            html += content.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
            html += '</div>';
            
            // Add source badge
            if (source) {
                const score = (source.score * 100).toFixed(1);
                const location = source.metadata?.location || 'Unknown';
                const bhk = source.metadata?.bhk_type || '';
                html += `
                    <div class="property-source">
                        <span class="source-badge">${score}% Match</span>
                        <span class="source-location">📍 ${escapeHtml(location)}${bhk ? ' · ' + escapeHtml(bhk) : ''}</span>
                    </div>
                `;
            }
            html += '</div>';
        } else if (line.startsWith('•') || line.startsWith('-') || line.startsWith('*')) {
            // Regular bullet point without source
            html += '<div class="simple-bullet">' + escapeHtml(line.substring(1).trim()) + '</div>';
        } else if (line.length > 0) {
            // Regular text
            html += '<p>' + escapeHtml(line) + '</p>';
        }
    }
    
    return html || escapeHtml(text);
}

// Add message to chat
function addMessage(type, content, sources = [], error = null) {
    const chatContainer = document.getElementById('chatContainer');
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}`;
    
    const icon = type === 'user' ? '👤' : '🤖';
    const label = type === 'user' ? 'You' : 'Assistant';
    
    let html = `
        <div class="message-header">
            <span>${icon}</span>
            <span>${label}</span>
        </div>
        <div class="message-content">
            ${type === 'assistant' ? parseResponse(content, sources) : escapeHtml(content)}
        </div>
    `;
    
    // Add error if present (sources are now inline in property cards)
    if (error) {
        html += `<div class="error-message">⚠️ Error: ${escapeHtml(error)}</div>`;
    }
    
    messageDiv.innerHTML = html;
    chatContainer.appendChild(messageDiv);
    
    // Scroll to bottom
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

// Add loading message
function addLoadingMessage() {
    const chatContainer = document.getElementById('chatContainer');
    const loadingId = 'loading-' + Date.now();
    
    const loadingDiv = document.createElement('div');
    loadingDiv.id = loadingId;
    loadingDiv.className = 'message assistant';
    loadingDiv.innerHTML = `
        <div class="message-header">
            <span>🤖</span>
            <span>Assistant</span>
        </div>
        <div class="message-content">
            <div class="loading">
                <span>Thinking</span>
                <div class="loading-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </div>
        </div>
    `;
    
    chatContainer.appendChild(loadingDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    
    return loadingId;
}

// Remove loading message
function removeLoadingMessage(loadingId) {
    const loadingDiv = document.getElementById(loadingId);
    if (loadingDiv) {
        loadingDiv.remove();
    }
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
