const BACKEND_URL = ''; // Flask is running on the same domain/port

// Utility function to display the chosen file name
function displayFileName() {
    const fileInput = document.getElementById('pdfFile');
    const fileNameDisplay = document.getElementById('fileNameDisplay');
    if (fileInput.files.length > 0) {
        fileNameDisplay.textContent = fileInput.files[0].name;
    } else {
        fileNameDisplay.textContent = "Click to Select PDF (Only .pdf)";
    }
}

// --- PDF Upload Logic ---
async function uploadPdf() {
    const fileInput = document.getElementById('pdfFile');
    const pdfStatus = document.getElementById('pdfStatus');
    const file = fileInput.files[0];

    if (!file) {
        pdfStatus.innerHTML = "⚠️ Please select a **PDF** file first.";
        pdfStatus.className = 'status-message error';
        return;
    }

    pdfStatus.innerHTML = `⏳ Uploading and processing "${file.name}"... **Do not refresh.**`;
    pdfStatus.className = 'status-message processing';

    const formData = new FormData();
    formData.append('pdf_file', file);

    try {
        const response = await fetch(`${BACKEND_URL}/upload_pdf`, {
            method: 'POST',
            body: formData,
        });

        const result = await response.json();

        if (response.ok) {
            pdfStatus.innerHTML = `✅ **Success:** ${result.message}`;
            pdfStatus.className = 'status-message success';
        } else {
            pdfStatus.innerHTML = `❌ **Error:** ${result.message}`;
            pdfStatus.className = 'status-message error';
        }
    } catch (error) {
        pdfStatus.innerHTML = `❌ **Network Error:** Could not connect to the server.`;
        pdfStatus.className = 'status-message error';
        console.error('Upload Error:', error);
    }
}


// --- Query Submission Logic (Chat Interface) ---
async function sendQuery() {
    const queryInput = document.getElementById('queryInput');
    const chatHistory = document.getElementById('chatHistory');
    const query = queryInput.value.trim();

    if (!query) return;

    // 1. Add User Query to History
    appendMessage(query, 'user');
    queryInput.value = ''; // Clear input field

    // 2. Add a loading message (System is working)
    const loadingId = appendMessage('<span class="loading-dots">Controller is deciding and running agent...</span>', 'system processing-status');

    try {
        const response = await fetch(`${BACKEND_URL}/ask`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ query: query }),
        });

        const result = await response.json();

        // 3. Update the loading message with the final response
        let answerContent;
        let agentInfo;
        
        if (response.ok) {
            const agent = result.agents_used.toUpperCase();
            agentInfo = `Agent Used: **${agent}** | Rationale: ${result.decision_rationale}`;
            answerContent = `
                <div class="agent-info-box">${agentInfo}</div>
                ${formatAnswer(result.answer)}
            `;
            updateMessage(loadingId, answerContent, 'system');
        } else {
            // Handle server-side errors (500)
            agentInfo = `Agent Used: **FAILURE**`;
            answerContent = `
                <div class="agent-info-box error-box">${agentInfo}</div>
                <p>❌ **System Error:** ${result.answer}</p>
            `;
            updateMessage(loadingId, answerContent, 'system error');
        }
    } catch (error) {
        // Handle network/fetch errors
        const errorContent = `
            <div class="agent-info-box error-box">Agent Used: **NETWORK FAILURE**</div>
            <p>❌ **Network Error:** Could not connect to the server. Check console.</p>
        `;
        updateMessage(loadingId, errorContent, 'system error');
        console.error('Query Error:', error);
    }
}

// Helper to append a message to the chat history
function appendMessage(content, type) {
    const chatHistory = document.getElementById('chatHistory');
    const messageDiv = document.createElement('div');
    const messageId = `msg-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    
    messageDiv.className = `message ${type}`;
    messageDiv.id = messageId;
    messageDiv.innerHTML = content;
    
    chatHistory.appendChild(messageDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight; // Auto-scroll
    return messageId;
}

// Helper to update a message (used for loading status)
function updateMessage(id, content, type) {
    const messageDiv = document.getElementById(id);
    if (messageDiv) {
        messageDiv.className = `message ${type}`; // Update class to final state
        messageDiv.innerHTML = content;
        document.getElementById('chatHistory').scrollTop = document.getElementById('chatHistory').scrollHeight; // Auto-scroll
    }
}

// Simple Markdown-like formatting for output
function formatAnswer(text) {
    // Basic formatting for better readability (handling **bold** and new lines)
    let formattedText = text.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    formattedText = formattedText.replace(/### (.*)/g, '<h3>$1</h3>');
    formattedText = formattedText.replace(/## (.*)/g, '<h2>$1</h2>');
    formattedText = formattedText.replace(/---\n/g, '<hr>'); // Replace separator lines
    formattedText = formattedText.replace(/\n/g, '<br>'); // Convert newlines to breaks
    return formattedText;
}