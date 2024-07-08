function sendMessage() {
    var userInput = document.getElementById('user-input').value;
    var sessionId = document.getElementById('session_id').value;
    if (userInput.trim() === '') return;

    // Display user message with label
    var userMessageContainer = document.createElement('div');
    var userMessageLabel = document.createElement('div');
    var userMessage = document.createElement('div');
    userMessage.innerHTML = "<img src='/static/images/ulogo2.png' alt='Logo' class='ulogo'>You<br>" + userInput;
    userMessage.classList.add('message', 'user-message');
    userMessageContainer.appendChild(userMessageLabel);
    userMessageContainer.appendChild(userMessage);
    document.getElementById('chat-box').appendChild(userMessageContainer);
    scrollToBottom(); // Scroll to bottom

    // Display typing indicator
    var typingIndicator = document.createElement('div');
    typingIndicator.classList.add('message', 'bot-message');
    var typingSpan = document.createElement('span');
    typingSpan.classList.add('typing-indicator');
    typingSpan.textContent = 'Typing...';
    typingIndicator.appendChild(typingSpan);
    document.getElementById('chat-box').appendChild(typingIndicator);
    scrollToBottom(); // Scroll to bottom

    // Clear user input
    document.getElementById('user-input').value = '';

    fetch('/chat/'+ sessionId, {
        method: 'POST',
        body: JSON.stringify({ user_input: userInput }),
        headers: {
            'Content-Type': 'application/json',
        },
    })
    .then(response => response.json())
    .then(data => {
        var botResponse = data.response;
        var index = 0;
        var interval = setInterval(function() {
            var partialResponse = botResponse.slice(0, index);
            typingSpan.innerText = partialResponse;
            index++;
            if (index > botResponse.length) {
                clearInterval(interval);
                typingIndicator.remove(); // Remove typing indicator
                // Display chatbot message with label and line break
                var botMessageContainer = document.createElement('div');
                botMessageContainer.classList.add('message', 'bot-message');
                var botMessage = document.createElement('div');
                botMessage.innerHTML = "<img src='/static/images/cuteblue.png' alt='Logo' class='logo'>Chatbot<br>" + botResponse; // Add logo before the message
                botMessage.classList.add('message-text');
                botMessageContainer.appendChild(botMessage);
                document.getElementById('chat-box').appendChild(botMessageContainer);
                scrollToBottom(); // Scroll to bottom
            }
        }, 15); // Adjusted typing speed
    })
    .catch(error => {
        console.error('Error:', error);
    });
}

function handleKeyDown(event) {
    if (event.keyCode === 13) { // Enter key
        event.preventDefault();
        sendMessage();
    }
}

function scrollToBottom() {
    var chatBox = document.getElementById('chat-box');
    chatBox.scrollTop = chatBox.scrollHeight;
}

function deleteChatSession(sessionId) {
    fetch(`/delete_chat_session/${sessionId}`, {
        method: 'DELETE',
    })
    .then(response => {
        if (response.ok) {
            // Remove the session from the sidebar list
            var sessionListItem = document.querySelector(`#active-chat-sessions li[data-session-id="${sessionId}"]`);
            if (sessionListItem) {
                sessionListItem.remove();
            } else {
                console.error('Session ID not found in sidebar');
            }
        } else {
            // Handle error response
            console.error('Failed to delete chat session');
        }
    })
    .catch(error => {
        console.error('Error:', error);
    });
}