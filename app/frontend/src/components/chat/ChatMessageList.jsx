import React from 'react';
import ChatBubble from './ChatBubble'; // Import the ChatMessage component

const ChatMessageList = ({ messages }) => {
  return (
    <div className="chatMessageList">
      {messages.map((message, index) => (
        <ChatBubble
          key={index}
          message={message}
          isUser={message.isUser} // Assume isUser is a property in each message indicating if it's from the user
        />
      ))}
    </div>
  );
}

export default ChatMessageList;