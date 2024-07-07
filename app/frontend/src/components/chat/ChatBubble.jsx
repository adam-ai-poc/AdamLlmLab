import React from 'react';

const ChatBubble = ({ message, isUser }) => {
  return (
    <div className={`chatMessage ${isUser ? 'userMessage' : 'botMessage'}`}>
        <p>{`${isUser ? 'You' : 'Bot'}: `}</p>
        <p>{message.text}</p>
        <span className="timestamp">{message.timestamp}</span>
    </div>
  );
}

export default ChatBubble;