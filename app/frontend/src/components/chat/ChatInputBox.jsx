import React, { useState} from 'react'
import './Chatbot.css';

const ChatInputBox = ({ question, setQuestion, getResponse }) => {

    const handleKeyPress = (e) => {
        if (e.key === 'Enter') {
          getResponse();
        }
      };

    return (
      <div className="chatInputBox">
        <input 
          className="inputBox"
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyPress={handleKeyPress} 
          placeholder="Ask a question"
        />
        <button onClick={getResponse}>Send</button>
      </div>
    );
  }

  export default ChatInputBox;