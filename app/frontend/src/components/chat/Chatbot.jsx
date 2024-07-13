import React, { useState} from 'react'
import axios from 'axios';
import './Chatbot.css';
import ChatInputBox from './ChatInputBox';
import ChatMessageList from './ChatMessageList';

const Chatbot = () => {

  const [question, setQuestion] = useState('');
  const [response, setResponse] = useState('');
  const [messages, setMessages] = useState([]);

  const getResponse = async () => {
    try {
      const result = await axios.post("http://localhost:5000/get_response", { question });
      const newMessage = { text: question, isUser: true }; // Assuming the user's message
      const botMessage = { text: result.data.response, isUser: false }; // Assuming the bot's response
      setMessages([...messages, newMessage, botMessage]);
      setQuestion('');
    } catch (error) {
        console.error("Error fetching response.", error);
        setResponse("Error fetching rsponse");
    }
  };

  return (
    <div className="container">
      <div className="chatboxBox">
        <h1>RAG Playground</h1>
        <ChatMessageList messages={messages} /> {/* Render ChatMessageList component */}
        <div className="inputBox">
          {/* Render ChatInputBox component */}
          <ChatInputBox
            question={question}
            setQuestion={setQuestion}
            getResponse={getResponse}
          />
        </div>
      </div>
    </div>
  );
}

export default Chatbot