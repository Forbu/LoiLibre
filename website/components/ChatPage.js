import { useState } from 'react';
import ChatMessages from './ChatMessages';
import ChatInput from './ChatInput';

function ChatPage() {
    const [messages, setMessages] = useState([]);
  
    function handleNewMessage(message) {
      setMessages((prevMessages) => [...prevMessages, message]);
    }
  
    return (
      <div className="container-fluid chat-page">
        <div className="row">
          <div className="col-md-8 offset-md-2 col-lg-6 offset-lg-3 my-5">
            <h1 className="chat-title">Chat Interface</h1>
            <ChatMessages messages={messages} />
            <ChatInput onNewMessage={handleNewMessage} />
          </div>
        </div>
      </div>
    );
  }
  
  export default ChatPage;