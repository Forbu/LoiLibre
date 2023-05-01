import { useState } from 'react';

function ChatInput({ onNewMessage }) {
    const [inputValue, setInputValue] = useState('');
  
    function handleSubmit(event) {
      event.preventDefault();
      onNewMessage(inputValue);
      setInputValue('');
    }
  
    function handleChange(event) {
      setInputValue(event.target.value);
    }
  
    return (
      <form className="chat-input" onSubmit={handleSubmit}>
        <div className="input-group mb-3">
          <input type="text" className="form-control" placeholder="Type your message here" value={inputValue} onChange={handleChange} />
          <div className="input-group-append">
            <button className="btn btn-primary" type="submit">Send</button>
          </div>
        </div>
        <div className="d-flex justify-content-start align-items-center">
          <span className="text-muted small mr-2">Press Enter to send</span>
        </div>
      </form>
    );
  }
  
  export default ChatInput;