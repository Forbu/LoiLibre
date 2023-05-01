function ChatMessages({ messages }) {
    return (
      <ul className="list-group chat-messages">
        {messages.map((message, index) => (
          <li key={index} className="list-group-item mb-4">{message}</li>
        ))}
      </ul>
    );
  }
  
  export default ChatMessages;