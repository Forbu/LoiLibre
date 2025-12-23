import React, { useState, useRef, useEffect } from 'react';
import { Send, Scale, Search, Globe, User, Bot, Settings, ChevronDown, ChevronUp, Terminal } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { cn } from './lib/utils';
import './App.css';

const DEFAULT_MESSAGES = [
  {
    id: '1',
    role: 'assistant',
    content: "Bonjour ! Je suis LoiLibre, votre assistant juridique spécialisé dans le droit français. Comment puis-je vous aider aujourd'hui ?"
  }
];

const MODELS = [
  { id: 'openai/gpt-4o-mini', name: 'GPT-4o Mini' },
  { id: 'openai/gpt-4o', name: 'GPT-4o' },
  { id: 'google/gemini-pro-1.5', name: 'Gemini Pro 1.5' },
  { id: 'anthropic/claude-3.5-sonnet', name: 'Claude 3.5 Sonnet' },
  { id: 'meta-llama/llama-3.1-8b-instruct:free', name: 'Llama 3.1 8B (Free)' },
];

const ToolInvocation = ({ invocation }) => {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <div className="my-2 border border-gray-200 rounded-lg overflow-hidden bg-gray-50">
      <button 
        onClick={() => setIsOpen(!isOpen)}
        className="w-full flex items-center justify-between px-3 py-2 text-xs font-mono text-gray-600 hover:bg-gray-100 transition-colors"
      >
        <div className="flex items-center gap-2">
          <Terminal className="w-3.5 h-3.5" />
          <span>Appel outil: <span className="font-bold text-blue-600">{invocation.tool}</span></span>
        </div>
        {isOpen ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
      </button>
      
      {isOpen && (
        <div className="p-3 bg-gray-900 text-gray-300 font-mono text-[10px] overflow-x-auto max-h-60">
          <div className="mb-2">
            <p className="text-blue-400 mb-1"># Arguments</p>
            <pre>{JSON.stringify(invocation.args, null, 2)}</pre>
          </div>
          <div>
            <p className="text-green-400 mb-1"># Résultat</p>
            <pre className="whitespace-pre-wrap">{JSON.stringify(invocation.result, null, 2)}</pre>
          </div>
        </div>
      )}
    </div>
  );
};

function App() {
  const [messages, setMessages] = useState(DEFAULT_MESSAGES);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [selectedModel, setSelectedModel] = useState(MODELS[0].id);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: input.trim()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch('http://localhost:3001/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: selectedModel,
          messages: [...messages, userMessage].map(m => ({
            role: m.role,
            content: m.content
          })),
        }),
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const data = await response.json();
      
      const assistantMessage = {
        id: Date.now().toString(),
        role: 'assistant',
        content: data.content,
        tool_invocations: data.tool_invocations
      };
      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error:', error);
      const errorMessage = {
        id: Date.now().toString(),
        role: 'assistant',
        content: "Désolé, une erreur est survenue lors de la communication avec le serveur."
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-gray-50 text-gray-900">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 px-6 py-3 flex items-center justify-between shadow-sm">
        <div className="flex items-center gap-2">
          <div className="bg-blue-600 p-2 rounded-lg">
            <Scale className="text-white w-6 h-6" />
          </div>
          <h1 className="text-xl font-bold tracking-tight text-blue-900">LoiLibre</h1>
        </div>

        <div className="flex items-center gap-6">
            <div className="flex items-center gap-2 bg-gray-100 px-3 py-1.5 rounded-xl border border-gray-200">
                <Settings className="w-4 h-4 text-gray-500" />
                <select 
                    value={selectedModel}
                    onChange={(e) => setSelectedModel(e.target.value)}
                    className="bg-transparent text-sm font-medium focus:outline-none cursor-pointer"
                >
                    {MODELS.map(model => (
                        <option key={model.id} value={model.id}>{model.name}</option>
                    ))}
                </select>
            </div>
            <div className="hidden md:flex items-center gap-4 text-sm font-medium text-gray-500">
                <div className="flex items-center gap-1">
                    <Globe className="w-4 h-4" />
                    <span>Droit Français</span>
                </div>
                <div className="flex items-center gap-1">
                    <Search className="w-4 h-4" />
                    <span>Recherche Web</span>
                </div>
            </div>
        </div>
      </header>

      {/* Chat Area */}
      <main className="flex-1 overflow-y-auto p-4 sm:p-6 space-y-6">
        <div className="max-w-3xl mx-auto space-y-6">
          {messages.map((message) => (
            <div
              key={message.id}
              className={cn(
                "flex gap-4 p-4 rounded-2xl shadow-sm border",
                message.role === 'user' 
                  ? "bg-blue-50 border-blue-100 ml-12" 
                  : "bg-white border-gray-100 mr-12"
              )}
            >
              <div className={cn(
                "w-8 h-8 rounded-full flex items-center justify-center shrink-0",
                message.role === 'user' ? "bg-blue-600" : "bg-gray-800"
              )}>
                {message.role === 'user' 
                  ? <User className="w-5 h-5 text-white" /> 
                  : <Bot className="w-5 h-5 text-white" />
                }
              </div>
              <div className="flex-1 overflow-hidden">
                <p className="font-semibold text-xs text-gray-400 uppercase tracking-wider mb-1">
                  {message.role === 'user' ? 'Vous' : 'LoiLibre'}
                </p>
                
                {message.tool_invocations && message.tool_invocations.map((inv, idx) => (
                  <ToolInvocation key={idx} invocation={inv} />
                ))}

                <div className="text-gray-800 leading-relaxed prose prose-sm max-w-none prose-blue">
                   <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {message.content}
                   </ReactMarkdown>
                </div>
              </div>
            </div>
          ))}
          {isLoading && (
            <div className="flex gap-4 p-4 rounded-2xl bg-white border border-gray-100 mr-12 shadow-sm italic text-gray-500 animate-pulse">
               LoiLibre analyse votre demande...
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </main>

      {/* Input Area */}
      <footer className="bg-white border-t border-gray-200 p-4 sm:p-6 text-center">
        <form 
          onSubmit={handleSubmit}
          className="max-w-3xl mx-auto relative group"
        >
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Posez votre question sur le droit français..."
            className="w-full bg-gray-50 border border-gray-200 rounded-2xl py-4 pl-6 pr-14 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all shadow-inner"
          />
          <button
            type="submit"
            disabled={!input.trim() || isLoading}
            className="absolute right-2 top-1/2 -translate-y-1/2 p-2 rounded-xl bg-blue-600 text-white hover:bg-blue-700 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors shadow-lg"
          >
            <Send className="w-5 h-5" />
          </button>
        </form>
        <p className="text-[10px] sm:text-xs mt-3 text-gray-400 uppercase tracking-widest font-medium">
          LoiLibre peut commettre des erreurs. Vérifiez les informations importantes.
        </p>
      </footer>
    </div>
  );
}

export default App;
