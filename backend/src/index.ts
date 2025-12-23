import express from 'express';
import cors from 'cors';
import dotenv from 'dotenv';
import OpenAI from 'openai';
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport } from "@modelcontextprotocol/sdk/client/stdio.js";

dotenv.config();

const app = express();
const port = process.env.PORT || 3001;

app.use(cors());
app.use(express.json());

const openai = new OpenAI({
  baseURL: 'https://openrouter.ai/api/v1',
  apiKey: process.env.OPENROUTER_API_KEY || '',
  defaultHeaders: {
    'HTTP-Referer': 'https://loilibre.fr',
    'X-Title': 'LoiLibre',
  },
});

// MCP Client Setup for Brave Search
const braveTransport = new StdioClientTransport({
  command: "npx",
  args: ["-y", "@modelcontextprotocol/server-brave-search"],
  env: { 
    ...process.env, 
    BRAVE_API_KEY: process.env.BRAVE_API_KEY || "" 
  } as Record<string, string>
});

const mcpClient = new Client(
  { name: "LoiLibre-Backend", version: "1.0.0" },
  { capabilities: {} }
);

async function initMCP() {
  try {
    await mcpClient.connect(braveTransport);
    console.log("Connected to Brave Search MCP");
  } catch (error) {
    console.error("Failed to connect to MCP:", error);
  }
}

initMCP();

app.post('/api/chat', async (req, res) => {
  try {
    const { messages, model } = req.body;

    if (!messages || !Array.isArray(messages)) {
      return res.status(400).json({ error: 'Messages are required' });
    }

    const selectedModel = model || 'openai/gpt-4o-mini';

    // 1. Get tools from MCP
    const { tools: mcpTools } = await mcpClient.listTools();
    
    // 2. Format tools for OpenRouter/OpenAI
    const tools: any[] = mcpTools.map(tool => ({
      type: 'function',
      function: {
        name: tool.name,
        description: tool.description,
        parameters: tool.inputSchema,
      }
    }));

    // 3. First call to LLM
    const response = await openai.chat.completions.create({
      model: selectedModel,
      messages: messages.map(m => ({
        role: m.role,
        content: m.content,
      })),
      tools: tools.length > 0 ? (tools as any) : undefined,
    });

    const choice = response.choices[0];
    if (!choice || !choice.message) {
      throw new Error('No response from OpenRouter');
    }

    let message = choice.message;

    // 4. Handle Tool Calls
    if (message.tool_calls && message.tool_calls.length > 0) {
      const toolMessages = [...messages, message];
      const toolCalsInfo: any[] = [];
      
      for (const toolCall of message.tool_calls) {
        if (toolCall.type !== 'function') continue;
        
        const toolName = toolCall.function.name;
        const toolArgs = JSON.parse(toolCall.function.arguments);

        console.log(`Executing MCP tool: ${toolName}`, toolArgs);

        const result = await mcpClient.callTool({
          name: toolName,
          arguments: toolArgs,
        });

        const toolResult = {
          role: 'tool',
          tool_call_id: toolCall.id,
          content: JSON.stringify(result.content),
        };

        toolMessages.push(toolResult as any);
        
        toolCalsInfo.push({
          tool: toolName,
          args: toolArgs,
          result: result.content
        });
      }

      // 5. Second call to LLM with tool results
      const finalResponse = await openai.chat.completions.create({
        model: selectedModel,
        messages: toolMessages.map(m => {
            const msg: any = { role: m.role, content: m.content || "" };
            if (m.role === 'assistant' && m.tool_calls) msg.tool_calls = m.tool_calls;
            if (m.role === 'tool') msg.tool_call_id = (m as any).tool_call_id;
            
            Object.keys(m).forEach(key => {
                if (!['role', 'content', 'tool_calls', 'tool_call_id'].includes(key)) {
                    msg[key] = (m as any)[key];
                }
            });
            return msg;
        }),
      });

      const finalChoice = finalResponse.choices[0];
      if (!finalChoice || !finalChoice.message) {
        throw new Error('No response from OpenRouter after tool execution');
      }

      // Append tool info to the final message so frontend can display it
      const responseToReturn: any = { ...finalChoice.message };
      responseToReturn.tool_invocations = toolCalsInfo;

      res.json(responseToReturn);
    } else {
      res.json(message);
    }

  } catch (error: any) {
    console.error('Error with OpenRouter/MCP:', error);
    res.status(500).json({ 
      error: 'An error occurred while processing your request',
      details: error.message 
    });
  }
});

app.listen(port, () => {
  console.log(`Backend server running at http://localhost:${port}`);
});
