import readlineSync from 'readline-sync';
import { Pinecone } from '@pinecone-database/pinecone';
import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import { GoogleGenAI } from "@google/genai";
import * as dotenv from 'dotenv';
dotenv.config();

const ai = new GoogleGenAI({});
const History = [];

async function transformQuery(question) {
  if (!question) throw new Error("Question cannot be empty");

  History.push({
    role: 'user',
    parts: [{ text: question }]
  });

  const response = await ai.models.generateContent({
    model: "gemini-2.0-flash",
    contents: History,
    config: {
      systemInstruction: `You are a query rewriting expert. Based on the chat history, rephrase the user question into a standalone one. Only return the rewritten question.`,
    },
  });

  History.pop();
  return response.text;
}

async function chatting(question) {
  const query = await transformQuery(question);

  // Step 1: Generate query vector
  const embeddings = new GoogleGenerativeAIEmbeddings({
    apiKey: process.env.GEMINI_API_KEY,
    model: 'text-embedding-004',
  });

  const queryVector = await embeddings.embedQuery(query);

  // Step 2: Query Pinecone
  const pinecone = new Pinecone();
  const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);

  const searchResults = await pineconeIndex.query({
    topK: 10,// Number of top results to return
    vector: queryVector,// The vector representation of the query
    includeMetadata: true, // Include metadata in the results
  });

  // Step 3: Extract context from metadata (either pageContent or text)
  const context = searchResults.matches
    .map(match =>
      match.metadata?.pageContent || match.metadata?.text || ""// Fallback to empty string if neither exists
    )
    .filter(Boolean)// Filter out any empty strings
    .join("\n\n---\n\n");// Join the context with separators

  // Log for verification
  console.log(`\n📚 Retrieved ${searchResults.matches.length} documents`);


  // Step 4: Ask Gemini with the retrieved context
  History.push({
    role: 'user',
    parts: [{ text: question }]
  });

  const response = await ai.models.generateContent({
    model: "gemini-2.0-flash",
    contents: History,
    config: {
      systemInstruction: `You are a Data Structure and Algorithm Expert.
You will be given a context of relevant information and a user question.
Your task is to answer the user's question based ONLY on the provided context.
If the answer is not in the context, say "I could not find the answer in the provided document."
Keep your answers clear, concise, and educational.

Context: ${context}`,
    },
  });

  History.push({
    role: 'model',
    parts: [{ text: response.text }]
  });

  console.log("\n🧠 Answer:\n" + response.text);
}

async function main() {
  const userProblem = readlineSync.question("Ask me anything--> ");
  
  if (userProblem.toLowerCase() === 'exit' || userProblem.toLowerCase() === 'quit') {
    console.log("👋 Exiting. Have a great day!");
    return;
  }

  await chatting(userProblem);
  main(); // loop again
}


main();
