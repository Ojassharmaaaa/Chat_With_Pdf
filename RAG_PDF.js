import { PDFLoader } from '@langchain/community/document_loaders/fs/pdf';
import { RecursiveCharacterTextSplitter } from '@langchain/textsplitters';
import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import { Pinecone } from '@pinecone-database/pinecone';
import { PineconeStore } from '@langchain/pinecone';
import * as dotenv from 'dotenv';
dotenv.config();


async function IndexDocument() {
    console.log("⚡ Script Started");

    const PDF_PATH = './DSA.pdf';// Path to your PDF file
    if (!PDF_PATH) {
        console.error("Please provide a valid PDF path.");
        return;
    }
    const pdfLoader = new PDFLoader(PDF_PATH);// Load the PDF file using PDFLoader from LangChain
    const rawDocs = await pdfLoader.load();// Load the documents from the PDF
    console.log(`📄 Loaded ${rawDocs.length} documents from the PDF.`);

    //Chunking Phase
    
const textSplitter = new RecursiveCharacterTextSplitter({// Split the text into smaller chunks
    chunkSize: 1000,//Size of each chunk
    chunkOverlap: 200, //Overlap between chunks , So that context is not lost
  });
const chunkedDocs = await textSplitter.splitDocuments(rawDocs);// Split the raw documents into chunks
console.log(`✂️ Split into ${chunkedDocs.length} chunks.`);

//Vector Embedding Model
const embeddings = new GoogleGenerativeAIEmbeddings({
    apiKey: process.env.GEMINI_API_KEY,
    model: 'text-embedding-004',//Model for text embeddings
  });

  //Database Configuration
  //Initialize Pinecone Client

const pinecone = new Pinecone();
const pineconeIndex = pinecone.Index(process.env.PINECONE_INDEX_NAME);
console.log("🌲 Pinecone Configured");

const testEmbedding = await embeddings.embedQuery("test input");
console.log("Test embedding length:", testEmbedding.length);


//LangChain (chunking,embeddings,db) Converting chunks into vector embeddings and storing them in Database
await PineconeStore.fromDocuments(chunkedDocs, embeddings, {
    pineconeIndex,
    maxConcurrency: 5,
  });
console.log("📦 Documents indexed successfully in Pinecone.");
}

IndexDocument();
