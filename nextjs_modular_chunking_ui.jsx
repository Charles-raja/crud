/*
This is a modular and scalable Next.js 14+ UI for your FastAPI backend.
It allows the user to upload PDF/CSV/etc. files, choose a chunking/embedding strategy,
and then send it to the backend for processing. We use TailwindCSS with orange + light grey theme.
*/

// app/page.tsx
"use client";
import { useState } from "react";
import axios from "axios";

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [strategy, setStrategy] = useState("fixed");
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState<any>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
    }
  };

  const handleSubmit = async () => {
    if (!file) return;
    setLoading(true);
    const formData = new FormData();
    formData.append("file", file);
    formData.append("strategy", strategy);
    try {
      const res = await axios.post("http://localhost:8000/upload", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setResponse(res.data);
    } catch (err) {
      console.error(err);
    }
    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col items-center justify-center p-6">
      <div className="max-w-xl w-full bg-white rounded-2xl shadow-lg p-8">
        <h1 className="text-3xl font-bold text-orange-600 mb-6 text-center">Document Embedding Uploader</h1>
        <input type="file" onChange={handleFileChange} className="mb-4 w-full text-gray-700" />
        <label className="block mb-2 text-gray-600 font-medium">Choose Chunking Strategy</label>
        <select
          value={strategy}
          onChange={(e) => setStrategy(e.target.value)}
          className="w-full border border-gray-300 rounded-lg p-2 mb-4"
        >
          <option value="fixed">Fixed-size Chunking</option>
          <option value="recursive">Recursive Chunking</option>
          <option value="semantic">Semantic Chunking</option>
          <option value="summarize">Summarization-based Embedding</option>
          <option value="csv_row">CSV Row-based Embedding</option>
          <option value="csv_column">CSV Column-based Embedding</option>
          <option value="multimodal">Multi-modal Parent-Child Linking</option>
        </select>
        <button
          onClick={handleSubmit}
          className="w-full bg-orange-500 hover:bg-orange-600 text-white font-semibold py-2 px-4 rounded-lg transition duration-300"
        >
          {loading ? "Uploading..." : "Upload & Process"}
        </button>
        {response && (
          <div className="mt-6 bg-gray-100 p-4 rounded-lg">
            <h2 className="text-xl font-semibold text-gray-800">Server Response</h2>
            <pre className="text-sm text-gray-700 mt-2 overflow-x-auto">
              {JSON.stringify(response, null, 2)}
            </pre>
          </div>
        )}
      </div>
    </div>
  );
}

// styles/globals.css (add Tailwind config accordingly)
/* Tailwind base imports here */
@tailwind base;
@tailwind components;
@tailwind utilities;

:root {
  --color-primary: #f97316; /* orange-500 */
  --color-secondary: #f3f4f6; /* gray-50 */
}

body {
  background-color: var(--color-secondary);
}

// tailwind.config.js
module.exports = {
  content: ["./app/**/*.{js,ts,jsx,tsx}", "./components/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        orange: {
          500: "#f97316",
          600: "#ea580c",
        },
        gray: {
          50: "#f9fafb",
          100: "#f3f4f6",
        },
      },
    },
  },
  plugins: [],
};

// package.json dependencies:
// "next": "14.x",
// "react": "18.x",
// "react-dom": "18.x",
// "axios": "^1.6.0",
// "tailwindcss": "^3.3.0",

/*
How it works:
- User uploads PDF/CSV/etc.
- Selects chunking strategy.
- Hits backend FastAPI at /upload.
- Backend (previous file) chooses correct chunker and embeds.
- Response displayed.

Modular & Scalable:
- Strategies are listed in dropdown but you can dynamically fetch from backend if you add new ones.
- You can break out UI into components inside /components (UploadForm.tsx, StrategySelector.tsx) for more modularity.
*/
