"use client";

import { useEffect, useRef, useState } from "react";
import Image from "next/image";
import { DM_Sans, Sora } from "next/font/google";
import ReactMarkdown from "react-markdown";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";

const dmSans = DM_Sans({
  subsets: ["latin"],
  weight: ["400", "500", "700"],
});

const sora = Sora({
  subsets: ["latin"],
  weight: ["600", "700"],
});

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type Citation = { index: number; chunk: string; score: number };

type Tab = "chat" | "flashcards" | "quiz" | "formulas";

// Shapes match the JSON the backend prompts ask the model to return.
type Flashcard = {
  id: number;
  question: string;
  answer: string;
  type?: string;
  difficulty?: string;
};

type QuizQuestion = {
  id: number;
  type: "mcq" | "truefalse" | "shortanswer";
  question: string;
  options?: string[];
  correct?: string | boolean;
  model_answer?: string;
  explanation?: string;
};

type Formula = {
  id: number;
  formula: string;
  latex: string;
  name: string;
  variables?: Record<string, string>;
  domain?: string;
};

type Message = {
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  isStreaming?: boolean;
};

export default function Page() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loadingSummary, setLoadingSummary] = useState(false);
  const [chatLoading, setChatLoading] = useState(false);
  const [ollamaStatus, setOllamaStatus] = useState<
    "connected" | "offline" | "checking"
  >("checking");

  const [activeTab, setActiveTab] = useState<Tab>("chat");
  const [flashcards, setFlashcards] = useState<Flashcard[]>([]);
  const [currentCard, setCurrentCard] = useState(0);
  const [cardFlipped, setCardFlipped] = useState(false);
  const [quiz, setQuiz] = useState<QuizQuestion[]>([]);
  const [quizAnswers, setQuizAnswers] = useState<Record<number, string>>({});
  const [quizSubmitted, setQuizSubmitted] = useState(false);
  const [quizDifficulty, setQuizDifficulty] = useState("medium");
  const [formulas, setFormulas] = useState<Formula[]>([]);
  const [featureLoading, setFeatureLoading] = useState(false);

  const [sessionStatus, setSessionStatus] = useState<"idle" | "embedding" | "ready" | "error">("idle");
  const [embeddingProgress, setEmbeddingProgress] = useState(0);

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const endRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, chatLoading]);

  useEffect(() => {
    const checkHealth = async () => {
      try {
        const response = await fetch(`${API_URL}/health`);
        const data = await response.json();
        setOllamaStatus(data.ollama === "connected" ? "connected" : "offline");
      } catch {
        setOllamaStatus("offline");
      }
    };

    checkHealth();
  }, []);

  const parseError = async (
    response: Response,
    fallback: string
  ): Promise<string> => {
    try {
      const data = await response.json();
      if (data && typeof data.detail === "string") {
        return data.detail;
      }
      if (data && typeof data.error === "string") {
        return data.error;
      }
    } catch {
      return fallback;
    }

    return fallback;
  };

  const pollStatus = async (sid: string) => {
    try {
      const response = await fetch(`${API_URL}/status/${sid}`);
      if (response.ok) {
        const data = await response.json();
        setSessionStatus(data.status);
        setEmbeddingProgress(data.progress || 0);

        if (data.status === "embedding") {
          setTimeout(() => pollStatus(sid), 1000);
        } else if (data.status === "error") {
          alert(`Indexing failed: ${data.error_detail}`);
        }
      } else {
         setTimeout(() => pollStatus(sid), 1000); // retry
      }
    } catch {
       setTimeout(() => pollStatus(sid), 1000);
    }
  };

  const handleFileUpload = async (file: File) => {
    setUploading(true);
    setMessages([]);
    setInput("");
    setSessionStatus("idle");
    setEmbeddingProgress(0);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API_URL}/upload`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const message = await parseError(response, "Upload failed");
        throw new Error(message);
      }

      const data = await response.json();
      setSessionId(data.session_id);
      setFileName(file.name);
      
      setSessionStatus("embedding");
      pollStatus(data.session_id);
    } catch (error) {
      alert(error instanceof Error ? error.message : "Upload failed");
    } finally {
      setUploading(false);
    }
  };

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      handleFileUpload(file);
    }
  };

  const handleDrop = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    if (uploading) {
      return;
    }

    const file = event.dataTransfer.files?.[0];
    if (file) {
      handleFileUpload(file);
    }
  };

  const handleDragOver = (event: React.DragEvent<HTMLDivElement>) => {
    event.preventDefault();
  };

  const handleSummary = async () => {
    if (!sessionId) {
      return;
    }

    setLoadingSummary(true);
    try {
      const response = await fetch(`${API_URL}/summarize`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId }),
      });

      if (!response.ok) {
        const message = await parseError(response, "Summary failed");
        throw new Error(message);
      }

      const data = await response.json();
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: data.summary },
      ]);
    } catch (error) {
      alert(error instanceof Error ? error.message : "Summary failed");
    } finally {
      setLoadingSummary(false);
    }
  };

  const sendMessage = async () => {
    if (!input.trim() || !sessionId || chatLoading) return;
    const userMessage = input.trim();
    setInput("");
    setMessages((prev) => [...prev, { role: "user", content: userMessage }]);
    setMessages((prev) => [
      ...prev,
      { role: "assistant", content: "", isStreaming: true },
    ]);
    setChatLoading(true);

    const updateLast = (patch: (msg: Message) => Partial<Message>) =>
      setMessages((prev) => {
        const updated = [...prev];
        const last = updated[updated.length - 1];
        updated[updated.length - 1] = { ...last, ...patch(last) };
        return updated;
      });

    try {
      const response = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          message: userMessage,
          history: messages,
        }),
      });

      if (!response.ok || !response.body) {
        throw new Error(await parseError(response, "Chat request failed"));
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let citations: Citation[] = [];
      let streamError: string | null = null;

      while (!streamError) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          let data;
          try {
            data = JSON.parse(line.slice(6));
          } catch {
            continue; // ignore a malformed event rather than abort the reply
          }
          if (data.error) {
            streamError = data.error;
            break;
          }
          if (data.token) {
            updateLast((last) => ({ content: last.content + data.token }));
          }
          if (data.done) citations = data.citations || [];
        }
      }

      if (streamError) {
        await reader.cancel();
        throw new Error(streamError);
      }

      updateLast(() => ({ isStreaming: false, citations }));
    } catch (e) {
      const detail =
        e instanceof TypeError
          ? "Could not reach the backend."
          : e instanceof Error
            ? e.message
            : "Something went wrong.";
      updateLast(() => ({ content: `Error: ${detail}`, isStreaming: false }));
    }
    setChatLoading(false);
  };

  const generateFlashcards = async () => {
    if (!sessionId) return;
    setFeatureLoading(true);
    try {
      const res = await fetch(`${API_URL}/flashcards`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId }),
      });
      if (!res.ok) throw new Error(await parseError(res, "Flashcard generation failed"));
      const data = await res.json();
      setFlashcards(data.flashcards || []);
      setCurrentCard(0);
      setCardFlipped(false);
    } catch (error) {
      alert(error instanceof Error ? error.message : "Flashcard generation failed");
    }
    setFeatureLoading(false);
  };

  const generateQuiz = async () => {
    if (!sessionId) return;
    setFeatureLoading(true);
    try {
      const res = await fetch(`${API_URL}/quiz`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, difficulty: quizDifficulty }),
      });
      if (!res.ok) throw new Error(await parseError(res, "Quiz generation failed"));
      const data = await res.json();
      setQuiz(data.questions || []);
      setQuizAnswers({});
      setQuizSubmitted(false);
    } catch (error) {
      alert(error instanceof Error ? error.message : "Quiz generation failed");
    }
    setFeatureLoading(false);
  };

  const generateFormulas = async () => {
    if (!sessionId) return;
    setFeatureLoading(true);
    try {
      const res = await fetch(`${API_URL}/formulas`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId }),
      });
      if (!res.ok) throw new Error(await parseError(res, "Formula extraction failed"));
      const data = await res.json();
      setFormulas(data.formulas || []);
    } catch (error) {
      alert(error instanceof Error ? error.message : "Formula extraction failed");
    }
    setFeatureLoading(false);
  };

  const exportFlashcardsCSV = () => {
    const headers = ["ID", "Question", "Answer", "Type", "Difficulty"];
    const rows = flashcards.map(card => [
      card.id,
      `"${card.question.replace(/"/g, '""')}"`,
      `"${card.answer.replace(/"/g, '""')}"`,
      card.type,
      card.difficulty
    ]);
    const csv = [headers, ...rows].map(r => r.join(",")).join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "flashcards.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleKeyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (event.key === "Enter") {
      event.preventDefault();
      sendMessage();
    }
  };

  const canSend = Boolean(
    sessionId && sessionStatus === "ready" && !chatLoading && input.trim().length > 0
  );

  return (
    <div className={`${dmSans.className} min-h-screen bg-slate-100 text-slate-900`}>
      <div className="min-h-screen md:grid md:grid-cols-[320px_1fr]">
        <aside className="relative flex flex-col gap-6 bg-[#0f172a] px-6 py-8 text-slate-200 sticky top-0 h-screen overflow-y-auto">
          <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_top,_#1e293b_0%,_transparent_60%)] opacity-70" />
          <div className="relative flex flex-col gap-6">
            <div>
              <h1 className={`${sora.className} text-2xl text-white`}>
                Study Assistant
              </h1>
              <p className="mt-1 text-sm text-slate-400">
                Upload a document to unlock chat and summaries.
              </p>
            </div>

            <div
              role="button"
              tabIndex={0}
              onClick={() => fileInputRef.current?.click()}
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              className="group flex min-h-[170px] cursor-pointer flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed border-slate-600/70 bg-slate-900/30 p-6 text-center transition-all duration-200 hover:border-indigo-400 hover:bg-slate-900/60"
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf,.docx,.txt"
                onChange={handleFileChange}
                className="hidden"
              />

              {uploading ? (
                <div className="flex flex-col items-center gap-3 text-indigo-200">
                  <div className="h-8 w-8 animate-spin rounded-full border-2 border-indigo-300 border-t-transparent" />
                  <span className="text-sm">Uploading...</span>
                </div>
              ) : sessionStatus === "embedding" ? (
                <div className="flex flex-col items-center gap-3 text-indigo-200 w-full max-w-[200px]">
                  <div className="h-8 w-8 animate-spin rounded-full border-2 border-indigo-300 border-t-transparent" />
                  <span className="text-sm">Indexing your document... {embeddingProgress}%</span>
                  <div className="h-1.5 w-full bg-slate-700 rounded-full overflow-hidden mt-2">
                    <div 
                      className="h-full bg-indigo-400 transition-all duration-300"
                      style={{ width: `${embeddingProgress}%` }}
                    />
                  </div>
                </div>
              ) : fileName ? (
                <div className="flex flex-col items-center gap-3 text-emerald-300">
                  <div className="flex h-10 w-10 items-center justify-center rounded-full bg-emerald-500/20">
                    <svg
                      width="22"
                      height="22"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="2"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    >
                      <polyline points="20 6 9 17 4 12" />
                    </svg>
                  </div>
                  <span className="text-sm text-slate-200">{fileName}</span>
                </div>
              ) : (
                <div className="flex flex-col items-center gap-3 text-slate-300">
                  <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-indigo-500/10 text-indigo-300 transition-all duration-200 group-hover:bg-indigo-500/20">
                    <svg
                      width="24"
                      height="24"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="1.8"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    >
                      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                      <polyline points="17 8 12 3 7 8" />
                      <line x1="12" y1="3" x2="12" y2="15" />
                    </svg>
                  </div>
                  <span className="text-sm">
                    Drop your PDF, DOCX or TXT here
                  </span>
                </div>
              )}
            </div>

            <div className="flex flex-col gap-4 rounded-2xl border border-slate-700/70 bg-slate-900/40 p-4 max-h-96 overflow-y-auto">
              <div className="flex items-center justify-between">
                <h2
                  className={`${sora.className} text-sm uppercase tracking-[0.2em] text-slate-400`}
                >
                  Summary
                </h2>
              </div>

              <button
                type="button"
                onClick={handleSummary}
                disabled={!sessionId || loadingSummary}
                className="rounded-xl bg-indigo-500 px-4 py-2 text-sm font-medium text-white shadow-lg shadow-indigo-500/30 transition-all duration-200 hover:-translate-y-0.5 hover:bg-indigo-400 disabled:cursor-not-allowed disabled:opacity-50"
              >
                {loadingSummary ? "Generating..." : "Generate Summary"}
              </button>

              <div className="rounded-xl border border-slate-700/70 bg-slate-950/40 p-3 text-xs text-slate-300">
                Summary will be posted to the chat as a new assistant message.
              </div>
            </div>
          </div>
        </aside>

        <main className="relative flex min-h-screen flex-col bg-white">
          <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_top,_#eef2ff_0%,_#ffffff_45%,_#f8fafc_100%)]" />
          <div className="relative flex min-h-screen flex-col">
            <header className="px-6 pt-6">
              <div className="flex flex-wrap items-center justify-end gap-4">
                <span className="text-xs text-slate-500">
                  {ollamaStatus === "connected"
                    ? "🟢 Ollama Connected"
                    : ollamaStatus === "offline"
                    ? "🔴 Ollama Offline"
                    : "Checking Ollama..."}
                </span>
              </div>
              <div className="mt-4">
                <h2 className={`${sora.className} text-3xl text-slate-900`}>
                  Welcome to NeroEd
                </h2>
                <p className="mt-2 text-sm text-slate-500">
                  Ask questions grounded in the uploaded material.
                </p>
              </div>
            </header>

            <div className="flex gap-1 border-b border-gray-200 mb-4 overflow-x-auto px-6 mt-4">
              {[
                { key: "chat", label: "💬 Chat" },
                { key: "flashcards", label: "🃏 Flashcards" },
                { key: "quiz", label: "📝 Quiz" },
                { key: "formulas", label: "🔢 Formulas" }
              ].map(tab => {
                const isReady = sessionStatus === "ready";
                return (
                <button
                  key={tab.key}
                  disabled={!isReady}
                  onClick={() => setActiveTab(tab.key as Tab)}
                  className={`px-4 py-2 text-sm font-medium whitespace-nowrap border-b-2 transition-colors
                    ${activeTab === tab.key 
                      ? "border-indigo-500 text-indigo-600" 
                      : "border-transparent text-gray-500 hover:text-gray-700"}
                    ${!isReady ? "opacity-50 cursor-not-allowed" : ""}`}
                >
                  {tab.label}
                </button>
              )})}
            </div>

            <div className="relative flex-1 overflow-y-auto px-6 pb-24 pt-6">
              <div className="absolute inset-0 flex items-center justify-center pointer-events-none overflow-hidden">
                <div className="relative h-full w-[80%] max-w-3xl">
                  <Image
                    src="/NeroEd.webp"
                    alt="NeroEd Background Logo"
                    fill
                    priority
                    sizes="(max-width: 960px) 80vw, 768px"
                    className="object-contain select-none"
                    style={{ opacity: 0.26 }}
                  />
                </div>
              </div>

              {activeTab === "chat" && (
                <>
                  {messages.length === 0 && !chatLoading ? null : (
                    <div className="flex flex-col gap-4">
                      {messages.map((message, index) =>
                    message.role === "user" ? (
                      <div key={index} className="flex justify-end">
                        <div className="max-w-[80%] rounded-2xl rounded-br-sm bg-slate-900 px-4 py-3 text-sm text-white shadow-lg shadow-slate-900/10">
                          {message.content}
                        </div>
                      </div>
                    ) : message.isStreaming && !message.content ? null : (
                      <div key={index} className="flex items-start gap-3">
                        <div className="flex h-8 w-8 items-center justify-center rounded-full bg-indigo-100 text-indigo-600">
                          <svg
                            width="16"
                            height="16"
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="1.8"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                          >
                            <rect x="3" y="7" width="18" height="11" rx="2" />
                            <path d="M7 7V5a2 2 0 0 1 2-2h6a2 2 0 0 1 2 2v2" />
                            <path d="M8 16v2" />
                            <path d="M16 16v2" />
                          </svg>
                        </div>
                        <div className="max-w-[80%] space-y-3">
                          <div className="rounded-2xl rounded-tl-sm border border-slate-200 bg-white px-4 py-3 text-sm text-slate-800 shadow-sm prose prose-sm max-w-none">
                            <ReactMarkdown
                              components={{
                                strong: ({ node, ...props }) => <strong className="font-bold text-gray-900" {...props} />,
                                ul: ({ node, ...props }) => <ul className="list-disc ml-4 space-y-1" {...props} />,
                                ol: ({ node, ...props }) => <ol className="list-decimal ml-4 space-y-1" {...props} />,
                                h3: ({ node, ...props }) => <h3 className="font-bold text-base mt-3 mb-1" {...props} />,
                                p: ({ node, ...props }) => <p className="mb-2" {...props} />,
                                hr: ({ node, ...props }) => <hr className="my-3 border-gray-200" {...props} />,
                              }}
                            >
                              {message.content}
                            </ReactMarkdown>
                          </div>
                          {message.citations && message.citations.length > 0 && (
                            <div className="space-y-2">
                              <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                                Sources
                              </p>
                              {message.citations.map((citation) => (
                                <details
                                  key={citation.index}
                                  className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2 text-xs text-slate-600"
                                >
                                  <summary className="cursor-pointer text-slate-500">
                                    [{citation.index}] {citation.chunk.slice(0, 120)}
                                    {citation.chunk.length > 120 ? "..." : ""} (score:{" "}
                                    {citation.score.toFixed(2)})
                                  </summary>
                                  <p className="mt-2 whitespace-pre-wrap text-slate-600">
                                    {citation.chunk}
                                  </p>
                                </details>
                              ))}
                            </div>
                          )}
                        </div>
                      </div>
                    )
                  )}

                  {chatLoading && !messages[messages.length - 1]?.content && (
                    <div className="flex items-start gap-3">
                      <div className="flex h-8 w-8 items-center justify-center rounded-full bg-indigo-100 text-indigo-600">
                        <svg
                          width="16"
                          height="16"
                          viewBox="0 0 24 24"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth="1.8"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        >
                          <rect x="3" y="7" width="18" height="11" rx="2" />
                          <path d="M7 7V5a2 2 0 0 1 2-2h6a2 2 0 0 1 2 2v2" />
                          <path d="M8 16v2" />
                          <path d="M16 16v2" />
                        </svg>
                      </div>
                      <div className="rounded-2xl rounded-tl-sm border border-slate-200 bg-white px-4 py-3 text-sm text-slate-800 shadow-sm">
                        <div className="flex items-center gap-1">
                          <span className="h-2 w-2 animate-bounce rounded-full bg-indigo-400" />
                          <span
                            className="h-2 w-2 animate-bounce rounded-full bg-indigo-400"
                            style={{ animationDelay: "0.1s" }}
                          />
                          <span
                            className="h-2 w-2 animate-bounce rounded-full bg-indigo-400"
                            style={{ animationDelay: "0.2s" }}
                          />
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              )}
              <div ref={endRef} />
              </>
              )}

              {activeTab === "flashcards" && (
                <div className="flex flex-col gap-6">
                  <div className="flex items-center justify-between">
                    <button
                      onClick={generateFlashcards}
                      disabled={!sessionId || featureLoading}
                      className="rounded-xl bg-indigo-500 px-4 py-2 text-sm font-medium text-white shadow-lg transition-all hover:bg-indigo-400 disabled:opacity-50"
                    >
                      {featureLoading ? "Generating..." : "Generate Flashcards"}
                    </button>
                    {flashcards.length > 0 && (
                      <button
                        onClick={exportFlashcardsCSV}
                        className="rounded-xl border border-slate-300 px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50"
                      >
                        Export CSV
                      </button>
                    )}
                  </div>
                  
                  {featureLoading && <div className="text-center py-10 text-slate-500">Loading flashcards...</div>}

                  {!featureLoading && flashcards.length > 0 && (
                    <div className="flex flex-col items-center gap-6">
                      <p className="text-sm font-semibold text-slate-500">
                        {currentCard + 1} / {flashcards.length}
                      </p>
                      <div
                        className="relative w-full max-w-lg cursor-pointer"
                        style={{ perspective: "1000px", minHeight: "280px" }}
                        onClick={() => setCardFlipped(!cardFlipped)}
                      >
                        <div
                          className="relative w-full h-full transition-transform duration-500"
                          style={{
                            transformStyle: "preserve-3d",
                            transform: cardFlipped ? "rotateY(180deg)" : "rotateY(0deg)",
                            minHeight: "280px"
                          }}
                        >
                          {/* FRONT - Question */}
                          <div
                            className="absolute inset-0 bg-white rounded-2xl shadow-md p-8 flex flex-col justify-between"
                            style={{ backfaceVisibility: "hidden", WebkitBackfaceVisibility: "hidden" }}
                          >
                            <div className="flex justify-between items-start">
                              <span className="text-xs font-semibold uppercase tracking-widest text-gray-400">
                                {flashcards[currentCard]?.type || "card"}
                              </span>
                              <span className="text-xs font-bold px-2 py-1 rounded-full bg-indigo-50 text-indigo-500 uppercase">
                                {flashcards[currentCard]?.difficulty || "medium"}
                              </span>
                            </div>
                            <p className="text-xl font-semibold text-gray-800 text-center">
                              {flashcards[currentCard]?.question}
                            </p>
                            <p className="text-xs text-gray-400 text-center">Click to flip</p>
                          </div>

                          {/* BACK - Answer */}
                          <div
                            className="absolute inset-0 bg-indigo-600 rounded-2xl shadow-md p-8 flex flex-col justify-between"
                            style={{
                              backfaceVisibility: "hidden",
                              WebkitBackfaceVisibility: "hidden",
                              transform: "rotateY(180deg)"
                            }}
                          >
                            <div className="flex justify-between items-start">
                              <span className="text-xs font-semibold uppercase tracking-widest text-indigo-200">
                                Answer
                              </span>
                            </div>
                            <p className="text-lg font-medium text-white text-center flex-1 flex items-center justify-center">
                              {flashcards[currentCard]?.answer}
                            </p>
                            <p className="text-xs text-indigo-300 text-center">Click to flip back</p>
                          </div>
                        </div>
                      </div>
                      
                      <div className="flex gap-4">
                        <button 
                          onClick={() => { setCurrentCard(i => Math.max(0, i - 1)); setCardFlipped(false); }}
                          disabled={currentCard === 0}
                          className="px-4 py-2 bg-slate-200 rounded-xl disabled:opacity-50"
                        >
                          Previous
                        </button>
                        <button 
                          onClick={() => { setCurrentCard(i => Math.min(flashcards.length - 1, i + 1)); setCardFlipped(false); }}
                          disabled={currentCard === flashcards.length - 1}
                          className="px-4 py-2 bg-slate-200 rounded-xl disabled:opacity-50"
                        >
                          Next
                        </button>
                      </div>
                    </div>
                  )}
                </div>
              )}

              {activeTab === "quiz" && (
                <div className="flex flex-col gap-6 max-w-3xl mx-auto w-full">
                  <div className="flex items-center gap-4 bg-white p-4 rounded-2xl border border-slate-200 shadow-sm">
                    <span className="text-sm font-semibold text-slate-600">Difficulty:</span>
                    {["easy", "medium", "exam-level"].map(level => (
                      <button
                        key={level}
                        onClick={() => setQuizDifficulty(level)}
                        className={`px-3 py-1 text-sm rounded-lg capitalize border ${quizDifficulty === level ? "bg-indigo-500 text-white border-indigo-500" : "bg-white text-slate-600 border-slate-200 hover:bg-slate-50"}`}
                      >
                        {level}
                      </button>
                    ))}
                    <button
                      onClick={generateQuiz}
                      disabled={!sessionId || featureLoading}
                      className="ml-auto rounded-xl bg-indigo-500 px-4 py-2 text-sm font-medium text-white shadow-lg transition-all hover:bg-indigo-400 disabled:opacity-50"
                    >
                      {featureLoading ? "Generating..." : "Generate Quiz"}
                    </button>
                  </div>

                  {featureLoading && <div className="text-center py-10 text-slate-500">Generating quiz...</div>}

                  {!featureLoading && quiz.length > 0 && (
                    <div className="flex flex-col gap-8">
                      {quiz.map((q, idx) => (
                        <div key={idx} className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm space-y-4">
                          <p className="font-medium text-slate-800"><span className="text-indigo-500 mr-2">{idx + 1}.</span>{q.question}</p>
                          
                          {q.type === "mcq" && (
                            <div className="grid grid-cols-1 gap-2">
                              {q.options?.map((opt: string, i: number) => {
                                const isSelected = quizAnswers[idx] === String.fromCharCode(65 + i);
                                const isCorrect = q.correct === String.fromCharCode(65 + i);
                                let btnClass = "border-slate-200 hover:bg-slate-50 text-slate-700";
                                if (quizSubmitted) {
                                  if (isCorrect) btnClass = "border-emerald-500 bg-emerald-50 text-emerald-700";
                                  else if (isSelected && !isCorrect) btnClass = "border-red-500 bg-red-50 text-red-700";
                                  else btnClass = "border-slate-200 bg-slate-50 text-slate-400";
                                } else if (isSelected) {
                                  btnClass = "border-indigo-500 bg-indigo-50 text-indigo-700";
                                }
                                return (
                                  <button
                                    key={i}
                                    disabled={quizSubmitted}
                                    onClick={() => setQuizAnswers(prev => ({...prev, [idx]: String.fromCharCode(65 + i)}))}
                                    className={`text-left px-4 py-3 rounded-xl border text-sm transition-colors ${btnClass}`}
                                  >
                                    {opt}
                                  </button>
                                );
                              })}
                            </div>
                          )}

                          {q.type === "truefalse" && (
                            <div className="flex gap-4">
                              {[true, false].map((val) => {
                                const isSelected = quizAnswers[idx] === String(val);
                                const isCorrect = String(q.correct) === String(val);
                                let btnClass = "border-slate-200 hover:bg-slate-50 text-slate-700";
                                if (quizSubmitted) {
                                  if (isCorrect) btnClass = "border-emerald-500 bg-emerald-50 text-emerald-700";
                                  else if (isSelected && !isCorrect) btnClass = "border-red-500 bg-red-50 text-red-700";
                                  else btnClass = "border-slate-200 bg-slate-50 text-slate-400";
                                } else if (isSelected) {
                                  btnClass = "border-indigo-500 bg-indigo-50 text-indigo-700";
                                }
                                return (
                                  <button
                                    key={String(val)}
                                    disabled={quizSubmitted}
                                    onClick={() => setQuizAnswers(prev => ({...prev, [idx]: String(val)}))}
                                    className={`flex-1 px-4 py-3 rounded-xl border text-sm transition-colors text-center font-medium ${btnClass}`}
                                  >
                                    {val ? "True" : "False"}
                                  </button>
                                );
                              })}
                            </div>
                          )}

                          {q.type === "shortanswer" && (
                            <div>
                              <textarea
                                disabled={quizSubmitted}
                                value={quizAnswers[idx] || ""}
                                onChange={(e) => setQuizAnswers(prev => ({...prev, [idx]: e.target.value}))}
                                placeholder="Type your answer here..."
                                className="w-full min-h-[100px] p-3 rounded-xl border border-slate-200 text-sm focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500 outline-none resize-none"
                              />
                              {quizSubmitted && (
                                <div className="mt-4 p-4 rounded-xl bg-blue-50 border border-blue-100">
                                  <p className="text-xs font-semibold text-blue-600 uppercase mb-1">Model Answer:</p>
                                  <p className="text-sm text-blue-800">{q.model_answer}</p>
                                </div>
                              )}
                            </div>
                          )}

                          {quizSubmitted && q.explanation && (
                            <div className="mt-4 p-3 bg-slate-50 rounded-lg text-sm text-slate-600 border border-slate-100">
                              <span className="font-semibold">Explanation: </span>{q.explanation}
                            </div>
                          )}
                        </div>
                      ))}
                      
                      {!quizSubmitted && (
                        <button
                          onClick={() => setQuizSubmitted(true)}
                          className="w-full py-4 rounded-2xl bg-indigo-600 text-white font-medium text-lg hover:bg-indigo-500 shadow-xl"
                        >
                          Submit Quiz
                        </button>
                      )}
                      
                      {quizSubmitted && (
                        <div className="text-center p-6 bg-slate-900 border border-slate-700 rounded-2xl shadow-xl">
                          <h3 className="text-2xl font-bold text-white mb-2">Quiz Completed!</h3>
                          <p className="text-slate-300">Scroll up to check your answers and read the explanations.</p>
                          <button
                            onClick={() => { setQuizSubmitted(false); setQuizAnswers({}); setQuiz([]); }}
                            className="mt-6 px-6 py-2 bg-indigo-500 text-white rounded-xl hover:bg-indigo-400"
                          >
                            Try Another Quiz
                          </button>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}

              {activeTab === "formulas" && (
                <div className="flex flex-col gap-6 max-w-3xl mx-auto w-full">
                  <div className="flex justify-end">
                    <button
                      onClick={generateFormulas}
                      disabled={!sessionId || featureLoading}
                      className="rounded-xl bg-indigo-500 px-4 py-2 text-sm font-medium text-white shadow-lg transition-all hover:bg-indigo-400 disabled:opacity-50"
                    >
                      {featureLoading ? "Extracting..." : "Extract Formulas"}
                    </button>
                  </div>

                  {featureLoading && <div className="text-center py-10 text-slate-500">Scanning document for formulas...</div>}

                  {!featureLoading && formulas.length === 0 && (
                    <div className="text-center py-10 text-slate-500">No formulas found in the document.</div>
                  )}

                  {!featureLoading && formulas.length > 0 && (
                    <div className="flex flex-col gap-4">
                      {formulas.map((form, idx) => (
                        <div key={idx} className="bg-white rounded-2xl border border-slate-200 shadow-sm p-5 space-y-4">
                          <div className="flex items-start justify-between">
                            <h3 className="font-bold text-lg text-slate-800">{form.name}</h3>
                            <span className="px-2.5 py-1 rounded-md bg-purple-50 text-purple-600 text-xs font-semibold uppercase tracking-wider">{form.domain}</span>
                          </div>
                          <div className="bg-slate-900 rounded-xl p-4 overflow-x-auto text-white">
                            <ReactMarkdown
                              remarkPlugins={[remarkMath]}
                              rehypePlugins={[rehypeKatex]}
                            >
                              {`$$${form.latex}$$`}
                            </ReactMarkdown>
                          </div>
                          {form.variables && Object.keys(form.variables).length > 0 && (
                            <div className="mt-4 border-t border-slate-100 pt-4">
                              <p className="text-xs font-semibold uppercase text-slate-400 mb-2">Variables</p>
                              <div className="grid grid-cols-1 md:grid-cols-2 gap-2 text-sm">
                                {Object.entries(form.variables).map(([v, desc]) => (
                                  <div key={v} className="flex gap-2">
                                    <span className="font-mono font-bold text-indigo-600 bg-indigo-50 px-1.5 rounded">{v}</span>
                                    <span className="text-slate-600">{String(desc)}</span>
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>

            {activeTab === "chat" && (
            <div className="sticky bottom-0 border-t border-slate-200 bg-white/80 px-6 py-4 backdrop-blur">
              <div className="flex items-center gap-3 rounded-2xl border border-slate-200 bg-white px-4 py-3 shadow-sm transition-all duration-200 focus-within:border-indigo-300 focus-within:ring-2 focus-within:ring-indigo-100">
                <input
                  type="text"
                  placeholder="Ask something about your document..."
                  value={input}
                  onChange={(event) => setInput(event.target.value)}
                  onKeyDown={handleKeyDown}
                  className="flex-1 bg-transparent text-sm text-slate-900 outline-none"
                />
                <button
                  type="button"
                  onClick={sendMessage}
                  disabled={!canSend}
                  className="rounded-xl bg-slate-900 px-4 py-2 text-sm font-medium text-white transition-all duration-200 hover:-translate-y-0.5 hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-50"
                >
                  Send
                </button>
              </div>
            </div>
            )}
          </div>
        </main>
      </div>
    </div>
  );
}
