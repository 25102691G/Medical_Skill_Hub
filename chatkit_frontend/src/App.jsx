import { ChatKit, useChatKit } from "@openai/chatkit-react";
import { useCallback, useEffect, useState } from "react";


export default function App() {
  const [displayLanguage, setDisplayLanguage] = useState(
    () =>
      window.localStorage.getItem("medical-skill-hub-language") === "en"
        ? "en"
        : "zh-CN",
  );
  const [threadId, setThreadId] = useState(
    () => window.localStorage.getItem("medical-skill-hub-thread"),
  );
  const [responseRunning, setResponseRunning] = useState(false);

  const text = {
    "zh-CN": {
      title: "消化内科诊断助手",
      instructions:
        "分次发送病例资料，完成后发送“开始诊断”；发送“清空病例”可重置当前病例。",
      languageLabel: "显示语言",
      placeholder: "请输入病例资料或诊断指令",
      chatLabel: "医学诊断聊天窗口",
      disclaimer: "仅用于技术演示和辅助分析，不能替代临床医生诊断。",
    },
    en: {
      title: "Gastroenterology Diagnosis Assistant",
      instructions:
        'Send the case in one or more messages, then send “start diagnosis”. Send “clear case” to reset it.',
      languageLabel: "Display language",
      placeholder: "Enter case information or a diagnosis command",
      chatLabel: "Medical diagnosis chat",
      disclaimer:
        "For technical demonstration and decision support only. It does not replace a clinician's diagnosis.",
    },
  }[displayLanguage];

  const localizedFetch = useCallback(
    (input, init = {}) => {
      const requestHeaders =
        input instanceof Request ? new Headers(input.headers) : new Headers();
      const headers = new Headers(requestHeaders);
      new Headers(init.headers).forEach((value, key) => headers.set(key, value));
      headers.set("X-Display-Language", displayLanguage);
      return window.fetch(input, { ...init, headers });
    },
    [displayLanguage],
  );

  const chatkit = useChatKit({
    api: {
      url: "http://localhost:8000/chatkit",
      domainKey: "local-dev",
      fetch: localizedFetch,
    },
    initialThread: threadId,
    locale: displayLanguage,
    composer: {
      placeholder: text.placeholder,
    },
    onError: () => setResponseRunning(false),
    onResponseStart: () => setResponseRunning(true),
    onResponseEnd: () => setResponseRunning(false),
    onThreadChange: ({ threadId: nextThreadId }) => {
      setThreadId(nextThreadId);
      if (nextThreadId) {
        window.localStorage.setItem("medical-skill-hub-thread", nextThreadId);
      } else {
        window.localStorage.removeItem("medical-skill-hub-thread");
      }
    },
  });

  useEffect(() => {
    document.documentElement.lang = displayLanguage;
    window.localStorage.setItem("medical-skill-hub-language", displayLanguage);
  }, [displayLanguage]);

  return (
    <main className="page-shell">
      <header className="app-header">
        <div>
          <p className="eyebrow">Medical Skill Hub</p>
          <h1>{text.title}</h1>
        </div>
        <div className="header-controls">
          <label className="language-control">
            <span>{text.languageLabel}</span>
            <select
              value={displayLanguage}
              disabled={responseRunning}
              onChange={(event) => setDisplayLanguage(event.target.value)}
            >
              <option value="zh-CN">简体中文</option>
              <option value="en">English</option>
            </select>
          </label>
          <p className="instructions">{text.instructions}</p>
        </div>
      </header>
      <section className="chat-panel" aria-label={text.chatLabel}>
        <ChatKit key={displayLanguage} control={chatkit.control} />
      </section>
      <footer>{text.disclaimer}</footer>
    </main>
  );
}
