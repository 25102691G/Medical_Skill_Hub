import { ChatKit, useChatKit } from "@openai/chatkit-react";


export default function App() {
  const chatkit = useChatKit({
    api: {
      url: "http://localhost:8000/chatkit",
      domainKey: "local-dev",
    },
    locale: "zh-CN",
    composer: {
      placeholder: "请输入病例资料或诊断指令",
    },
  });

  return (
    <main className="page-shell">
      <header className="app-header">
        <div>
          <p className="eyebrow">Medical Skill Hub</p>
          <h1>消化内科诊断助手</h1>
        </div>
        <p className="instructions">
          分次发送病例资料，完成后发送“开始诊断”；发送“清空病例”可重置当前病例。
        </p>
      </header>
      <section className="chat-panel" aria-label="医学诊断聊天窗口">
        <ChatKit control={chatkit.control} />
      </section>
      <footer>仅用于技术演示和辅助分析，不能替代临床医生诊断。</footer>
    </main>
  );
}
