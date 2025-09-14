import gradio as gr
from app.pipeline.chat_pipeline import answer_with_rag

def respond(user_message: str, chat_history: list[dict]):
    chat_history = (chat_history or []) + [{"role": "user", "content": user_message}]

    reply, sources, emotion = answer_with_rag(user_message, k=4)
    reply_with_sources = (
        f"**Detected emotion:** {emotion}\n\n"
        f"{reply}\n\n**Sources:** {', '.join(sources)}"
    )

    chat_history.append({"role": "assistant", "content": reply_with_sources})
    return "", chat_history

def main():
    with gr.Blocks(theme="soft") as demo:
        gr.Markdown("# 🧘 SerenityAI\nChat with a local Llama + RAG knowledge base.")

        chatbot = gr.Chatbot(height=450, type="messages")
        with gr.Row():
            txt = gr.Textbox(
                placeholder="Type your message here...",
                lines=2,
                scale=8,
                show_label=False,
            )
            send = gr.Button("Send", variant="primary", scale=1)

        clear = gr.Button("Clear Chat")

        # Enter key submits
        txt.submit(respond, [txt, chatbot], [txt, chatbot])
        # Send button also submits
        send.click(respond, [txt, chatbot], [txt, chatbot])
        # Clear button clears chat + input
        clear.click(lambda: ([], ""), None, [chatbot, txt], queue=False)

    return demo

if __name__ == "__main__":
    demo = main()
    demo.launch(server_name="127.0.0.1", server_port=7860)
