from app.llm import chat_once

if __name__ == "__main__":
    print("Running minimal chat check…")
    out = chat_once("Say hello in one friendly sentence.")
    print("\n--- Response ---")
    print(out)