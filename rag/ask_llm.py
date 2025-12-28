# import pckgs
import ollama

def ask_llm(context, question):
    prompt = f"""
        You are a legal assistant.

        Answer the question using only the information provided below.
        Explain the legal principle clearly and cautiously.

        Important rules:
        - Do NOT mention sources, documents, or “provided context”.
        - Do NOT attribute laws, statutes, or governing acts unless they are explicitly stated.
        - Do NOT create numbered or exhaustive lists unless the text itself lists them.
        - Stay strictly within the legal domain discussed in the text.
        - If the information is insufficient or not clearly stated, say so briefly and honestly.

        INFORMATION:
        {context}

        QUESTION:
        {question}

    """

    # generate response
    response = ollama.chat(
        model="mistral",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return response["message"]["content"]
