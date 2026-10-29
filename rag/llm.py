"""
rag/llm.py - builds the chat model. Provider is chosen with LLM_PROVIDER in your .env file,
so you can move from Groq to OpenAI or a local Ollama model WITHOUT changing any code.
"""
from __future__ import annotations

import os

from config import Settings


def build_llm(settings: Settings):
    provider = settings.llm_provider.lower()

    if provider == "groq":
        if not os.getenv("GROQ_API_KEY"):
            raise RuntimeError("GROQ_API_KEY is missing. Add it to your .env file.")
        from langchain_groq import ChatGroq   # imported here so unused providers need no install
        # ChatGroq reads GROQ_API_KEY from the environment by itself.
        return ChatGroq(model=settings.llm_model, temperature=settings.temperature,
                        max_tokens=settings.max_tokens)

    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is missing. Add it to your .env file.")
        from langchain_openai import ChatOpenAI   # pip install langchain-openai
        return ChatOpenAI(model=settings.llm_model, temperature=settings.temperature,
                          max_completion_tokens=settings.max_tokens)

    if provider == "ollama":                       # free, runs on your own computer
        from langchain_ollama import ChatOllama   # pip install langchain-ollama
        return ChatOllama(model=settings.llm_model, temperature=settings.temperature)

    raise ValueError(f"Unknown LLM_PROVIDER '{settings.llm_provider}'. Use groq, openai or ollama.")
