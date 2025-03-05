full_templates = {
    "paper": "Read the papers below and answer a question \n\n{context}\n\nQuestion: {question}\n\n Be concise.",
    "financial": "Read the financial statement below and answer a question \n\n{context}\n\nQuestion: {question}\n\n Be concise.",
    "book": "Read the book below and answer a question.\n\n{context}\n\nQuestion: {question}\n\nBe concise.",  # noqa
}

rag_templates = {
    "paper": "Here are some chunks retrieved from some papers. Read these chunks to answer a question.\n\n{context}\n\nQuestion: {question}\n\n Be concise.",
    "financial": "Here are some chunks retrieved from a financial statement. Read these chunks to answer a question. \n\n{context}\n\nQuestion: {question}\n\n Be concise.",
    "book": "Here are some chunks retrieved from a book. Read these chunks to answer a question.\n\n{context}\n\nQuestion: {question}\n\nBe concise.",  # noqa
}