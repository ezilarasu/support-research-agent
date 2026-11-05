"""
Module: core
Layer:  Infrastructure
Owner:  core/

Purpose:
    Infrastructure layer — configuration, logging, factories, exceptions.
    Nothing in this package knows about LangGraph graphs or business logic.
    Everything else imports from here; this package imports from nothing internal.

Public API:
    get_settings()        — returns the singleton Settings instance
    configure_logging()   — sets up structured logging at app startup
    get_checkpointer()    — returns the active checkpointer (Concept 23)
    get_vector_store()    — returns the active vector store (Module 9 stub)
"""
