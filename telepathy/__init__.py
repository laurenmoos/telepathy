"""A model writes a program that runs inside its own prompt pass, then attempts a task with it.

    loop.py         the loop: read, match, write, trace, check, cache (one problem at a time)
    model/          the model: the prompt pass with a hook after each layer, and generation
    instructions/   read instructions: candidates from the problem at hand, and the cache
    programs/       the llm_program: running it, tracing it, the model writing it
    tasks/          the tasks; only these know about the task (tasks/lean: Lean on miniF2F)
    setup/          settings and the log
"""
