"""A frozen model fills in sudoku cells while a program reads and writes words inside it.

    loop.py     the outer loop: writers build programs step by step
    agent.py    a writer builds a program by calling tools (run, submit)
    setup/      settings, logging, loading the model and its words
    model/      the frozen model: loading, its state at word places, replies
    words/      finding the words: answer words and board words
    programs/   the program language, running programs, writing them
    eval/       scoring programs: practice cells, playing boards, the check on held-out boards
    sudoku/     boards, the board prompt, the board sets
"""
