"""One model writes small programs that run inside its own forward pass while it fills in sudoku cells.

    loop.py     the outer loop: the model writes, runs, verifies and improves a program for itself
    check.py    check a program file on the practice and held-out boards
    model/      the model, and running programs inside its forward pass with peek and push (live.py)
    eval/       the practice cells with feedback, playing whole boards, the check on held-out boards
    setup/      settings and the log
    sudoku/     boards, the board prompt, the board sets
"""
