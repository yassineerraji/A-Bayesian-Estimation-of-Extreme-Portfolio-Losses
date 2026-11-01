For the full duration of the project, the workflow will be as follows:

We will be working step by step and not one-shot the full project.

You will explain very concisely your thoughts, assumptions and rationale in a folder called AgentDiscussion, where you will add a new markdown file at the beginning of each of the steps of the project.

I will review in details the markdown files and either leave them as is or bring modification to them.

You will check the last state of the markdown file and build according to it.

After you end coding, write a recap markdown file in the same AgentDiscussion folder and wait for me to review the code and the markdown file before signaling that we can proceed with the next step of the project.

OUR SUPREME GOAL IS TO MAXIMIZE THE EXPECTED GRADE FOR THIS PROJECT. Therefore, we need to follow very precisely the official instructions file, and afterwards the submission template. We will also need to pay particular attention to. the way we code.

We will be coding in Python scripts (.py), each script covering one part of the implementation of our project.
Reusable code (model, sampler, diagnostics) is written once in a module and imported by the other scripts, never copy-pasted.
Each script runs from start to end with a single command (`python <script>.py`), uses fixed random seeds, prints its key numbers, and saves every figure and table it produces to a file.

IMPORTANT: WE MUST CODE IN THE MOST SIMPLE WAY POSSIBLE.

WHAT IS MEANT BY SIMPLE IS:


% We always privilege code minimality, what can be done in two lines of codes must be done in no more than two lines, always.

We always privilege simplicity and correctness over complexity and long code that is hard to visually understand.
Every script starts with a module docstring explaining what it does, what it reads, and what it produces (figures, tables, files).
Every function has a short docstring: a one-line summary, then its arguments and returns when they are not obvious.
We always include a comment at the beginning of each block of code to explain what is being done.

All the steps we code must build on top of the previous ones.
When doing assumptions, or making decisions, they must never be done silently and appear in the form of concise comments in the code, tagged `# ASSUMPTION:` or `# DECISION:` so they are easy to find.

Every script must follow the same clean and coherent structure: module docstring, imports, constants (including the random seed), functions, then the main code split into numbered sections (`# %% 1. Title`; VS Code can run each `# %%` section on its own, like a notebook cell).

Also, throughout the implementation, every time we are using a concept from the course, I want a visual comment flag in the code with a few info on what concept we are using and where exactly I can find it in the lecture notes, always in the same format:

```python
# ======================================================================
# [COURSE] Metropolis–Hastings: Ch. 6, Sec. 6.2, Algorithm 4
# Accept with probability min{1, π̃(θ')/π̃(θ)}: only ratios are needed.
# ======================================================================
```