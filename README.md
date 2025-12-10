Recursive-Descent Parser - Mini-Language

This application implements a recursive-descent parser for a mini programming language that supports:

- If-else statements

- Variable assignments with arithmetic operators

- Sequential statement execution

- The parser validates input code against a formal context-free grammar (CFG) and reports whether the syntax is valid.

Features:

Tokenization - A lexical analysis of source code into tokens

Parsing - A top-down recursive-descent parsing with one-token lookahead

Dangling-Else Resolution - Unambiguous grammar using matched/unmatched statement distinction

GUI Interface - A Tkinter-based graphical interface for easy interaction

Error Reporting - With precise error messages with position information

Requirements:

Python 3.7+
tkinter

Installation:

Save the Python code to a file named parser.py

Ensure Python 3 is installed:

python3 --version

Run the application:

python3 parser.py

Usage:

GUI Mode

Launch the application:

python3 parser.py

Enter code in the input text area, for example:

text
if (x == 5) {
    y = x + 1;
} else {
    y = x - 1;
}

Click "Parse & Validate" to check syntax

View results in the output area:

✓ ACCEPT: Valid syntax

✗ REJECT: [error details]