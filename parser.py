#!/usr/bin/env python3
"""
Recursive-Descent Parser for Mini-Language
===========================================
Grammar: If-else statements, assignments, sequential statements
Algorithm: Top-down recursive descent with one-token lookahead
GUI: Tkinter-based input/output interface
"""

import tkinter as tk
from tkinter import scrolledtext, messagebox
import re
from enum import Enum, auto


# ============================================================================
# TOKENIZER
# ============================================================================

class TokenType(Enum):
    """Enumeration of token types in the mini-language."""
    IF = auto()
    ELSE = auto()
    LBRACE = auto()
    RBRACE = auto()
    LPAREN = auto()
    RPAREN = auto()
    ASSIGN = auto()
    SEMICOLON = auto()
    PLUS = auto()
    MINUS = auto()
    EQ = auto()
    NEQ = auto()
    ID = auto()
    NUM = auto()
    EOF = auto()


class Token:
    """Represents a single token with type and value."""
    def __init__(self, token_type, value, position):
        self.type = token_type
        self.value = value
        self.position = position
    
    def __repr__(self):
        return f"Token({self.type.name}, '{self.value}', pos={self.position})"


class Tokenizer:
    """Converts input string into a stream of tokens."""
    
    TOKEN_PATTERNS = [
        (r'\bif\b', TokenType.IF),
        (r'\belse\b', TokenType.ELSE),
        (r'\{', TokenType.LBRACE),
        (r'\}', TokenType.RBRACE),
        (r'\(', TokenType.LPAREN),
        (r'\)', TokenType.RPAREN),
        (r'==', TokenType.EQ),
        (r'!=', TokenType.NEQ),
        (r'=', TokenType.ASSIGN),
        (r';', TokenType.SEMICOLON),
        (r'\+', TokenType.PLUS),
        (r'-', TokenType.MINUS),
        (r'[a-zA-Z_][a-zA-Z0-9_]*', TokenType.ID),
        (r'\d+', TokenType.NUM),
        (r'\s+', None),  # Whitespace (ignored)
    ]
    
    def __init__(self, source):
        self.source = source
        self.tokens = []
        self.tokenize()
    
    def tokenize(self):
        """Scan source and generate token list."""
        position = 0
        while position < len(self.source):
            matched = False
            for pattern, token_type in self.TOKEN_PATTERNS:
                regex = re.compile(pattern)
                match = regex.match(self.source, position)
                if match:
                    value = match.group(0)
                    if token_type:  # Ignore whitespace
                        self.tokens.append(Token(token_type, value, position))
                    position = match.end()
                    matched = True
                    break
            
            if not matched:
                raise SyntaxError(f"Illegal character '{self.source[position]}' at position {position}")
        
        # Add EOF token
        self.tokens.append(Token(TokenType.EOF, '', position))
    
    def get_tokens(self):
        """Return the list of tokens."""
        return self.tokens


# ============================================================================
# RECURSIVE-DESCENT PARSER
# ============================================================================

class Parser:
    """Recursive-descent parser for the mini-language CFG."""
    
    def __init__(self, tokens):
        self.tokens = tokens
        self.current_index = 0
        self.current_token = self.tokens[0] if tokens else None
    
    def advance(self):
        """Move to the next token."""
        if self.current_index < len(self.tokens) - 1:
            self.current_index += 1
            self.current_token = self.tokens[self.current_index]
    
    def peek(self):
        """Return current token type without advancing."""
        return self.current_token.type if self.current_token else TokenType.EOF
    
    def match(self, expected_type):
        """Consume token if it matches expected type."""
        if self.peek() == expected_type:
            token = self.current_token
            self.advance()
            return token
        else:
            raise SyntaxError(
                f"Expected {expected_type.name} but found {self.peek().name} "
                f"at position {self.current_token.position}"
            )
    
    # Grammar Rules
    
    def parse_program(self):
        """Program → StmtList"""
        self.parse_stmt_list()
        if self.peek() != TokenType.EOF:
            raise SyntaxError(
                f"Unexpected token {self.peek().name} at position {self.current_token.position}"
            )
    
    def parse_stmt_list(self):
        """StmtList → Stmt StmtList | ε"""
        while self.peek() != TokenType.EOF and self.peek() != TokenType.RBRACE:
            self.parse_stmt()
    
    def parse_stmt(self):
        """Stmt → MatchedStmt | UnmatchedStmt"""
        if self.peek() == TokenType.IF:
            self.parse_if_stmt()
        elif self.peek() == TokenType.ID:
            self.parse_assign()
        elif self.peek() == TokenType.LBRACE:
            self.parse_block()
        else:
            raise SyntaxError(
                f"Unexpected statement starting with {self.peek().name} "
                f"at position {self.current_token.position}"
            )
    
    def parse_if_stmt(self):
        """
        Parse if-statement (matched or unmatched).
        Uses lookahead to determine if else-clause exists.
        """
        self.match(TokenType.IF)
        self.match(TokenType.LPAREN)
        self.parse_expr()
        self.match(TokenType.RPAREN)
        self.match(TokenType.LBRACE)
        self.parse_stmt_list()
        self.match(TokenType.RBRACE)
        
        # Check for else clause
        if self.peek() == TokenType.ELSE:
            self.match(TokenType.ELSE)
            self.match(TokenType.LBRACE)
            self.parse_stmt_list()
            self.match(TokenType.RBRACE)
    
    def parse_block(self):
        """Block → { StmtList }"""
        self.match(TokenType.LBRACE)
        self.parse_stmt_list()
        self.match(TokenType.RBRACE)
    
    def parse_assign(self):
        """Assign → ID = Expr ;"""
        self.match(TokenType.ID)
        self.match(TokenType.ASSIGN)
        self.parse_expr()
        self.match(TokenType.SEMICOLON)
    
    def parse_expr(self):
        """Expr → Term ((+|-|==|!=) Term)?"""
        self.parse_term()
        if self.peek() in [TokenType.PLUS, TokenType.MINUS, TokenType.EQ, TokenType.NEQ]:
            self.advance()
            self.parse_term()
    
    def parse_term(self):
        """Term → ID | NUM"""
        if self.peek() == TokenType.ID:
            self.match(TokenType.ID)
        elif self.peek() == TokenType.NUM:
            self.match(TokenType.NUM)
        else:
            raise SyntaxError(
                f"Expected identifier or number but found {self.peek().name} "
                f"at position {self.current_token.position}"
            )
    
    def parse(self):
        """Main entry point for parsing."""
        try:
            self.parse_program()
            return True, "ACCEPT: Valid syntax"
        except SyntaxError as e:
            return False, f"REJECT: {str(e)}"


# ============================================================================
# TKINTER GUI APPLICATION
# ============================================================================

class ParserGUI:
    """Tkinter-based GUI for the recursive-descent parser."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Recursive-Descent Parser - Mini-Language")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        self.setup_ui()
    
    def setup_ui(self):
        """Create GUI components."""
        # Title
        title_label = tk.Label(
            self.root, 
            text="Mini-Language Parser (If-Else, Assignments, Sequences)",
            font=("Arial", 14, "bold"),
            bg="#2c3e50",
            fg="white",
            pady=10
        )
        title_label.pack(fill="x")
        
        # Input section
        input_frame = tk.Frame(self.root, padx=10, pady=10)
        input_frame.pack(fill="both", expand=True)
        
        tk.Label(input_frame, text="Input Code:", font=("Arial", 11, "bold")).pack(anchor="w")
        
        self.input_text = scrolledtext.ScrolledText(
            input_frame,
            height=12,
            width=90,
            font=("Courier New", 10),
            wrap=tk.WORD
        )
        self.input_text.pack(fill="both", expand=True, pady=(5, 10))
        
        # Buttons
        button_frame = tk.Frame(self.root)
        button_frame.pack(pady=5)
        
        parse_btn = tk.Button(
            button_frame,
            text="Parse & Validate",
            command=self.parse_input,
            bg="#27ae60",
            fg="white",
            font=("Arial", 11, "bold"),
            padx=20,
            pady=8,
            cursor="hand2"
        )
        parse_btn.pack(side="left", padx=5)
        
        clear_btn = tk.Button(
            button_frame,
            text="Clear",
            command=self.clear_all,
            bg="#e74c3c",
            fg="white",
            font=("Arial", 11, "bold"),
            padx=20,
            pady=8,
            cursor="hand2"
        )
        clear_btn.pack(side="left", padx=5)
        
        example_btn = tk.Button(
            button_frame,
            text="Load Example",
            command=self.load_example,
            bg="#3498db",
            fg="white",
            font=("Arial", 11, "bold"),
            padx=20,
            pady=8,
            cursor="hand2"
        )
        example_btn.pack(side="left", padx=5)
        
        # Output section
        output_frame = tk.Frame(self.root, padx=10, pady=10)
        output_frame.pack(fill="both", expand=True)
        
        tk.Label(output_frame, text="Output:", font=("Arial", 11, "bold")).pack(anchor="w")
        
        self.output_text = scrolledtext.ScrolledText(
            output_frame,
            height=8,
            width=90,
            font=("Courier New", 10),
            wrap=tk.WORD,
            state="disabled",
            bg="#ecf0f1"
        )
        self.output_text.pack(fill="both", expand=True, pady=(5, 10))
    
    def parse_input(self):
        """Parse the input code and display result."""
        source_code = self.input_text.get("1.0", tk.END).strip()
        
        if not source_code:
            messagebox.showwarning("Empty Input", "Please enter code to parse.")
            return
        
        self.output_text.config(state="normal")
        self.output_text.delete("1.0", tk.END)
        
        try:
            # Tokenization
            tokenizer = Tokenizer(source_code)
            tokens = tokenizer.get_tokens()
            
            self.output_text.insert(tk.END, "TOKENIZATION:\n")
            self.output_text.insert(tk.END, "-" * 70 + "\n")
            for token in tokens:
                if token.type != TokenType.EOF:
                    self.output_text.insert(tk.END, f"{token}\n")
            
            # Parsing
            self.output_text.insert(tk.END, "\n" + "=" * 70 + "\n")
            self.output_text.insert(tk.END, "PARSING RESULT:\n")
            self.output_text.insert(tk.END, "=" * 70 + "\n")
            
            parser = Parser(tokens)
            success, message = parser.parse()
            
            if success:
                self.output_text.insert(tk.END, "✓ " + message + "\n", "success")
                self.output_text.tag_config("success", foreground="#27ae60", font=("Courier New", 11, "bold"))
            else:
                self.output_text.insert(tk.END, "✗ " + message + "\n", "error")
                self.output_text.tag_config("error", foreground="#e74c3c", font=("Courier New", 11, "bold"))
        
        except SyntaxError as e:
            self.output_text.insert(tk.END, f"✗ REJECT: Tokenization error - {str(e)}\n", "error")
            self.output_text.tag_config("error", foreground="#e74c3c", font=("Courier New", 11, "bold"))
        
        except Exception as e:
            self.output_text.insert(tk.END, f"✗ REJECT: Unexpected error - {str(e)}\n", "error")
            self.output_text.tag_config("error", foreground="#e74c3c", font=("Courier New", 11, "bold"))
        
        finally:
            self.output_text.config(state="disabled")
    
    def clear_all(self):
        """Clear input and output fields."""
        self.input_text.delete("1.0", tk.END)
        self.output_text.config(state="normal")
        self.output_text.delete("1.0", tk.END)
        self.output_text.config(state="disabled")
    
    def load_example(self):
        """Load example code into input field."""
        example = """if (a == b) {
    a = a + 1;
} else {
    b = b - 1;
}"""
        self.input_text.delete("1.0", tk.END)
        self.input_text.insert("1.0", example)


# ============================================================================
# MAIN APPLICATION ENTRY POINT
# ============================================================================

def main():
    """Launch the Tkinter application."""
    root = tk.Tk()
    app = ParserGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
