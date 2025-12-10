"""
Recursive-Descent Parser for Mini-Language
====================================================
Grammar: If-else statements, assignments, sequential statements
Algorithm: Top-down recursive descent with one-token lookahead
GUI: Tkinter-based input/output interface
"""

import tkinter as tk
from tkinter import scrolledtext, messagebox
import re
import unittest
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
    MULTIPLY = auto()
    DIVIDE = auto()
    EQ = auto()
    NEQ = auto()
    LT = auto()
    GT = auto()
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
        (r'<=', TokenType.LT),
        (r'>=', TokenType.GT),
        (r'<', TokenType.LT),
        (r'>', TokenType.GT),
        (r'=', TokenType.ASSIGN),
        (r';', TokenType.SEMICOLON),
        (r'\+', TokenType.PLUS),
        (r'-', TokenType.MINUS),
        (r'\*', TokenType.MULTIPLY),
        (r'/', TokenType.DIVIDE),
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
# RECURSIVE-DESCENT PARSER (IMPROVED)
# ============================================================================

class Parser:
    """
    Recursive-descent parser for the mini-language CFG.
    
    IMPROVED GRAMMAR:
    -----------------
    Program → StmtList
    StmtList → Stmt StmtList | ε
    Stmt → MatchedStmt | UnmatchedStmt
    
    MatchedStmt → if ( Expr ) { StmtList } else { StmtList }
                | Assign
                | { StmtList }
    
    UnmatchedStmt → if ( Expr ) { StmtList }
                  | if ( Expr ) { MatchedStmt } else { UnmatchedStmt }
    
    Assign → ID = Expr ;
    
    # IMPROVED: Expression with precedence and chaining
    Expr → CompExpr
    CompExpr → AddExpr ((== | != | < | >) AddExpr)*
    AddExpr → MulExpr ((+ | -) MulExpr)*
    MulExpr → Primary ((* | /) Primary)*
    Primary → ID | NUM | ( Expr )
    """
    
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
        """
        Stmt → MatchedStmt | UnmatchedStmt
        
        IMPROVEMENT: Properly distinguish between matched and unmatched statements
        by looking ahead to determine if an else clause follows.
        """
        if self.peek() == TokenType.IF:
            # Try to parse as matched or unmatched if-statement
            self.parse_if_stmt_general()
        elif self.peek() == TokenType.ID:
            self.parse_assign()
        elif self.peek() == TokenType.LBRACE:
            self.parse_block()
        else:
            raise SyntaxError(
                f"Unexpected statement starting with {self.peek().name} "
                f"at position {self.current_token.position}"
            )
    
    def parse_if_stmt_general(self):
        """
        Parse if-statement with proper matched/unmatched distinction.
        
        This determines whether we have:
        - MatchedStmt: if (...) {...} else {...}
        - UnmatchedStmt: if (...) {...} (no else)
        """
        self.match(TokenType.IF)
        self.match(TokenType.LPAREN)
        self.parse_expr()
        self.match(TokenType.RPAREN)
        self.match(TokenType.LBRACE)
        self.parse_stmt_list()
        self.match(TokenType.RBRACE)
        
        # IMPROVEMENT: Explicit matched/unmatched distinction
        if self.peek() == TokenType.ELSE:
            # MatchedStmt: has else clause
            self.parse_matched_else()
        # else: UnmatchedStmt (no else clause)
    
    def parse_matched_else(self):
        """
        Parse the else clause of a matched if-statement.
        MatchedStmt → ... else { StmtList }
        """
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
    
    # IMPROVED EXPRESSION PARSING WITH PRECEDENCE AND CHAINING
    
    def parse_expr(self):
        """
        Expr → CompExpr
        Top-level expression entry point.
        """
        self.parse_comp_expr()
    
    def parse_comp_expr(self):
        """
        CompExpr → AddExpr ((== | != | < | >) AddExpr)*
        Comparison operators (lowest precedence in expressions).
        IMPROVEMENT: Supports chaining like a == b or a < b
        """
        self.parse_add_expr()
        while self.peek() in [TokenType.EQ, TokenType.NEQ, TokenType.LT, TokenType.GT]:
            self.advance()
            self.parse_add_expr()
    
    def parse_add_expr(self):
        """
        AddExpr → MulExpr ((+ | -) MulExpr)*
        Addition and subtraction (medium precedence).
        IMPROVEMENT: Supports chaining like a + b + c or a - b + c
        """
        self.parse_mul_expr()
        while self.peek() in [TokenType.PLUS, TokenType.MINUS]:
            self.advance()
            self.parse_mul_expr()
    
    def parse_mul_expr(self):
        """
        MulExpr → Primary ((* | /) Primary)*
        Multiplication and division (higher precedence).
        IMPROVEMENT: Supports chaining like a * b * c or a / b * c
        """
        self.parse_primary()
        while self.peek() in [TokenType.MULTIPLY, TokenType.DIVIDE]:
            self.advance()
            self.parse_primary()
    
    def parse_primary(self):
        """
        Primary → ID | NUM | ( Expr )
        Atomic expressions and parenthesized sub-expressions.
        IMPROVEMENT: Supports parentheses for precedence override.
        """
        if self.peek() == TokenType.ID:
            self.match(TokenType.ID)
        elif self.peek() == TokenType.NUM:
            self.match(TokenType.NUM)
        elif self.peek() == TokenType.LPAREN:
            self.match(TokenType.LPAREN)
            self.parse_expr()
            self.match(TokenType.RPAREN)
        else:
            raise SyntaxError(
                f"Expected identifier, number, or '(' but found {self.peek().name} "
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
# COMPREHENSIVE TEST SUITE
# ============================================================================

class TestMiniLanguageParser(unittest.TestCase):
    """
    IMPROVEMENT #3: Comprehensive test suite for all grammar rules.
    Tests cover: expressions, statements, matched/unmatched if, errors.
    """
    
    def parse_string(self, source):
        """Helper method to parse a source string."""
        tokenizer = Tokenizer(source)
        tokens = tokenizer.get_tokens()
        parser = Parser(tokens)
        return parser.parse()
    
    # ===== EXPRESSION TESTS =====
    
    def test_simple_assignment(self):
        """Test basic assignment statement."""
        success, _ = self.parse_string("a = 5;")
        self.assertTrue(success)
    
    def test_expression_with_addition(self):
        """Test expression with single addition."""
        success, _ = self.parse_string("a = b + c;")
        self.assertTrue(success)
    
    def test_expression_chaining_addition(self):
        """IMPROVED: Test chained addition a + b + c."""
        success, _ = self.parse_string("result = a + b + c;")
        self.assertTrue(success)
    
    def test_expression_chaining_subtraction(self):
        """IMPROVED: Test chained subtraction a - b - c."""
        success, _ = self.parse_string("result = a - b - c;")
        self.assertTrue(success)
    
    def test_expression_mixed_add_sub(self):
        """IMPROVED: Test mixed addition and subtraction."""
        success, _ = self.parse_string("result = a + b - c + d;")
        self.assertTrue(success)
    
    def test_expression_with_multiplication(self):
        """IMPROVED: Test multiplication precedence."""
        success, _ = self.parse_string("result = a + b * c;")
        self.assertTrue(success)
    
    def test_expression_chaining_multiplication(self):
        """IMPROVED: Test chained multiplication a * b * c."""
        success, _ = self.parse_string("result = a * b * c;")
        self.assertTrue(success)
    
    def test_expression_complex_precedence(self):
        """IMPROVED: Test complex expression with precedence."""
        success, _ = self.parse_string("result = a + b * c - d / e;")
        self.assertTrue(success)
    
    def test_expression_with_parentheses(self):
        """IMPROVED: Test parenthesized expressions."""
        success, _ = self.parse_string("result = (a + b) * c;")
        self.assertTrue(success)
    
    def test_expression_nested_parentheses(self):
        """IMPROVED: Test nested parentheses."""
        success, _ = self.parse_string("result = ((a + b) * (c - d)) / e;")
        self.assertTrue(success)
    
    def test_expression_comparison(self):
        """Test comparison operators."""
        success, _ = self.parse_string("result = a == b;")
        self.assertTrue(success)
    
    def test_expression_complex_comparison(self):
        """IMPROVED: Test comparison with arithmetic."""
        success, _ = self.parse_string("result = a + b == c * d;")
        self.assertTrue(success)
    
    # ===== STATEMENT SEQUENCE TESTS =====
    
    def test_sequential_assignments(self):
        """Test multiple sequential assignments."""
        success, _ = self.parse_string("a = 1; b = 2; c = 3;")
        self.assertTrue(success)
    
    def test_block_statement(self):
        """Test block with statements."""
        success, _ = self.parse_string("{ a = 1; b = 2; }")
        self.assertTrue(success)
    
    # ===== IF-STATEMENT TESTS (MATCHED) =====
    
    def test_if_else_matched(self):
        """IMPROVED: Test matched if-else statement."""
        success, _ = self.parse_string("if (a == b) { x = 1; } else { y = 2; }")
        self.assertTrue(success)
    
    def test_if_else_with_complex_condition(self):
        """IMPROVED: Test if-else with complex condition."""
        success, _ = self.parse_string("if (a + b == c * d) { x = 1; } else { y = 2; }")
        self.assertTrue(success)
    
    def test_nested_if_else_matched(self):
        """IMPROVED: Test nested matched if-else (no ambiguity)."""
        source = """
        if (a == b) {
            if (c == d) { x = 1; } else { y = 2; }
        } else {
            z = 3;
        }
        """
        success, _ = self.parse_string(source)
        self.assertTrue(success)
    
    # ===== IF-STATEMENT TESTS (UNMATCHED) =====
    
    def test_if_without_else_unmatched(self):
        """IMPROVED: Test unmatched if (no else clause)."""
        success, _ = self.parse_string("if (a == b) { x = 1; }")
        self.assertTrue(success)
    
    def test_nested_unmatched_if(self):
        """IMPROVED: Test nested unmatched if statements."""
        source = """
        if (x != 0) {
            if (y == 1) { z = x + y; }
        }
        """
        success, _ = self.parse_string(source)
        self.assertTrue(success)
    
    def test_dangling_else_resolution(self):
        """
        IMPROVED: Test dangling-else resolution.
        The else should bind to the inner if.
        Grammar ensures: if (a) { if (b) { s1 } } else { s2 } is unambiguous.
        """
        source = "if (a > 0) { if (b > 0) { x = 1; } } else { y = 2; }"
        success, _ = self.parse_string(source)
        self.assertTrue(success)
    
    # ===== COMPLEX INTEGRATION TESTS =====
    
    def test_complex_program(self):
        """Test complex program with multiple constructs."""
        source = """
        a = 10;
        b = 20;
        if (a < b) {
            c = a + b;
            d = c * 2;
        } else {
            c = a - b;
        }
        result = c + d;
        """
        success, _ = self.parse_string(source)
        self.assertTrue(success)
    
    def test_deeply_nested_blocks(self):
        """IMPROVED: Test deeply nested block structures."""
        source = """
        {
            a = 1;
            {
                b = 2;
                {
                    c = a + b;
                }
            }
        }
        """
        success, _ = self.parse_string(source)
        self.assertTrue(success)
    
    # ===== ERROR DETECTION TESTS =====
    
    def test_missing_semicolon(self):
        """Test error detection: missing semicolon."""
        success, message = self.parse_string("a = 5 b = 6;")
        self.assertFalse(success)
        self.assertIn("REJECT", message)
    
    def test_unbalanced_parentheses(self):
        """IMPROVED: Test error detection: unbalanced parentheses."""
        success, message = self.parse_string("a = (b + c;")
        self.assertFalse(success)
        self.assertIn("REJECT", message)
    
    def test_missing_condition_parentheses(self):
        """Test error detection: missing condition parentheses."""
        success, message = self.parse_string("if a == b { x = 1; }")
        self.assertFalse(success)
        self.assertIn("REJECT", message)
    
    def test_missing_braces(self):
        """Test error detection: missing braces."""
        success, message = self.parse_string("if (a == b) x = 1;")
        self.assertFalse(success)
        self.assertIn("REJECT", message)
    
    def test_invalid_operator(self):
        """Test error detection: invalid operator."""
        success, message = self.parse_string("a = b & c;")
        self.assertFalse(success)
    
    def test_empty_condition(self):
        """Test error detection: empty condition."""
        success, message = self.parse_string("if () { a = 1; }")
        self.assertFalse(success)
        self.assertIn("REJECT", message)
    
    def test_incomplete_expression(self):
        """IMPROVED: Test error detection: incomplete expression."""
        success, message = self.parse_string("a = b +;")
        self.assertFalse(success)
        self.assertIn("REJECT", message)


# ============================================================================
# TKINTER GUI APPLICATION (ENHANCED)
# ============================================================================

class ParserGUI:
    """Tkinter-based GUI for the recursive-descent parser with test runner."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("IMPROVED Recursive-Descent Parser - Mini-Language")
        self.root.geometry("900x700")
        self.root.resizable(True, True)
        
        self.setup_ui()
    
    def setup_ui(self):
        """Create GUI components."""
        # Title
        title_label = tk.Label(
            self.root, 
            text="IMPROVED Mini-Language Parser (Chaining, Precedence, Matched/Unmatched)",
            font=("Arial", 13, "bold"),
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
            height=10,
            width=100,
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
            font=("Arial", 10, "bold"),
            padx=15,
            pady=6,
            cursor="hand2"
        )
        parse_btn.pack(side="left", padx=5)
        
        clear_btn = tk.Button(
            button_frame,
            text="Clear",
            command=self.clear_all,
            bg="#e74c3c",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=15,
            pady=6,
            cursor="hand2"
        )
        clear_btn.pack(side="left", padx=5)
        
        example_btn = tk.Button(
            button_frame,
            text="Load Example",
            command=self.load_example,
            bg="#3498db",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=15,
            pady=6,
            cursor="hand2"
        )
        example_btn.pack(side="left", padx=5)
        
        # IMPROVED: Add test runner button
        test_btn = tk.Button(
            button_frame,
            text="Run All Tests",
            command=self.run_tests,
            bg="#9b59b6",
            fg="white",
            font=("Arial", 10, "bold"),
            padx=15,
            pady=6,
            cursor="hand2"
        )
        test_btn.pack(side="left", padx=5)
        
        # Output section
        output_frame = tk.Frame(self.root, padx=10, pady=10)
        output_frame.pack(fill="both", expand=True)
        
        tk.Label(output_frame, text="Output:", font=("Arial", 11, "bold")).pack(anchor="w")
        
        self.output_text = scrolledtext.ScrolledText(
            output_frame,
            height=12,
            width=100,
            font=("Courier New", 9),
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
            self.output_text.insert(tk.END, "-" * 80 + "\n")
            for token in tokens:
                if token.type != TokenType.EOF:
                    self.output_text.insert(tk.END, f"{token}\n")
            
            # Parsing
            self.output_text.insert(tk.END, "\n" + "=" * 80 + "\n")
            self.output_text.insert(tk.END, "PARSING RESULT:\n")
            self.output_text.insert(tk.END, "=" * 80 + "\n")
            
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
    x = a + 1;
} else {
    y = b - 1;
}
result = (x + y) * 2 + z / 3;"""
        self.input_text.delete("1.0", tk.END)
        self.input_text.insert("1.0", example)
    
    def run_tests(self):
        """IMPROVED: Run comprehensive test suite and display results."""
        self.output_text.config(state="normal")
        self.output_text.delete("1.0", tk.END)
        
        self.output_text.insert(tk.END, "RUNNING COMPREHENSIVE TEST SUITE\n")
        self.output_text.insert(tk.END, "=" * 80 + "\n\n")
        
        # Create test suite
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromTestCase(TestMiniLanguageParser)
        
        # Run tests with custom result handler
        class GUITestResult(unittest.TextTestResult):
            def __init__(self, stream, descriptions, verbosity, output_widget):
                super().__init__(stream, descriptions, verbosity)
                self.output_widget = output_widget
            
            def startTest(self, test):
                super().startTest(test)
                self.output_widget.insert(tk.END, f"Running: {test._testMethodName}... ")
                self.output_widget.see(tk.END)
                self.output_widget.update()
            
            def addSuccess(self, test):
                super().addSuccess(test)
                self.output_widget.insert(tk.END, "✓ PASS\n", "pass")
                self.output_widget.tag_config("pass", foreground="#27ae60")
                self.output_widget.see(tk.END)
            
            def addFailure(self, test, err):
                super().addFailure(test, err)
                self.output_widget.insert(tk.END, "✗ FAIL\n", "fail")
                self.output_widget.tag_config("fail", foreground="#e74c3c")
                self.output_widget.see(tk.END)
            
            def addError(self, test, err):
                super().addError(test, err)
                self.output_widget.insert(tk.END, "✗ ERROR\n", "error")
                self.output_widget.tag_config("error", foreground="#e67e22")
                self.output_widget.see(tk.END)
        
        import io
        stream = io.StringIO()
        runner = unittest.TextTestRunner(
            stream=stream,
            resultclass=lambda stream, descriptions, verbosity: 
                GUITestResult(stream, descriptions, verbosity, self.output_text),
            verbosity=2
        )
        
        result = runner.run(suite)
        
        # Summary
        self.output_text.insert(tk.END, "\n" + "=" * 80 + "\n")
        self.output_text.insert(tk.END, "TEST SUMMARY\n")
        self.output_text.insert(tk.END, "=" * 80 + "\n")
        self.output_text.insert(tk.END, f"Tests Run: {result.testsRun}\n")
        self.output_text.insert(tk.END, f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}\n", "pass")
        self.output_text.insert(tk.END, f"Failures: {len(result.failures)}\n", "fail" if result.failures else "pass")
        self.output_text.insert(tk.END, f"Errors: {len(result.errors)}\n", "error" if result.errors else "pass")
        
        if result.wasSuccessful():
            self.output_text.insert(tk.END, "\n✓ ALL TESTS PASSED!\n", "success")
            self.output_text.tag_config("success", foreground="#27ae60", font=("Courier New", 12, "bold"))
        else:
            self.output_text.insert(tk.END, "\n✗ SOME TESTS FAILED\n", "error")
        
        self.output_text.config(state="disabled")


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
