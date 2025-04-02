import streamlit as st
import graphviz
import time
from Lalr.LALR import LALR
import pandas as pd

st.set_page_config(page_title="LALR Parser Visualization", layout="wide")

st.title("LALR Parser Visualization")

# Sidebar for grammar input
with st.sidebar:
    st.header("Grammar Definition")
    
    # Example grammar selection
    example_grammars = {
        "Simple Expression": {
            "productions": [
                ("E", ["E", "+", "T"]),
                ("E", ["T"]),
                ("T", ["T", "*", "F"]),
                ("T", ["F"]),
                ("F", ["(", "E", ")"]),
                ("F", ["id"])
            ],
            "start": "E"
        },
        "Simple If-Then-Else": {
            "productions": [
                ("S", ["if", "C", "then", "S", "else", "S"]),
                ("S", ["if", "C", "then", "S"]),
                ("S", ["a"]),
                ("C", ["b"])
            ],
            "start": "S"
        },
        "Empty String": {
            "productions": [
                ("S", ["A", "B"]),
                ("A", ["a", "A"]),
                ("A", []),  # epsilon
                ("B", ["b"])
            ],
            "start": "S"
        }
    }
    
    selected_grammar = st.selectbox(
        "Select Example Grammar",
        list(example_grammars.keys()) + ["Custom Grammar"]
    )
    
    # Custom grammar input
    if selected_grammar == "Custom Grammar":
        st.subheader("Define Productions")
        st.markdown("Format: `LHS -> RHS`, one per line (use 'ε' for empty string)")
        custom_grammar = st.text_area(
            "Grammar Productions",
            "E -> E + T\nE -> T\nT -> T * F\nT -> F\nF -> ( E )\nF -> id",
            height=200
        )
        
        start_symbol = st.text_input("Start Symbol", "E")
        
        # Parse custom grammar
        productions = []
        for line in custom_grammar.strip().split("\n"):
            if "->" in line:
                lhs, rhs = line.split("->")
                lhs = lhs.strip()
                rhs_symbols = [sym.strip() for sym in rhs.split()]
                if rhs_symbols == ["ε"]:
                    productions.append((lhs, []))
                else:
                    productions.append((lhs, rhs_symbols))
    else:
        # Use selected example grammar
        productions = example_grammars[selected_grammar]["productions"]
        start_symbol = example_grammars[selected_grammar]["start"]
    
    # Input string to parse
    st.subheader("Input String")
    input_string = st.text_input("Enter string to parse", "id+id*id")

# Function to build and analyze the parser
def build_parser():
    parser = LALR()
    
    # Add productions
    for lhs, rhs in productions:
        parser.add_production(lhs, rhs)
    
    # Set start symbol
    parser.set_start_symbol(start_symbol)
    
    # Add terminals
    parser.add_terminals_from_grammar()
    
    # Compute FIRST and FOLLOW sets
    parser.compute_first_sets()
    parser.compute_follow_sets()
    
    # Build parsing table
    parser.build_parsing_table()
    
    return parser

# Function to tokenize the input string
def tokenize_input(input_str, terminals):
    # Sort terminals by length (longest first) to avoid partial matches
    sorted_terminals = sorted(terminals, key=len, reverse=True)
    
    tokens = []
    i = 0
    while i < len(input_str):
        # Try to match a terminal
        matched = False
        for terminal in sorted_terminals:
            if terminal != "ε" and input_str[i:].startswith(terminal):
                tokens.append(terminal)
                i += len(terminal)
                matched = True
                break
        
        # If no terminal matches, check if it's a space
        if not matched:
            if input_str[i].isspace():
                i += 1
            else:
                # If it's not a space and not a terminal, add it as a single character
                tokens.append(input_str[i])
                i += 1
    
    return tokens

# Function to visualize the state machine
def visualize_states(parser):
    graph = graphviz.Digraph()
    
    # Add states as nodes
    for i, state in enumerate(parser.states):
        state_items = "<br/>".join([str(item) for item in state])
        graph.node(str(i), f"<b>State {i}</b><br/>{state_items}", shape="box", style="rounded,filled", fillcolor="lightblue", fontname="Arial", fontsize="10", margin="0.2", labelloc="t", html="true")
    
    # Add transitions as edges
    for src, dst, symbol in parser.transitions:
        graph.edge(str(src), str(dst), label=symbol, fontname="Arial", fontsize="10")
    
    return graph

# Function to display the parsing table
def display_parsing_table(parser):
    # Create a dataframe for the ACTION table
    action_rows = []
    for state in range(len(parser.states)):
        row = {}
        for terminal in sorted(parser.terminals) + ["$"]:
            if state in parser.actions and terminal in parser.actions[state]:
                action, value = parser.actions[state][terminal]
                row[terminal] = f"{action} {value}" if value is not None else action
            else:
                row[terminal] = ""
        action_rows.append(row)
    
    action_df = pd.DataFrame(action_rows)
    
    # Create a dataframe for the GOTO table
    goto_rows = []
    for state in range(len(parser.states)):
        row = {}
        for non_terminal in sorted(parser.non_terminals):
            if state in parser.goto and non_terminal in parser.goto[state]:
                # Convert to string to avoid type conversion issues
                row[non_terminal] = str(parser.goto[state][non_terminal])
            else:
                row[non_terminal] = ""
        goto_rows.append(row)
    
    goto_df = pd.DataFrame(goto_rows)
    
    return action_df, goto_df

# Function to visualize parsing steps
def visualize_parsing(parse_steps, parser):
    for step in parse_steps:
        stack = step['stack']
        input_tokens = step['input']
        action = step['action']
        
        # Create a visualization of the stack and input
        stack_display = " ".join(map(str, stack))
        input_display = " ".join(map(str, input_tokens))
        
        st.markdown(f"**Stack**: {stack_display}")
        st.markdown(f"**Input**: {input_display}")
        st.markdown(f"**Action**: {action}")
        
        # Add a divider
        st.markdown("---")
        time.sleep(0.5)  # Slight delay for animation effect

# Main layout
col1, col2 = st.columns([3, 2])

# Build parser and run parser
if st.button("Build Parser and Parse Input"):
    with st.spinner("Building parser and analyzing grammar..."):
        parser = build_parser()
        
        # Display FIRST and FOLLOW sets
        with col2:
            st.header("Grammar Analysis")
            
            # First sets
            st.subheader("FIRST Sets")
            first_sets = {}
            for symbol in sorted(parser.non_terminals):
                if symbol in parser.first_sets:
                    first_sets[symbol] = ", ".join(sorted(parser.first_sets[symbol]))
                else:
                    first_sets[symbol] = ""
            
            st.table(pd.DataFrame.from_dict(first_sets, orient='index', columns=["FIRST"]))
            
            # Follow sets
            st.subheader("FOLLOW Sets")
            follow_sets = {}
            for symbol in sorted(parser.non_terminals):
                if symbol in parser.follow_sets:
                    follow_sets[symbol] = ", ".join(sorted(parser.follow_sets[symbol]))
                else:
                    follow_sets[symbol] = ""
            
            st.table(pd.DataFrame.from_dict(follow_sets, orient='index', columns=["FOLLOW"]))
            
            # Display parsing table
            st.subheader("Parsing Tables")
            action_df, goto_df = display_parsing_table(parser)
            
            st.markdown("**ACTION Table**")
            st.dataframe(action_df)
            
            st.markdown("**GOTO Table**")
            st.dataframe(goto_df)
        
        # Display state machine
        with col1:
            st.header("State Machine")
            graph = visualize_states(parser)
            st.graphviz_chart(graph)
            
            # Parse input string
            st.header("Parsing Animation")
            
            # Tokenize the input string
            tokens = tokenize_input(input_string, parser.terminals)
            st.write(f"Tokenized input: {' '.join(tokens)}")

            success, message, parse_steps = parser.parse(tokens)
            
            st.subheader(f"Result: {'Success' if success else 'Error'}")
            st.markdown(message)
            
            if parse_steps:
                st.subheader("Parsing Steps")
                for i, step in enumerate(parse_steps):
                    stack = step['stack']
                    input_tokens = step['input']
                    action = step['action']
                    
                    # Create a visualization of the stack and input
                    stack_display = " ".join(map(str, stack))
                    input_display = " ".join(map(str, input_tokens))
                    
                    st.markdown(f"**Step {i+1}**")
                    st.markdown(f"**Stack**: {stack_display}")
                    st.markdown(f"**Input**: {input_display}")
                    st.markdown(f"**Action**: {action}")
                    
                    # Add a divider
                    st.markdown("---")
                    
if __name__ == "__main__":
    st.sidebar.markdown("---")
    st.sidebar.markdown("### How to Use")
    st.sidebar.markdown("""
    1. Select a grammar or define your own
    2. Enter an input string to parse
    3. Click 'Build Parser and Parse Input'
    4. Explore the state machine, FIRST/FOLLOW sets, and parsing animation
    """) 