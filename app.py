import streamlit as st
import pandas as pd
import base64
import io

# Import the three parser implementations
from slr_parser import SLRParser
from clr_parser import (
    parse_grammar_clr, compute_first_sets_clr, compute_follow_sets_clr,
    construct_clr1_table_clr, generate_grammar_graph_clr, 
    display_parsing_table_clr, tokenize_clr, parse_string_clr
)
from lalr_parser import LALRParser

def main():
    st.title("Parser Visualization Tool")
    
    # Sidebar menu for parser selection
    parser_type = st.sidebar.selectbox(
        "Select Parser Type",
        ["SLR Parser", "CLR(1) Parser", "LALR Parser"]
    )

    # Common grammar input
    st.header("Enter Grammar Rules")
    st.write("Enter each grammar rule in the format 'NonTerminal -> production | production'. Press Enter after each rule.")
    
    # Example grammars
    st.write("Choose an example grammar or enter your own:")
    example_grammars = {
        "Arithmetic Expression": """E -> E + T | E - T | T
T -> T * F | T / F | F
F -> ( E ) | id | num""",
        
        "Simple Assignment": """S -> id = E
E -> E + T | E - T | T
T -> T * F | T / F | F
F -> ( E ) | id | num""",
        
        "If-Then Statement": """S -> if C then S | id = E
C -> E > E | E < E | E = E
E -> E + T | T
T -> T * F | F
F -> ( E ) | id | num""",
        
        "Custom Grammar": ""  # Empty option for custom input
    }
    
    selected_grammar = st.selectbox("Select Grammar Example", list(example_grammars.keys()))
    
    if selected_grammar == "Custom Grammar":
        st.write("Enter your custom grammar rules below:")
        grammar_input = st.text_area(
            "Grammar Rules (one per line):",
            value="",
            height=150,
            help="Enter each production rule on a new line. Use '|' to separate multiple productions."
        )
    else:
        grammar_input = st.text_area(
            "Grammar Rules (one per line):",
            value=example_grammars[selected_grammar],
            height=150,
            help="Enter each production rule on a new line. Use '|' to separate multiple productions."
        )

    # Add grammar validation and helpful messages
    st.info("""Grammar Writing Tips:
    1. Use uppercase letters for non-terminals (E, T, F, S, etc.)
    2. Use lowercase letters or symbols for terminals (id, num, +, -, etc.)
    3. Each rule must have a non-terminal on the left side
    4. Use '|' to separate multiple productions
    5. Use spaces between symbols in productions
    """)

    # Validate and clean grammar rules with better error messages
    grammar_rules = []
    has_errors = False
    for rule in grammar_input.strip().split('\n'):
        rule = rule.strip()
        if rule:
            if '->' not in rule:
                st.warning(f"⚠ Invalid rule (missing ->): {rule}")
                has_errors = True
                continue
            
            left, right = [part.strip() for part in rule.split('->')]
            if not left:
                st.warning(f"⚠ Invalid rule (empty left side): {rule}")
                has_errors = True
                continue
            
            if not right:
                st.warning(f"⚠ Invalid rule (empty right side): {rule}")
                has_errors = True
                continue
            
            if not left[0].isupper():
                st.warning(f"⚠ Left side should be a non-terminal (uppercase): {rule}")
                has_errors = True
                continue
                
            grammar_rules.append(rule)

    if has_errors:
        st.error("Please fix the grammar errors before continuing.")
        return

    if not grammar_rules:
        st.error("Please enter at least one valid grammar rule.")
        return

    # SLR Parser
    if parser_type == "SLR Parser":
        try:
            parser = SLRParser(grammar_rules)
            if not parser.grammar:
                st.error("Failed to parse grammar rules. Please check your input format.")
                return
        except Exception as e:
            st.error(f"Error parsing grammar: {str(e)}")
            return

        try:
            st.header("DFA for Grammar")
            dfa_buf = parser.visualize_dfa()
            st.image(dfa_buf, caption="DFA for SLR Parser")

            st.header("SLR Parsing Tables")
            action_table, goto_table = parser.get_tables()
            
            st.subheader("Action Table")
            st.dataframe(action_table)
            
            st.subheader("Goto Table")
            st.dataframe(goto_table)

            st.header("Parse a String")
            input_string = st.text_input("Enter a string to parse:", value="id+id*id")
            
            if st.button("Parse"):
                if not input_string:
                    st.error("Please enter a string to parse.")
                    return
                
                accepted, steps, result = parser.parse(input_string)
                
                st.subheader(f"Parsing '{input_string}':")
                st.write(f"*Result:* {result}")
                
                st.subheader("Parsing Steps")
                steps_df = pd.DataFrame(steps)
                st.dataframe(steps_df)

                st.subheader("Parsing Process Visualization")
                if steps:
                    buf = parser.visualize_parse(steps)
                    st.image(buf, caption="SLR Parsing Process")
                else:
                    st.write("No parsing steps to visualize.")
        except Exception as e:
            st.error(f"Error visualizing parser: {str(e)}")
            return

    # CLR(1) Parser
    elif parser_type == "CLR(1) Parser":
        grammar, terminals, non_terminals = parse_grammar_clr(grammar_input)
        if not grammar:
            st.error("Grammar is empty or invalid. Please enter a valid grammar.")
            return

        first_sets = compute_first_sets_clr(grammar, terminals, non_terminals)
        follow_sets = compute_follow_sets_clr(grammar, terminals, non_terminals, first_sets)

        st.write("### FIRST Sets")
        first_data = {}
        for symbol in sorted(terminals | non_terminals):
            first_data[symbol] = ', '.join(sorted(first_sets[symbol]))
        first_df = pd.DataFrame([first_data])
        st.dataframe(first_df)

        st.write("### FOLLOW Sets")
        follow_data = {}
        for symbol in sorted(non_terminals):
            follow_data[symbol] = ', '.join(sorted(follow_sets[symbol]))
        follow_df = pd.DataFrame([follow_data])
        st.dataframe(follow_df)

        action, goto_table, states, state_map = construct_clr1_table_clr(
            grammar, terminals, non_terminals, first_sets, follow_sets)

        st.write("### CLR(1) Parsing Table")
        parsing_table = display_parsing_table_clr(
            action, goto_table, terminals, non_terminals, len(states))
        st.dataframe(parsing_table)

        st.write("### Grammar Visualization")
        grammar_graph = generate_grammar_graph_clr(grammar)
        st.graphviz_chart(grammar_graph)

        st.write("### Canonical Collection of CLR(1) Items")
        num_states = len(states)
        tab_groups = [states[i:i+5] for i in range(0, num_states, 5)]
        tab_labels = [f"States {i}-{min(i+4, num_states-1)}" for i in range(0, num_states, 5)]
        tabs = st.tabs(tab_labels)
        for tab_idx, tab in enumerate(tabs):
            with tab:
                for i in range(tab_idx * 5, min((tab_idx + 1) * 5, num_states)):
                    with st.expander(f"State {i}"):
                        items_data = []
                        for item in sorted(states[i], key=str):
                            body_str = ' '.join(item.body) if item.body else 'ε'
                            dot_pos = '•' + ' ' + body_str if item.dot == 0 else \
                                      body_str[:item.dot*2] + ' •' + body_str[item.dot*2:] if item.dot < len(item.body) else \
                                      body_str + ' •'
                            items_data.append({
                                "Production": f"{item.head} → {dot_pos}",
                                "Lookahead": item.lookahead
                            })
                        st.table(pd.DataFrame(items_data))
        
        # Add parsing functionality for CLR parser
        st.header("Parse a String")
        input_string = st.text_input("Enter a string to parse (CLR):", value="id+id*id")
        
        if st.button("Parse with CLR"):
            if not input_string:
                st.error("Please enter a string to parse.")
                return
            
            accepted, steps, result = parse_string_clr(input_string, action, goto_table)
            
            st.subheader(f"Parsing '{input_string}':")
            st.write(f"*Result:* {result}")
            
            st.subheader("Parsing Steps")
            steps_df = pd.DataFrame(steps)
            st.dataframe(steps_df)

    # LALR Parser
    elif parser_type == "LALR Parser":
        try:
            parser = LALRParser(grammar_rules)
        except Exception as e:
            st.error(f"Error parsing grammar: {str(e)}")
            return

        st.write("### FIRST Sets")
        first_data = {symbol: ', '.join(sorted(parser.first[symbol])) for symbol in sorted(parser.terminals | parser.non_terminals)}
        st.dataframe(pd.DataFrame([first_data]))

        st.write("### FOLLOW Sets")
        follow_data = {symbol: ', '.join(sorted(parser.follow[symbol])) for symbol in sorted(parser.non_terminals)}
        st.dataframe(pd.DataFrame([follow_data]))

        st.write("### LALR Parsing Table")
        parsing_table = parser.get_tables()
        st.dataframe(parsing_table)

        st.write("### Grammar Visualization")
        grammar_graph = generate_grammar_graph_clr(parser.grammar)
        st.graphviz_chart(grammar_graph)

        st.write("### DFA for LALR Parser")
        dfa_graph = parser.visualize_dfa()
        st.graphviz_chart(dfa_graph)

        st.write("### Canonical Collection of LALR(1) Items")
        num_states = len(parser.states)
        tab_groups = [parser.states[i:i+5] for i in range(0, num_states, 5)]
        tab_labels = [f"States {i}-{min(i+4, num_states-1)}" for i in range(0, num_states, 5)]
        tabs = st.tabs(tab_labels)
        for tab_idx, tab in enumerate(tabs):
            with tab:
                for i in range(tab_idx * 5, min((tab_idx + 1) * 5, num_states)):
                    with st.expander(f"State {i}"):
                        items_data = []
                        for item in sorted(parser.states[i], key=str):
                            body_str = ' '.join(item.body) if item.body else 'ε'
                            dot_pos = '•' + ' ' + body_str if item.dot == 0 else \
                                      body_str[:item.dot*2] + ' •' + body_str[item.dot*2:] if item.dot < len(item.body) else \
                                      body_str + ' •'
                            items_data.append({
                                "Production": f"{item.head} → {dot_pos}",
                                "Lookahead": item.lookahead
                            })
                        st.table(pd.DataFrame(items_data))
        
        # Add parsing functionality for LALR parser
        st.header("Parse a String")
        input_string = st.text_input("Enter a string to parse (LALR):", value="id+id*id")
        
        if st.button("Parse with LALR"):
            if not input_string:
                st.error("Please enter a string to parse.")
                return
            
            accepted, steps, result = parser.parse(input_string)
            
            st.subheader(f"Parsing '{input_string}':")
            st.write(f"*Result:* {result}")
            
            st.subheader("Parsing Steps")
            steps_df = pd.DataFrame(steps)
            st.dataframe(steps_df)

if __name__ == "__main__":
    main() 