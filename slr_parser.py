import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import io
from collections import defaultdict

class SLRParser:
    def __init__(self, grammar_rules):
        self.grammar = {}
        self.terminals = set()
        self.non_terminals = set()
        self.start_symbol = None
        self.parse_grammar(grammar_rules)
        
        self.first = self.compute_first()
        self.follow = self.compute_follow()
        
        self.states = []
        self.action = {}
        self.goto_table = {}
        self.build_slr_table()

    def parse_grammar(self, grammar_rules):
        for rule in grammar_rules:
            if not rule.strip() or '->' not in rule:
                continue
            try:
                left, right = [part.strip() for part in rule.split('->')]
                if not left or not right:
                    continue
                
                # Handle productions separated by |
                productions = [prod.strip() for prod in right.split('|')]
                productions = [prod.split() for prod in productions if prod]  # Split into symbols
                
                if not self.start_symbol:
                    self.start_symbol = left
                self.non_terminals.add(left)
                
                for prod in productions:
                    for symbol in prod:
                        if symbol.isupper():
                            self.non_terminals.add(symbol)
                        elif symbol in {'id', 'num'} or symbol in {'+', '-', '*', '/', '=', '(', ')'} or symbol.startswith("'") or symbol.startswith('"'):
                            # Handle quoted terminals, special tokens, and operators
                            self.terminals.add(symbol.strip("'\""))
                        else:
                            self.terminals.add(symbol)
                            
                    if left in self.grammar:
                        self.grammar[left].append(prod)
                    else:
                        self.grammar[left] = [prod]
            except Exception as e:
                print(f"Error parsing rule '{rule}': {str(e)}")
                continue
        
        if not self.grammar:
            print("No valid grammar rules were parsed. Please check your input format.")
            return
            
        self.terminals.add('$')
        
        # Add augmented grammar rule S' -> S
        if self.start_symbol:
            augmented_start = "S'"
            self.grammar[augmented_start] = [[self.start_symbol]]
            self.non_terminals.add(augmented_start)
            self.start_symbol = augmented_start

    def compute_first(self):
        first = {nt: set() for nt in self.non_terminals}
        changed = True
        
        while changed:
            changed = False
            for nt in self.non_terminals:
                for prod in self.grammar[nt]:
                    if not prod or prod[0] in self.terminals:
                        if prod and prod[0] not in first[nt]:
                            first[nt].add(prod[0])
                            changed = True
                    elif prod[0] in self.non_terminals:
                        old_size = len(first[nt])
                        first[nt].update(first[prod[0]])
                        if len(first[nt]) > old_size:
                            changed = True
        return first

    def compute_follow(self):
        follow = {nt: set() for nt in self.non_terminals}
        follow[self.start_symbol].add('$')
        changed = True
        
        while changed:
            changed = False
            for nt in self.non_terminals:
                for A in self.non_terminals:
                    for prod in self.grammar[A]:
                        for i, symbol in enumerate(prod):
                            if symbol == nt:
                                if i + 1 < len(prod):
                                    next_symbol = prod[i + 1]
                                    if next_symbol in self.terminals:
                                        if next_symbol not in follow[nt]:
                                            follow[nt].add(next_symbol)
                                            changed = True
                                    else:
                                        old_size = len(follow[nt])
                                        follow[nt].update(self.first[next_symbol])
                                        if len(follow[nt]) > old_size:
                                            changed = True
                                elif A != nt:
                                    old_size = len(follow[nt])
                                    follow[nt].update(follow[A])
                                    if len(follow[nt]) > old_size:
                                        changed = True
        return follow

    def closure(self, items):
        closure_set = set(items)
        while True:
            new_items = set()
            for item in closure_set:
                A, prod, dot = item
                if dot < len(prod) and prod[dot] in self.non_terminals:
                    for production in self.grammar[prod[dot]]:
                        new_item = (prod[dot], tuple(production), 0)
                        if new_item not in closure_set:
                            new_items.add(new_item)
            if not new_items:
                break
            closure_set.update(new_items)
        return frozenset(closure_set)

    def goto(self, state, symbol):
        items = set()
        for item in state:
            A, prod, dot = item
            if dot < len(prod) and prod[dot] == symbol:
                items.add((A, prod, dot + 1))
        return self.closure(items)

    def build_slr_table(self):
        initial_items = self.closure({('S\'', (self.start_symbol,), 0)})
        self.states.append(initial_items)
        
        state_queue = [initial_items]
        state_map = {initial_items: 0}
        i = 0
        
        while i < len(state_queue):
            state = state_queue[i]
            state_num = state_map[state]
            
            symbols = self.terminals | self.non_terminals
            for symbol in symbols:
                next_state = self.goto(state, symbol)
                if next_state and next_state not in state_map:
                    state_queue.append(next_state)
                    state_map[next_state] = len(state_map)
                
                if symbol in self.terminals:
                    if next_state:
                        self.action[(state_num, symbol)] = f's{state_map[next_state]}'
                    for item in state:
                        A, prod, dot = item
                        if dot == len(prod) and A != 'S\'' and symbol in self.follow[A]:
                            self.action[(state_num, symbol)] = (f'r{A}', prod)
                        elif dot == len(prod) and A == 'S\'' and symbol == '$':
                            self.action[(state_num, symbol)] = 'acc'
                elif symbol in self.non_terminals and next_state:
                    self.goto_table[(state_num, symbol)] = state_map[next_state]
            i += 1

    def tokenize(self, input_string):
        tokens = []
        i = 0
        
        # Define all possible tokens
        operators = {'+', '-', '*', '/', '=', '>', '<', '(', ')'}
        keywords = {'if', 'then', 'else'}
        
        while i < len(input_string):
            char = input_string[i]
            
            # Skip whitespace
            if char.isspace():
                i += 1
                continue
            
            # Handle operators
            if char in operators:
                tokens.append(char)
                i += 1
                continue
            
            # Handle identifiers and keywords
            if char.isalpha() or char == '_':
                identifier = char
                i += 1
                while i < len(input_string) and (input_string[i].isalnum() or input_string[i] == '_'):
                    identifier += input_string[i]
                    i += 1
                if identifier in keywords:
                    tokens.append(identifier)
                else:
                    tokens.append('id')
                continue
            
            # Handle numbers
            if char.isdigit():
                number = char
                i += 1
                while i < len(input_string) and (input_string[i].isdigit() or input_string[i] == '.'):
                    number += input_string[i]
                    i += 1
                tokens.append('num')
                continue
            
            # Handle invalid characters
            print(f"Invalid character in input: {char}")
            return []
        
        return tokens

    def parse(self, input_string):
        tokens = self.tokenize(input_string)
        tokens.append('$')
        stack = [0]  # State stack
        symbols = []  # Symbol stack
        steps = []
        
        i = 0
        while i < len(tokens):
            state = stack[-1]
            symbol = tokens[i]
            
            # Create detailed step information
            current_step = {
                'stack_state': stack.copy(),
                'stack_symbols': symbols.copy(),
                'remaining_input': ' '.join(tokens[i:]),
                'action': '',
                'step_description': ''
            }
            
            if (state, symbol) not in self.action:
                current_step['action'] = 'Error'
                current_step['step_description'] = f"No action defined for state {state} and symbol {symbol}"
                steps.append(current_step)
                return False, steps, f"Syntax Error: Invalid token {symbol} in state {state}"
            
            action = self.action[(state, symbol)]
            
            # Handle different types of actions
            if isinstance(action, str):
                if action.startswith('s'):  # Shift
                    next_state = int(action[1:])
                    current_step['action'] = f"Shift {symbol}"
                    current_step['step_description'] = f"Shift {symbol} and go to state {next_state}"
                    stack.append(next_state)
                    symbols.append(symbol)
                    i += 1
                elif action == 'acc':  # Accept
                    current_step['action'] = 'Accept'
                    current_step['step_description'] = 'Input successfully parsed!'
                    steps.append(current_step)
                    return True, steps, "Input Accepted"
                else:  # Error
                    current_step['action'] = 'Error'
                    current_step['step_description'] = f"Invalid action {action}"
                    steps.append(current_step)
                    return False, steps, "Syntax Error: Invalid action"
            elif isinstance(action, tuple):  # Reduce
                action_str, prod = action
                A = action_str[1:]  # Get the non-terminal
                prod_len = len(prod)
                
                if len(symbols) < prod_len:
                    current_step['action'] = 'Error'
                    current_step['step_description'] = "Not enough symbols to reduce"
                    steps.append(current_step)
                    return False, steps, "Syntax Error: Not enough symbols to reduce"
                
                # Pop the required number of symbols and states
                popped_symbols = symbols[-prod_len:] if prod_len > 0 else []
                current_step['action'] = f"Reduce {A} → {' '.join(prod)}"
                current_step['step_description'] = f"Reduce {prod_len} symbols to {A}"
                
                for _ in range(prod_len):
                    stack.pop()
                    symbols.pop()
                
                # GOTO
                state = stack[-1]
                if (state, A) not in self.goto_table:
                    current_step['action'] = 'Error'
                    current_step['step_description'] = f"No GOTO action for state {state} and non-terminal {A}"
                    steps.append(current_step)
                    return False, steps, f"Syntax Error: Invalid GOTO transition for {A} in state {state}"
                
                next_state = self.goto_table[(state, A)]
                stack.append(next_state)
                symbols.append(A)
            
            steps.append(current_step)
        
        return False, steps, "Syntax Error: Unexpected end of input"

    def visualize_parse(self, steps):
        G = nx.DiGraph()
        
        # Create nodes for each step with detailed information
        for i, step in enumerate(steps):
            # Format the information for display
            stack_info = f"States: {' '.join(map(str, step['stack_state']))}\n"
            stack_info += f"Symbols: {' '.join(step['stack_symbols'])}\n"
            stack_info += f"Input: {step['remaining_input']}\n"
            stack_info += f"Action: {step['action']}\n"
            stack_info += f"Step {i+1}: {step['step_description']}"
            
            # Add node with formatted information
            G.add_node(i, label=stack_info)
            
            # Add edge from previous step
            if i > 0:
                G.add_edge(i-1, i)
        
        plt.figure(figsize=(15, len(steps)))  # Adjust figure size based on number of steps
        
        # Use hierarchical layout for better visualization of steps
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos,
                             node_size=5000,
                             node_color='lightblue',
                             alpha=0.9,
                             node_shape='o',
                             linewidths=2,
                             edgecolors='darkblue')
        
        # Draw edges
        nx.draw_networkx_edges(G, pos,
                             edge_color='gray',
                             arrows=True,
                             arrowsize=20,
                             arrowstyle='->',
                             width=2)
        
        # Draw labels
        nx.draw_networkx_labels(G, pos,
                              labels=nx.get_node_attributes(G, 'label'),
                              font_size=10,
                              font_weight='bold',
                              bbox=dict(facecolor='white',
                                      edgecolor='gray',
                                      alpha=0.9,
                                      pad=0.5))
        
        plt.title("Parsing Process Visualization\nStep-by-step parsing actions",
                 fontsize=16, fontweight='bold', pad=20)
        
        plt.axis('off')
        plt.tight_layout()
        
        # Convert plot to image
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=300, bbox_inches='tight',
                   facecolor='white', edgecolor='none')
        buf.seek(0)
        plt.close()
        return buf

    def get_tables(self):
        states = range(len(self.states))
        df_action = pd.DataFrame(index=states, columns=list(self.terminals))
        df_goto = pd.DataFrame(index=states, columns=list(self.non_terminals))
        
        for (state, symbol), action in self.action.items():
            if isinstance(action, tuple):
                action_str, prod = action
                df_action.at[state, symbol] = f"{action_str}->{''.join(prod)}"
            else:
                df_action.at[state, symbol] = str(action)
        for (state, symbol), goto_state in self.goto_table.items():
            df_goto.at[state, symbol] = str(goto_state)
        
        return df_action, df_goto

    def visualize_dfa(self):
        plt.figure(figsize=(24, 20))
        
        G = nx.DiGraph()
        
        # Add nodes for each state with detailed information
        for i, state in enumerate(self.states):
            # Group items by non-terminal for better organization
            items_by_nt = {}
            for A, prod, dot in sorted(state, key=lambda x: (x[0], ' '.join(x[1]))):
                if A not in items_by_nt:
                    items_by_nt[A] = []
                # Format production with bullet and different colors
                prod_str = ' '.join(prod[:dot]) + '•' + ' '.join(prod[dot:])
                items_by_nt[A].append(prod_str)
            
            # Create a formatted string with grouped items
            items_str = [f"State {i}"]
            items_str.append("─" * 20)  # Separator line using a simple character
            for nt in sorted(items_by_nt.keys()):
                prods = items_by_nt[nt]
                items_str.append(f"{nt} →")
                for prod in prods:
                    items_str.append(f"    {prod}")
            
            G.add_node(i, label='\n'.join(items_str))
        
        # Add edges with detailed transition information
        for (state_num, symbol), next_state in self.goto_table.items():
            G.add_edge(state_num, next_state, 
                      label=f"GOTO on {symbol}",
                      color='blue',
                      style='solid')
        
        for (state_num, symbol), action in self.action.items():
            if isinstance(action, str) and action.startswith('s'):
                next_state = int(action[1:])
                G.add_edge(state_num, next_state,
                          label=f"Shift {symbol}",
                          color='green',
                          style='solid')
        
        # Use a more spread out layout
        pos = nx.spring_layout(G, k=3.0, iterations=50, seed=42)
        
        # Draw nodes with enhanced styling
        nx.draw_networkx_nodes(G, pos,
                             node_size=10000,
                             node_color='lightblue',
                             alpha=0.9,
                             node_shape='s',
                             linewidths=3,
                             edgecolors='darkblue')
        
        # Draw edges with different colors for different types
        edge_colors = [G.edges[edge].get('color', 'gray') for edge in G.edges()]
        edge_styles = [G.edges[edge].get('style', 'solid') for edge in G.edges()]
        
        nx.draw_networkx_edges(G, pos,
                             arrowsize=25,
                             width=2.5,
                             edge_color=edge_colors,
                             style=edge_styles,
                             arrowstyle='->',
                             connectionstyle='arc3,rad=0.2')
        
        # Draw edge labels with enhanced styling
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos,
                                   edge_labels=edge_labels,
                                   font_size=10,
                                   font_weight='bold',
                                   bbox=dict(facecolor='white',
                                           alpha=0.9,
                                           pad=0.5))
        
        # Draw node labels with enhanced styling
        node_labels = nx.get_node_attributes(G, 'label')
        nx.draw_networkx_labels(G, pos,
                              labels=node_labels,
                              font_size=9,
                              font_weight='bold',
                              bbox=dict(facecolor='white',
                                      alpha=0.9,
                                      pad=0.5))
        
        plt.title("Deterministic Finite Automaton (DFA) for SLR Parser\n"
                 "Blue edges: GOTO transitions | Green edges: Shift actions",
                 fontsize=20, fontweight='bold', pad=20)
        
        # Add a legend
        legend_elements = [
            plt.Line2D([0], [0], color='blue', label='GOTO Transition'),
            plt.Line2D([0], [0], color='green', label='Shift Action'),
        ]
        plt.legend(handles=legend_elements, loc='upper right', fontsize=12)
        
        plt.margins(0.2)
        plt.axis('off')
        
        # Save with high quality
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none',
                   pad_inches=0.8)
        buf.seek(0)
        plt.close()
        return buf 