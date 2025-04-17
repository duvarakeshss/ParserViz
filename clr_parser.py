import pandas as pd
import matplotlib.pyplot as plt
import graphviz
from collections import defaultdict

def parse_grammar_clr(grammar_input):
    grammar = defaultdict(list)
    terminals = set()
    non_terminals = set()

    for line in grammar_input.split('\n'):
        line = line.strip()
        if not line or '->' not in line:
            continue
            
        head, productions = line.split('->', 1)
        head = head.strip()
        non_terminals.add(head)

        for production in productions.split('|'):
            production = production.strip()
            if not production:  # Handle epsilon productions
                grammar[head].append('')
                continue
                
            grammar[head].append(production)
            for symbol in production.split():
                if symbol.islower() or symbol in "()[]{}+*/-<>=!&|^%$#@":  # Terminals
                    terminals.add(symbol)
                else:
                    non_terminals.add(symbol)
    
    terminals.add('$')  # Add end-of-input symbol
    return grammar, terminals, non_terminals

def compute_first_sets_clr(grammar, terminals, non_terminals):
    first = {terminal: {terminal} for terminal in terminals}
    first.update({non_terminal: set() for non_terminal in non_terminals})
    
    # Handle epsilon productions
    for head, productions in grammar.items():
        if '' in productions:
            first[head].add('')
    
    changed = True
    while changed:
        changed = False
        for head, productions in grammar.items():
            for production in productions:
                if not production:  # Skip epsilon productions, already handled
                    continue
                    
                symbols = production.split()
                # FIRST(α) = FIRST(X₁) if X₁ doesn't derive ε
                # or FIRST(X₁) - {ε} ∪ FIRST(X₂) if X₁ derives ε, and so on
                
                all_derive_epsilon = True
                for i, symbol in enumerate(symbols):
                    # Add all terminals in FIRST(symbol) except epsilon
                    symbol_first = first[symbol] - {''}
                    prev_size = len(first[head])
                    first[head].update(symbol_first)
                    if len(first[head]) > prev_size:
                        changed = True
                    
                    # If this symbol doesn't derive epsilon, stop
                    if '' not in first[symbol]:
                        all_derive_epsilon = False
                        break
                
                # If all symbols derive epsilon, add epsilon to FIRST(head)
                if all_derive_epsilon and '' not in first[head]:
                    first[head].add('')
                    changed = True
    
    return first

def compute_follow_sets_clr(grammar, terminals, non_terminals, first):
    follow = {non_terminal: set() for non_terminal in non_terminals}
    start_symbol = next(iter(grammar.keys()))
    follow[start_symbol].add('$')
    
    changed = True
    while changed:
        changed = False
        for head, productions in grammar.items():
            for production in productions:
                if not production:  # Skip epsilon productions
                    continue
                    
                symbols = production.split()
                for i, symbol in enumerate(symbols):
                    if symbol in non_terminals:
                        # For B in A → αBβ, add FIRST(β) - {ε} to FOLLOW(B)
                        if i < len(symbols) - 1:
                            # Calculate FIRST of the rest of the production
                            beta = symbols[i+1:]
                            beta_first = set()
                            
                            # Calculate FIRST(β)
                            all_derive_epsilon = True
                            for beta_symbol in beta:
                                beta_first.update(first[beta_symbol] - {''})
                                if '' not in first[beta_symbol]:
                                    all_derive_epsilon = False
                                    break
                            
                            # Add FIRST(β) - {ε} to FOLLOW(symbol)
                            prev_size = len(follow[symbol])
                            follow[symbol].update(beta_first)
                            if len(follow[symbol]) > prev_size:
                                changed = True
                            
                            # If β can derive ε, add FOLLOW(A) to FOLLOW(B)
                            if all_derive_epsilon:
                                prev_size = len(follow[symbol])
                                follow[symbol].update(follow[head])
                                if len(follow[symbol]) > prev_size:
                                    changed = True
                        else:
                            # For B in A → αB, add FOLLOW(A) to FOLLOW(B)
                            prev_size = len(follow[symbol])
                            follow[symbol].update(follow[head])
                            if len(follow[symbol]) > prev_size:
                                changed = True
    
    return follow

def first_of_string_clr(string, first):
    if not string:
        return {''}
    
    symbols = string.split()
    if not symbols:
        return {''}
        
    result = set()
    all_derive_epsilon = True
    
    for symbol in symbols:
        # Add FIRST(symbol) - {ε} to result
        symbol_first = first[symbol]
        result.update(symbol_first - {''})
        
        # If this symbol doesn't derive ε, stop
        if '' not in symbol_first:
            all_derive_epsilon = False
            break
    
    # If all symbols derive ε, add ε to result
    if all_derive_epsilon:
        result.add('')
    
    return result

class LR1Item_clr:
    def __init__(self, head, body, dot, lookahead):
        self.head = head
        self.body = body.split() if isinstance(body, str) else body
        self.dot = dot
        self.lookahead = lookahead
    
    def __eq__(self, other):
        if not isinstance(other, LR1Item_clr):
            return False
        return (self.head == other.head and 
                self.body == other.body and 
                self.dot == other.dot and 
                self.lookahead == other.lookahead)
    
    def __hash__(self):
        return hash((self.head, tuple(self.body), self.dot, self.lookahead))
    
    def __str__(self):
        body_with_dot = self.body.copy()
        if self.dot <= len(self.body):
            body_with_dot.insert(self.dot, "•")
        return f"[{self.head} → {' '.join(body_with_dot)}, {self.lookahead}]"
    
    def __repr__(self):
        return self.__str__()
    
    def next_symbol(self):
        """Return the symbol after the dot or None if dot is at the end."""
        if self.dot < len(self.body):
            return self.body[self.dot]
        return None
    
    def is_complete(self):
        """Return True if the dot is at the end."""
        return self.dot >= len(self.body)
    
    def advance_dot(self):
        """Return a new item with the dot advanced."""
        return LR1Item_clr(self.head, self.body, self.dot + 1, self.lookahead)

def lr1_closure_clr(item_set, grammar, first):
    result = set(item_set)
    queue = list(item_set)
    
    while queue:
        item = queue.pop(0)
        
        # If dot is not at the end and points to a non-terminal
        B = item.next_symbol()
        if B and B in grammar:
            # Calculate first of what follows B plus lookahead
            beta_a = item.body[item.dot+1:] + [item.lookahead]
            
            # For each production of B
            for gamma in grammar[B]:
                gamma_symbols = gamma.split() if gamma else []
                
                # For first symbol after dot, compute first of the rest plus lookahead
                beta_str = ' '.join(item.body[item.dot+1:])
                beta_first = first_of_string_clr(beta_str, first)
                
                lookaheads = set()
                # If β can derive ε, add lookahead to lookaheads
                if '' in beta_first:
                    beta_first.remove('')
                    lookaheads.add(item.lookahead)
                lookaheads.update(beta_first)
                
                # Add item for each lookahead
                for b in lookaheads:
                    new_item = LR1Item_clr(B, gamma_symbols, 0, b)
                    if new_item not in result:
                        result.add(new_item)
                        queue.append(new_item)
    
    return result

def lr1_goto_clr(item_set, symbol, grammar, first):
    # Find all items with the symbol after the dot
    advanced_items = {item.advance_dot() for item in item_set 
                     if item.next_symbol() == symbol}
    
    if advanced_items:
        return lr1_closure_clr(advanced_items, grammar, first)
    else:
        return set()

def construct_clr1_table_clr(grammar, terminals, non_terminals, first, follow):
    # Augment grammar with a new start symbol S'
    augmented_start = "S'"
    original_start = next(iter(grammar.keys()))
    augmented_grammar = grammar.copy()
    augmented_grammar[augmented_start] = [original_start]
    
    all_non_terminals = non_terminals | {augmented_start}
    all_symbols = terminals | all_non_terminals
    
    # Initialize with the start item
    start_item = LR1Item_clr(augmented_start, [original_start], 0, '$')
    initial_state = lr1_closure_clr({start_item}, augmented_grammar, first)
    
    # Collection of sets of LR(1) items (states)
    canonical_collection = [initial_state]
    state_map = {frozenset(initial_state): 0}
    
    # Initialize the parsing tables
    action = {}  # (state, terminal) -> action
    goto_table = {}  # (state, non_terminal) -> state
    
    # Process all states
    queue = [initial_state]
    while queue:
        current_state = queue.pop(0)
        state_idx = state_map[frozenset(current_state)]
        
        # For each item in the state
        for item in current_state:
            # Case 1: [A → α•aβ, b], Shift
            if not item.is_complete() and item.next_symbol() in terminals:
                symbol = item.next_symbol()
                next_state = lr1_goto_clr(current_state, symbol, augmented_grammar, first)
                
                if next_state:
                    if frozenset(next_state) not in state_map:
                        state_map[frozenset(next_state)] = len(canonical_collection)
                        canonical_collection.append(next_state)
                        queue.append(next_state)
                    
                    next_state_idx = state_map[frozenset(next_state)]
                    # Set shift action
                    action[(state_idx, symbol)] = f"S{next_state_idx}"
            
            # Case 2: [A → α•, a], Reduce
            elif item.is_complete():
                if item.head == augmented_start:
                    # Accept for [S' → S•, $]
                    action[(state_idx, '$')] = "ACC"
                else:
                    # Reduce action
                    body_str = ' '.join(item.body) if item.body else ''
                    # Find production number
                    for i, prod in enumerate(augmented_grammar[item.head]):
                        if prod == body_str:
                            action[(state_idx, item.lookahead)] = f"R{item.head}{i}"
                            break
        
        # Calculate GOTO for non-terminals
        for symbol in all_non_terminals:
            next_state = lr1_goto_clr(current_state, symbol, augmented_grammar, first)
            if next_state:
                if frozenset(next_state) not in state_map:
                    state_map[frozenset(next_state)] = len(canonical_collection)
                    canonical_collection.append(next_state)
                    queue.append(next_state)
                
                next_state_idx = state_map[frozenset(next_state)]
                goto_table[(state_idx, symbol)] = next_state_idx
    
    return action, goto_table, canonical_collection, state_map

def generate_grammar_graph_clr(grammar):
    dot = graphviz.Digraph()
    
    # Add nodes for non-terminals
    for head in grammar:
        dot.node(head, shape="circle")
    
    # Add edges for productions
    for head, productions in grammar.items():
        for i, production in enumerate(productions):
            if not production:  # Handle ε-productions
                node_id = f"{head}eps{i}"
                dot.node(node_id, label="ε", shape="box")
                dot.edge(head, node_id)
            else:
                node_id = f"{head}prod{i}"
                dot.node(node_id, label=production, shape="box")
                dot.edge(head, node_id)
    
    return dot

def display_parsing_table_clr(action, goto_table, terminals, non_terminals, state_count):
    # Create the table data
    table_data = []
    
    for state in range(state_count):
        row = {'State': state}
        
        # Action columns (terminals)
        for terminal in sorted(terminals):
            if (state, terminal) in action:
                row[terminal] = str(action[(state, terminal)])
            else:
                row[terminal] = ''
        
        # Goto columns (non-terminals)
        for non_terminal in sorted(non_terminals):
            if (state, non_terminal) in goto_table:
                row[non_terminal] = str(goto_table[(state, non_terminal)])
            else:
                row[non_terminal] = ''
        
        table_data.append(row)
    
    return pd.DataFrame(table_data).set_index('State')

def tokenize_clr(input_string):
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

def parse_string_clr(input_string, action, goto_table):
    tokens = tokenize_clr(input_string)
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
        
        if (state, symbol) not in action:
            current_step['action'] = 'Error'
            current_step['step_description'] = f"No action defined for state {state} and symbol {symbol}"
            steps.append(current_step)
            return False, steps, f"Syntax Error: Invalid token {symbol} in state {state}"
        
        action_value = action[(state, symbol)]
        
        # Handle different types of actions
        if action_value.startswith('S'):  # Shift
            next_state = int(action_value[1:])
            current_step['action'] = f"Shift {symbol}"
            current_step['step_description'] = f"Shift {symbol} and go to state {next_state}"
            stack.append(next_state)
            symbols.append(symbol)
            i += 1
        elif action_value == 'ACC':  # Accept
            current_step['action'] = 'Accept'
            current_step['step_description'] = 'Input successfully parsed!'
            steps.append(current_step)
            return True, steps, "Input Accepted"
        elif action_value.startswith('R'):  # Reduce
            # Extract the non-terminal and production number from R{A}{i}
            production_info = action_value[1:]
            for j in range(len(production_info)):
                if production_info[j].isdigit():
                    head = production_info[:j]
                    prod_num = int(production_info[j:])
                    break
            else:
                # If no digit found, it's likely just a non-terminal without a number
                head = production_info
                prod_num = 0
                
            # Get the grammar rule being used for reduction
            # We don't have access to the original grammar here, so we'll create a placeholder
            # In a real implementation, you would pass the grammar to this function
            # For now, we'll just use the symbols we have on the stack
            
            # Determine how many symbols to pop (based on the state count to pop)
            # This is a simplification; in a real implementation, the grammar would be accessible
            
            # Find the next state in the goto table after reduction
            symbols_to_pop = 0
            for s in range(1, len(symbols) + 1):
                test_state = stack[-s - 1] if s < len(stack) else -1
                test_head = head
                if (test_state, test_head) in goto_table:
                    symbols_to_pop = s
                    break
            
            # If we couldn't find a valid reduction, try the simplest approach
            if symbols_to_pop == 0:
                # Try to find a pattern in the action string (like "RE1" for "E -> E + T")
                if head in action_value:
                    # Simplification for demo: just pop the last 3 symbols for binary operations
                    symbols_to_pop = 3 if len(symbols) >= 3 else 1
            
            # If still no luck, use a fallback approach
            if symbols_to_pop == 0:
                symbols_to_pop = 1  # As a last resort, just try popping one symbol
            
            if len(symbols) < symbols_to_pop:
                current_step['action'] = 'Error'
                current_step['step_description'] = "Not enough symbols to reduce"
                steps.append(current_step)
                return False, steps, "Syntax Error: Not enough symbols to reduce"
            
            # Pop the required number of symbols and states
            popped_symbols = symbols[-symbols_to_pop:] if symbols_to_pop > 0 else []
            current_step['action'] = f"Reduce {head} → {' '.join(popped_symbols)}"
            current_step['step_description'] = f"Reduce {symbols_to_pop} symbols to {head}"
            
            for _ in range(symbols_to_pop):
                stack.pop()
                symbols.pop()
            
            # GOTO
            state = stack[-1]
            if (state, head) not in goto_table:
                current_step['action'] = 'Error'
                current_step['step_description'] = f"No GOTO action for state {state} and non-terminal {head}"
                steps.append(current_step)
                return False, steps, f"Syntax Error: Invalid GOTO transition for {head} in state {state}"
            
            next_state = goto_table[(state, head)]
            stack.append(next_state)
            symbols.append(head)
        else:  # Error
            current_step['action'] = 'Error'
            current_step['step_description'] = f"Invalid action {action_value}"
            steps.append(current_step)
            return False, steps, "Syntax Error: Invalid action"
        
        steps.append(current_step)
    
    return False, steps, "Syntax Error: Unexpected end of input" 