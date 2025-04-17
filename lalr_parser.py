import pandas as pd
import graphviz
from collections import defaultdict

# Import utility functions from CLR parser
from clr_parser import (
    parse_grammar_clr, compute_first_sets_clr, compute_follow_sets_clr, 
    first_of_string_clr
)

class LR1Item_lalr:
    def __init__(self, head, body, dot, lookahead):
        self.head = head
        self.body = body if isinstance(body, list) else body.split()
        self.dot = dot
        self.lookahead = lookahead
    
    def __eq__(self, other):
        if not isinstance(other, LR1Item_lalr):
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
    
    def next_symbol(self):
        if self.dot < len(self.body):
            return self.body[self.dot]
        return None
    
    def is_complete(self):
        return self.dot >= len(self.body)
    
    def advance_dot(self):
        return LR1Item_lalr(self.head, self.body, self.dot + 1, self.lookahead)

class LALRParser:
    def __init__(self, grammar_rules):
        self.grammar, self.terminals, self.non_terminals = parse_grammar_clr('\n'.join(grammar_rules))
        self.first = compute_first_sets_clr(self.grammar, self.terminals, self.non_terminals)
        self.follow = compute_follow_sets_clr(self.grammar, self.terminals, self.non_terminals, self.first)
        self.action, self.goto_table, self.states, self.state_map = self.construct_lalr_table()

    def lr1_closure(self, item_set):
        result = set(item_set)
        queue = list(item_set)
        while queue:
            item = queue.pop(0)
            B = item.next_symbol()
            if B and B in self.grammar:
                beta_str = ' '.join(item.body[item.dot+1:])
                beta_first = first_of_string_clr(beta_str, self.first)
                lookaheads = set()
                if '' in beta_first:
                    beta_first.remove('')
                    lookaheads.add(item.lookahead)
                lookaheads.update(beta_first)
                for gamma in self.grammar[B]:
                    gamma_symbols = gamma.split() if gamma else []
                    for b in lookaheads:
                        new_item = LR1Item_lalr(B, gamma_symbols, 0, b)
                        if new_item not in result:
                            result.add(new_item)
                            queue.append(new_item)
        return result

    def lr1_goto(self, item_set, symbol):
        advanced_items = {item.advance_dot() for item in item_set if item.next_symbol() == symbol}
        if advanced_items:
            return self.lr1_closure(advanced_items)
        return set()

    def construct_lalr_table(self):
        augmented_start = "S'"
        original_start = next(iter(self.grammar.keys()))
        augmented_grammar = self.grammar.copy()
        augmented_grammar[augmented_start] = [original_start]
        all_non_terminals = self.non_terminals | {augmented_start}
        all_symbols = self.terminals | all_non_terminals
        start_item = LR1Item_lalr(augmented_start, [original_start], 0, '$')
        initial_state = self.lr1_closure({start_item})
        states = [initial_state]
        state_map = {frozenset(initial_state): 0}
        action = {}
        goto_table = {}
        queue = [initial_state]
        while queue:
            current_state = queue.pop(0)
            state_idx = state_map[frozenset(current_state)]
            for item in current_state:
                if not item.is_complete() and item.next_symbol() in self.terminals:
                    symbol = item.next_symbol()
                    next_state = self.lr1_goto(current_state, symbol)
                    if next_state:
                        if frozenset(next_state) not in state_map:
                            state_map[frozenset(next_state)] = len(states)
                            states.append(next_state)
                            queue.append(next_state)
                        next_state_idx = state_map[frozenset(next_state)]
                        action[(state_idx, symbol)] = f"S{next_state_idx}"
                elif item.is_complete():
                    if item.head == augmented_start:
                        action[(state_idx, '$')] = "ACC"
                    else:
                        body_str = ' '.join(item.body) if item.body else ''
                        for i, prod in enumerate(augmented_grammar[item.head]):
                            if prod == body_str:
                                action[(state_idx, item.lookahead)] = f"R{item.head}{i}"
                                break
            for symbol in all_non_terminals:
                next_state = self.lr1_goto(current_state, symbol)
                if next_state:
                    if frozenset(next_state) not in state_map:
                        state_map[frozenset(next_state)] = len(states)
                        states.append(next_state)
                        queue.append(next_state)
                    next_state_idx = state_map[frozenset(next_state)]
                    goto_table[(state_idx, symbol)] = next_state_idx
        # LALR state merging
        core_to_states = defaultdict(list)
        for i, state in enumerate(states):
            core = frozenset((item.head, tuple(item.body), item.dot) for item in state)
            core_to_states[core].append(i)
        new_states = []
        state_remap = {}
        new_action = {}
        new_goto = {}
        for core, state_indices in core_to_states.items():
            merged_items = set()
            for idx in state_indices:
                for item in states[idx]:
                    merged_items.add(LR1Item_lalr(item.head, item.body, item.dot, item.lookahead))
            new_states.append(merged_items)
            for idx in state_indices:
                state_remap[idx] = len(new_states) - 1
        for (state_idx, symbol), act in action.items():
            new_state_idx = state_remap[state_idx]
            new_action[(new_state_idx, symbol)] = act
        for (state_idx, symbol), next_state in goto_table.items():
            new_state_idx = state_remap[state_idx]
            new_next_state = state_remap[next_state]
            new_goto[(new_state_idx, symbol)] = new_next_state
        return new_action, new_goto, new_states, {frozenset(state): i for i, state in enumerate(new_states)}

    def get_tables(self):
        table_data = []
        for state in range(len(self.states)):
            row = {'State': state}
            for terminal in sorted(self.terminals):
                if (state, terminal) in self.action:
                    row[terminal] = str(self.action[(state, terminal)])
                else:
                    row[terminal] = ''
            for non_terminal in sorted(self.non_terminals):
                if (state, non_terminal) in self.goto_table:
                    row[non_terminal] = str(self.goto_table[(state, non_terminal)])
                else:
                    row[non_terminal] = ''
            table_data.append(row)
        return pd.DataFrame(table_data).set_index('State')

    def visualize_dfa(self):
        dot = graphviz.Digraph(comment='DFA for LALR Parser')
        dot.attr(rankdir='LR')
        
        # Add nodes for each state
        for i, state in enumerate(self.states):
            # Group items by non-terminal for better organization
            items_by_nt = defaultdict(list)
            for item in sorted(state, key=lambda x: (x.head, ' '.join(x.body))):
                # Format production with bullet and different colors
                body_with_dot = item.body.copy()
                if item.dot <= len(item.body):
                    body_with_dot.insert(item.dot, "•")
                prod_str = ' '.join(body_with_dot)
                items_by_nt[item.head].append(f"{prod_str}, {item.lookahead}")
            
            # Create node label
            label = f"State {i}\\n"
            label += "─" * 20 + "\\n"  # Separator line
            for nt in sorted(items_by_nt.keys()):
                label += f"{nt} →\\n"
                for prod in sorted(items_by_nt[nt]):
                    label += f"    {prod}\\n"
            
            dot.node(str(i), label, shape='rectangle')
        
        # Add edges for GOTO transitions
        for (state_num, symbol), next_state in self.goto_table.items():
            dot.edge(str(state_num), str(next_state), 
                    label=f"GOTO {symbol}",
                    color='blue')
        
        # Add edges for Shift actions
        for (state_num, symbol), action in self.action.items():
            if isinstance(action, str) and action.startswith('S'):
                next_state = int(action[1:])
                dot.edge(str(state_num), str(next_state),
                        label=f"Shift {symbol}",
                        color='green')
        
        return dot

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
            
            action_value = self.action[(state, symbol)]
            
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
                
                # Find the production in the grammar
                if head in self.grammar and prod_num < len(self.grammar[head]):
                    production = self.grammar[head][prod_num]
                    symbols_to_pop = len(production.split()) if production else 0
                else:
                    # Fallback if we can't find the production (shouldn't happen in a real parser)
                    symbols_to_pop = 3 if len(symbols) >= 3 else 1
                
                if len(symbols) < symbols_to_pop:
                    current_step['action'] = 'Error'
                    current_step['step_description'] = "Not enough symbols to reduce"
                    steps.append(current_step)
                    return False, steps, "Syntax Error: Not enough symbols to reduce"
                
                # Pop the required number of symbols and states
                popped_symbols = symbols[-symbols_to_pop:] if symbols_to_pop > 0 else []
                production_str = " ".join(popped_symbols) if popped_symbols else "ε"
                current_step['action'] = f"Reduce {head} → {production_str}"
                current_step['step_description'] = f"Reduce {symbols_to_pop} symbols to {head}"
                
                for _ in range(symbols_to_pop):
                    stack.pop()
                    symbols.pop()
                
                # GOTO
                state = stack[-1]
                if (state, head) not in self.goto_table:
                    current_step['action'] = 'Error'
                    current_step['step_description'] = f"No GOTO action for state {state} and non-terminal {head}"
                    steps.append(current_step)
                    return False, steps, f"Syntax Error: Invalid GOTO transition for {head} in state {state}"
                
                next_state = self.goto_table[(state, head)]
                stack.append(next_state)
                symbols.append(head)
            else:  # Error
                current_step['action'] = 'Error'
                current_step['step_description'] = f"Invalid action {action_value}"
                steps.append(current_step)
                return False, steps, "Syntax Error: Invalid action"
            
            steps.append(current_step)
        
        return False, steps, "Syntax Error: Unexpected end of input" 