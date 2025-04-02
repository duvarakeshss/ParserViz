class LALR:
    def __init__(self):
        self.states = []
        self.transitions = []
        self.actions = {}
        self.goto = {}
        self.productions = []
        self.terminals = set()
        self.non_terminals = set()
        self.start_symbol = None
        self.grammar = {}
        self.first_sets = {}
        self.follow_sets = {}
        
    def add_production(self, lhs, rhs):
        """Add a production to the grammar"""
        if lhs not in self.grammar:
            self.grammar[lhs] = []
            self.non_terminals.add(lhs)
        
        # Handle epsilon productions (empty string)
        if not rhs:
            self.grammar[lhs].append(["ε"])
        else:
            self.grammar[lhs].append(rhs)
            
        # Add the production to the list
        self.productions.append((lhs, rhs))
        
    def set_start_symbol(self, symbol):
        """Set the start symbol of the grammar"""
        self.start_symbol = symbol
        if symbol not in self.non_terminals:
            self.non_terminals.add(symbol)
            self.grammar[symbol] = []
    
    def compute_first_sets(self):
        """Compute FIRST sets for all symbols"""
        # Initialize FIRST sets
        for terminal in self.terminals:
            self.first_sets[terminal] = {terminal}
        
        for non_terminal in self.non_terminals:
            self.first_sets[non_terminal] = set()
        
        # Special case for epsilon
        self.first_sets["ε"] = {"ε"}
        
        # Make sure all symbols in productions are accounted for
        for lhs, productions in self.grammar.items():
            for production in productions:
                for symbol in production:
                    if symbol not in self.first_sets and symbol != "ε":
                        # If not already in first_sets, it's a terminal
                        self.first_sets[symbol] = {symbol}
                        self.terminals.add(symbol)
        
        # Initialize augmented grammar symbol's FIRST set
        augmented_start = f"{self.start_symbol}'"
        if augmented_start in self.non_terminals:
            self.first_sets[augmented_start] = set()
        
        # Compute FIRST sets
        changed = True
        while changed:
            changed = False
            
            for lhs, productions in self.grammar.items():
                for production in productions:
                    # Handle epsilon production
                    if production == ["ε"]:
                        if "ε" not in self.first_sets[lhs]:
                            self.first_sets[lhs].add("ε")
                            changed = True
                        continue
                    
                    # Compute FIRST for the current production
                    all_can_derive_epsilon = True
                    for i, symbol in enumerate(production):
                        # If symbol is not in first_sets, add it (likely a terminal)
                        if symbol not in self.first_sets:
                            self.first_sets[symbol] = {symbol}
                            self.terminals.add(symbol)
                        
                        # If it's a terminal, add it to FIRST(lhs) and break
                        if symbol in self.terminals:
                            if symbol not in self.first_sets[lhs]:
                                self.first_sets[lhs].add(symbol)
                                changed = True
                            all_can_derive_epsilon = False
                            break
                        
                        # Add FIRST(symbol) - {ε} to FIRST(lhs)
                        for first_sym in self.first_sets[symbol] - {"ε"}:
                            if first_sym not in self.first_sets[lhs]:
                                self.first_sets[lhs].add(first_sym)
                                changed = True
                        
                        # If symbol cannot derive epsilon, break
                        if "ε" not in self.first_sets[symbol]:
                            all_can_derive_epsilon = False
                            break
                    
                    # If all symbols in the production can derive epsilon, add epsilon to FIRST(lhs)
                    if all_can_derive_epsilon and "ε" not in self.first_sets[lhs]:
                        self.first_sets[lhs].add("ε")
                        changed = True
        
        # Set FIRST set for augmented grammar symbol
        if augmented_start in self.non_terminals:
            self.first_sets[augmented_start] = self.first_sets[self.start_symbol]
    
    def compute_follow_sets(self):
        """Compute FOLLOW sets for all non-terminals"""
        # Initialize FOLLOW sets
        for non_terminal in self.non_terminals:
            self.follow_sets[non_terminal] = set()
        
        # Add $ to FOLLOW(start_symbol)
        self.follow_sets[self.start_symbol].add("$")
        
        # Compute FOLLOW sets
        changed = True
        while changed:
            changed = False
            
            for lhs, productions in self.grammar.items():
                for production in productions:
                    if production == ["ε"]:
                        continue
                    
                    for i, symbol in enumerate(production):
                        if symbol in self.non_terminals:
                            # Rule 2: If A -> αBβ, then add FIRST(β) - {ε} to FOLLOW(B)
                            if i < len(production) - 1:
                                beta = production[i+1:]
                                
                                # Compute FIRST(β)
                                first_beta = set()
                                all_can_derive_epsilon = True
                                
                                for beta_symbol in beta:
                                    if beta_symbol in self.first_sets:
                                        # Add FIRST(beta_symbol) - {ε} to first_beta
                                        first_beta.update(self.first_sets[beta_symbol] - {"ε"})
                                        
                                        # If beta_symbol cannot derive epsilon, break
                                        if "ε" not in self.first_sets[beta_symbol]:
                                            all_can_derive_epsilon = False
                                            break
                                    else:
                                        # If beta_symbol is not in first_sets, it's likely a terminal
                                        first_beta.add(beta_symbol)
                                        all_can_derive_epsilon = False
                                        break
                                
                                # Add FIRST(β) - {ε} to FOLLOW(symbol)
                                for first_sym in first_beta:
                                    if first_sym not in self.follow_sets[symbol]:
                                        self.follow_sets[symbol].add(first_sym)
                                        changed = True
                                
                                # Rule 3: If A -> αBβ and β can derive ε, then add FOLLOW(A) to FOLLOW(B)
                                if all_can_derive_epsilon:
                                    if lhs in self.follow_sets:  # Make sure lhs is in follow_sets
                                        for follow_sym in self.follow_sets[lhs]:
                                            if follow_sym not in self.follow_sets[symbol]:
                                                self.follow_sets[symbol].add(follow_sym)
                                                changed = True
                            
                            # Rule 3: If A -> αB, then add FOLLOW(A) to FOLLOW(B)
                            else:
                                if lhs in self.follow_sets:  # Make sure lhs is in follow_sets
                                    for follow_sym in self.follow_sets[lhs]:
                                        if follow_sym not in self.follow_sets[symbol]:
                                            self.follow_sets[symbol].add(follow_sym)
                                            changed = True
    
    def add_terminals_from_grammar(self):
        """Extract terminals from the grammar"""
        for _, productions in self.grammar.items():
            for production in productions:
                for symbol in production:
                    if symbol != "ε" and symbol not in self.non_terminals:
                        self.terminals.add(symbol)
    
    def build_parsing_table(self):
        """Build the LALR parsing table"""
        # Make sure the collections and FOLLOW sets are computed first
        if not self.follow_sets:
            self.compute_first_sets()
            self.compute_follow_sets()
        
        if not self.states:
            self.compute_canonical_collection()
        
        # Now build the parsing table
        for i, state in enumerate(self.states):
            self.actions[i] = {}
            self.goto[i] = {}
            
            for item in state:
                # For items of the form [A -> α.aβ], add shift action
                if not item.is_reduction() and item.get_next_symbol() in self.terminals:
                    next_symbol = item.get_next_symbol()
                    next_state = self.get_transition(i, next_symbol)
                    
                    if next_state is not None:
                        self.actions[i][next_symbol] = ('shift', next_state)
                
                # For items of the form [A -> α.], add reduce action
                elif item.is_reduction():
                    lhs, rhs = item.production
                    
                    # Special case for augmented grammar's accept action
                    if lhs == f"{self.start_symbol}'" and rhs == [self.start_symbol]:
                        self.actions[i]['$'] = ('accept', None)
                    else:
                        prod_idx = self.productions.index((lhs, rhs)) if (lhs, rhs) in self.productions else -1
                        
                        if prod_idx >= 0:
                            for follow_symbol in self.follow_sets.get(lhs, []):
                                self.actions[i][follow_symbol] = ('reduce', prod_idx)
            
            # For non-terminals, add goto actions
            for non_terminal in self.non_terminals:
                next_state = self.get_transition(i, non_terminal)
                if next_state is not None:
                    self.goto[i][non_terminal] = next_state
    
    def compute_canonical_collection(self):
        """Compute the canonical collection of LR(0) items"""
        # Create an augmented grammar with S' -> S
        augmented_start = f"{self.start_symbol}'"
        augmented_production = (augmented_start, [self.start_symbol])
        
        # Add the augmented production to grammar and non_terminals
        if augmented_start not in self.grammar:
            self.grammar[augmented_start] = [[self.start_symbol]]
            self.non_terminals.add(augmented_start)
        
        # Initialize with closure of the starting item
        start_item = Item(augmented_production, 0)
        self.states = [self.closure([start_item])]
        
        # Process all states
        i = 0
        while i < len(self.states):
            # Find the symbols that can follow dots in the current state
            symbols = set()
            for item in self.states[i]:
                next_symbol = item.get_next_symbol()
                if next_symbol:
                    symbols.add(next_symbol)
            
            # For each symbol, compute the goto state
            for symbol in symbols:
                next_state = self.goto_state(self.states[i], symbol)
                
                # If this is a new state, add it to the collection
                if next_state and next_state not in self.states:
                    self.states.append(next_state)
                    self.transitions.append((i, len(self.states) - 1, symbol))
                # If it's an existing state, just add the transition
                elif next_state:
                    state_idx = self.states.index(next_state)
                    self.transitions.append((i, state_idx, symbol))
            
            i += 1
    
    def closure(self, items):
        """Compute the closure of a set of LR(0) items"""
        result = set(items)
        changed = True
        
        while changed:
            changed = False
            new_items = set()
            
            for item in result:
                # If the item is of the form [A -> α.Bβ]
                next_symbol = item.get_next_symbol()
                if next_symbol and next_symbol in self.non_terminals:
                    for production in self.grammar.get(next_symbol, []):
                        new_item = Item((next_symbol, production), 0)
                        if new_item not in result:
                            new_items.add(new_item)
                            changed = True
            
            result.update(new_items)
        
        return result
    
    def goto_state(self, state, symbol):
        """Compute the goto state for a given state and symbol"""
        next_items = set()
        
        for item in state:
            if item.get_next_symbol() == symbol:
                next_items.add(item.advance())
        
        if not next_items:
            return None
        
        return self.closure(next_items)
    
    def get_transition(self, state_idx, symbol):
        """Get the next state for a given state and symbol"""
        for src, dst, sym in self.transitions:
            if src == state_idx and sym == symbol:
                return dst
        return None
    
    def parse(self, tokens):
        """Parse a list of tokens using the LALR parser"""
        # Add end marker if not already present
        if tokens and tokens[-1] != '$':
            tokens = tokens + ['$']
        
        # If tokens is empty, just add the end marker
        if not tokens:
            tokens = ['$']
        
        # Initialize the stack with state 0
        stack = [0]
        i = 0
        
        # For visualization
        parse_steps = []
        
        while True:
            current_state = stack[-1]
            current_token = tokens[i]
            
            # Record the current step
            parse_steps.append({
                'stack': list(stack),
                'input': tokens[i:],
                'action': None
            })
            
            # Check if there's an action for this state and token
            if current_state not in self.actions or current_token not in self.actions[current_state]:
                return False, f"Syntax error at position {i}, unexpected token: {current_token}", parse_steps
            
            action, value = self.actions[current_state][current_token]
            
            # Add the action to the last step
            parse_steps[-1]['action'] = f"{action} {value if value is not None else ''}"
            
            if action == 'shift':
                stack.append(current_token)
                stack.append(value)
                i += 1
            elif action == 'reduce':
                lhs, rhs = self.productions[value]
                
                # Pop 2 * |rhs| items from the stack
                if rhs != ["ε"]:  # Skip popping for epsilon productions
                    for _ in range(2 * len(rhs)):
                        stack.pop()
                
                # Get the state at the top of the stack
                top_state = stack[-1]
                
                # Push the non-terminal and the new state
                stack.append(lhs)
                stack.append(self.goto[top_state][lhs])
            elif action == 'accept':
                return True, "Input accepted", parse_steps
        
        return False, "Parsing failed", parse_steps


class Item:
    def __init__(self, production, dot_position):
        self.production = production
        self.dot_position = dot_position
    
    def __eq__(self, other):
        return (self.production == other.production and
                self.dot_position == other.dot_position)
    
    def __hash__(self):
        # Convert any lists in the production to tuples for hashing
        lhs, rhs = self.production
        if isinstance(rhs, list):
            hashable_rhs = tuple(rhs)
        else:
            hashable_rhs = rhs
        return hash((lhs, hashable_rhs, self.dot_position))
    
    def get_next_symbol(self):
        """Get the symbol following the dot"""
        lhs, rhs = self.production
        if rhs == ["ε"]:
            return None
        
        if self.dot_position < len(rhs):
            return rhs[self.dot_position]
        return None
    
    def is_reduction(self):
        """Check if this item represents a reduction"""
        lhs, rhs = self.production
        if rhs == ["ε"]:
            return True
        return self.dot_position >= len(rhs)
    
    def advance(self):
        """Advance the dot position"""
        return Item(self.production, self.dot_position + 1)
    
    def __str__(self):
        lhs, rhs = self.production
        if rhs == ["ε"]:
            return f"{lhs} → •ε" if self.dot_position == 0 else f"{lhs} → ε•"
        
        result = f"{lhs} → "
        for i, symbol in enumerate(rhs):
            if i == self.dot_position:
                result += "•"
            result += symbol
        
        if self.dot_position == len(rhs):
            result += "•"
        
        return result
    
    def __repr__(self):
        return self.__str__()
        
