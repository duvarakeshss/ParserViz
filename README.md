Paresr Viz

# LALR Parser Visualization

A Streamlit application for visualizing LALR parsing process and grammar analysis.

## Features

- Define custom context-free grammars or use built-in examples
- Compute and visualize FIRST and FOLLOW sets
- Visualize LALR state machine with transitions
- View ACTION and GOTO parsing tables
- Animated step-by-step parsing visualization
- Support for custom input strings

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the application:
   ```
   streamlit run app.py
   ```

## Usage

1. Select an example grammar or define your own custom grammar
2. Enter an input string to parse
3. Click "Build Parser and Parse Input"
4. Explore the visualization components:
   - State machine graph
   - FIRST and FOLLOW sets
   - Parsing tables
   - Step-by-step parsing animation

## Grammar Format

When defining a custom grammar, use the following format:
```
LHS -> RHS
```

For example:
```
E -> E + T
E -> T
T -> T * F
T -> F
F -> ( E )
F -> id
```

Use `ε` to represent an empty string (epsilon) production:
```
A -> ε
```

## Example Grammars

The application includes several example grammars:

1. **Simple Expression**: A grammar for basic arithmetic expressions
2. **Simple If-Then-Else**: A grammar demonstrating conditional statements
3. **Empty String**: A grammar that includes epsilon productions
